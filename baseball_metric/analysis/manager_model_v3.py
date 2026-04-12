"""Manager Value Model v3 — Retrosheet game-level decisions.

v2 used team-season tactical stats. v3 uses game-by-game Retrosheet data
to derive real managerial decisions: one-run game record, extra-innings
record, bullpen aggressiveness (pitchers per game), starting pitcher
hook tendencies, and close-game management.

The key insight: a manager's TRUE value lies in how often they win close
games. Blowouts are decided by talent; one-run and extra-inning games are
decided by bullpen management, pinch-hitting, and matchup optimization.

v3 target: "close game winning percentage above expected"
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ManagerNetV3(nn.Module):
    """Manager value prediction with Retrosheet features."""

    def __init__(self, n_features: int, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_manager_season_features(games: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-game Retrosheet data into manager-season rows.

    Each row = one manager, one season, one team.
    """
    # Build home and away views, then combine
    home = games[["date", "year", "home_team", "home_mgr_id", "home_mgr_name",
                  "home_runs", "vis_runs", "home_won", "run_diff",
                  "extra_innings", "one_run_game", "blowout",
                  "home_pitchers_used", "vis_pitchers_used",
                  "home_sb", "home_cs"]].rename(columns={
        "home_team": "team", "home_mgr_id": "mgr_id", "home_mgr_name": "mgr_name",
        "home_runs": "runs_for", "vis_runs": "runs_against", "home_won": "won",
        "home_pitchers_used": "pitchers_used", "vis_pitchers_used": "opp_pitchers_used",
        "home_sb": "sb", "home_cs": "cs",
    })
    home["is_home"] = 1

    away = games[["date", "year", "vis_team", "vis_mgr_id", "vis_mgr_name",
                  "vis_runs", "home_runs", "home_won", "run_diff",
                  "extra_innings", "one_run_game", "blowout",
                  "vis_pitchers_used", "home_pitchers_used",
                  "vis_sb", "vis_cs"]].rename(columns={
        "vis_team": "team", "vis_mgr_id": "mgr_id", "vis_mgr_name": "mgr_name",
        "vis_runs": "runs_for", "home_runs": "runs_against",
        "vis_pitchers_used": "pitchers_used", "home_pitchers_used": "opp_pitchers_used",
        "vis_sb": "sb", "vis_cs": "cs",
    })
    away["won"] = 1 - away.home_won
    away["is_home"] = 0

    # Both frames already have the same columns after renames
    away = away.drop(columns="home_won", errors="ignore")
    all_games = pd.concat([home, away], ignore_index=True)

    # Filter to valid manager IDs
    all_games = all_games[all_games.mgr_id.str.len() > 0].copy()

    # Aggregate per manager-season-team
    agg = all_games.groupby(["mgr_id", "year", "team"]).agg(
        mgr_name=("mgr_name", "first"),
        games=("date", "count"),
        wins=("won", "sum"),
        runs_for=("runs_for", "sum"),
        runs_against=("runs_against", "sum"),
        one_run_games=("one_run_game", "sum"),
        one_run_wins=("one_run_game", lambda x:
            (all_games.loc[x.index].one_run_game * all_games.loc[x.index].won).sum()),
        extra_inn_games=("extra_innings", "sum"),
        extra_inn_wins=("extra_innings", lambda x:
            (all_games.loc[x.index].extra_innings * all_games.loc[x.index].won).sum()),
        blowout_games=("blowout", "sum"),
        blowout_wins=("blowout", lambda x:
            (all_games.loc[x.index].blowout * all_games.loc[x.index].won).sum()),
        total_pitchers_used=("pitchers_used", "sum"),
        avg_opp_pitchers=("opp_pitchers_used", "mean"),
        sb=("sb", "sum"),
        cs=("cs", "sum"),
    ).reset_index()

    agg = agg[agg.games >= 80].copy()  # full-time managers

    # Derived features
    agg["winpct"] = agg.wins / agg.games
    agg["pyth_winpct"] = agg.runs_for ** 1.83 / (
        agg.runs_for ** 1.83 + agg.runs_against ** 1.83).clip(lower=0.01)
    agg["pyth_expected_W"] = agg.pyth_winpct * agg.games
    agg["pyth_residual"] = agg.wins - agg.pyth_expected_W

    agg["one_run_winpct"] = agg.one_run_wins / agg.one_run_games.clip(lower=1)
    agg["extra_inn_winpct"] = agg.extra_inn_wins / agg.extra_inn_games.clip(lower=1)
    agg["blowout_winpct"] = agg.blowout_wins / agg.blowout_games.clip(lower=1)

    # Close-game residual: one-run + extra-innings actual vs 50% expected
    agg["close_games"] = agg.one_run_games + agg.extra_inn_games
    agg["close_wins"] = agg.one_run_wins + agg.extra_inn_wins
    agg["close_winpct"] = agg.close_wins / agg.close_games.clip(lower=1)
    agg["close_residual"] = agg.close_wins - 0.5 * agg.close_games

    # Bullpen aggressiveness (pitchers per game)
    agg["bullpen_aggr"] = agg.total_pitchers_used / agg.games

    # SB strategy
    agg["sb_attempts"] = agg.sb + agg.cs
    agg["sb_success"] = agg.sb / agg.sb_attempts.clip(lower=1)
    agg["sb_per_game"] = agg.sb_attempts / agg.games

    return agg


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 72)
    print("  MANAGER MODEL v3 — Retrosheet Game-Level Features")
    print(f"  Training on {DEVICE}")
    print("=" * 72)

    print("\n--- Loading Retrosheet game logs ---")
    games = pd.read_csv("data/retrosheet/game_logs_parsed.csv")
    print(f"  {len(games):,} games from {games.year.min()}-{games.year.max()}")

    print("\n--- Building manager-season features ---")
    mgr = build_manager_season_features(games)
    print(f"  {len(mgr)} manager-season-team rows")
    print(f"  Unique managers: {mgr.mgr_id.nunique()}")

    # Merge with BRAVS roster quality
    seasons = pd.read_csv("data/bravs_all_seasons.csv")
    team_war = seasons.groupby(["team", "yearID"]).agg(
        bat_war=("bravs_war_eq", lambda x: seasons.loc[x.index].query("PA >= 50").bravs_war_eq.sum()),
        pit_war=("bravs_war_eq", lambda x: seasons.loc[x.index].query("IP >= 20").bravs_war_eq.sum()),
        top5_war=("bravs_war_eq", lambda x: x.nlargest(5).sum()),
    ).reset_index()
    team_war["total_war"] = team_war.bat_war + team_war.pit_war
    team_war = team_war.rename(columns={"yearID": "year"})

    mgr = mgr.merge(team_war, on=["team", "year"], how="left")

    # Drop rows without BRAVS roster data
    mgr = mgr[mgr.total_war.notna()].copy()
    print(f"  With BRAVS roster data: {len(mgr)}")

    # Career context
    mgr = mgr.sort_values(["mgr_id", "year"])
    mgr["years_experience"] = mgr.groupby("mgr_id").cumcount()
    mgr["career_games"] = mgr.groupby("mgr_id").games.cumsum() - mgr.games

    # Features
    feat_cols = [
        # Roster quality
        "total_war", "bat_war", "pit_war", "top5_war",
        # Run environment
        "runs_for", "runs_against",
        # Tactical
        "bullpen_aggr", "sb_per_game", "sb_success",
        # Career context
        "years_experience", "career_games",
        # Expected baseline
        "pyth_expected_W",
    ]

    valid = mgr[mgr[feat_cols].notna().all(axis=1) & (mgr.games >= 100)].copy()
    print(f"  Valid samples (80+ games, all features): {len(valid)}")

    # TARGET: Pythagorean residual (actual W - expected W from R/RA)
    y = valid.pyth_residual.values.astype(np.float32)
    X = valid[feat_cols].values.astype(np.float32)

    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = np.clip((X - X_mean) / X_std, -5, 5)

    X_t = torch.tensor(X_norm, device=DEVICE)
    y_t = torch.tensor(y, device=DEVICE)

    # Random split (NOT time-based — manager skill is constant across eras)
    n = len(X_t)
    n_val = int(n * 0.20)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    model = ManagerNetV3(len(feat_cols)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)

    print("\n--- Training ---")
    best_val = float("inf")
    best_state = None

    for epoch in range(400):
        model.train()
        batch_size = 128
        epoch_loss = 0
        n_batches = 0
        shuf = train_idx[torch.randperm(len(train_idx))]
        for s in range(0, len(shuf), batch_size):
            idx = shuf[s:s + batch_size]
            pred = model(X_t[idx])
            loss = F.smooth_l1_loss(pred, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                vp = model(X_t[val_idx])
                vl = F.smooth_l1_loss(vp, y_t[val_idx])
                rmse = (vp - y_t[val_idx]).pow(2).mean().sqrt()
                corr = np.corrcoef(vp.cpu().numpy(), y_t[val_idx].cpu().numpy())[0, 1]

            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            print(f"  Epoch {epoch+1:>3}: train={epoch_loss/n_batches:.3f} "
                  f"val_rmse={rmse:.2f} val_r={corr:.3f}")

    if best_state:
        model.load_state_dict(best_state)

    # Full eval
    model.eval()
    with torch.no_grad():
        all_pred = model(X_t).cpu().numpy()

    full_corr = np.corrcoef(all_pred, y)[0, 1]
    full_rmse = np.sqrt(((all_pred - y) ** 2).mean())
    val_corr = np.corrcoef(all_pred[val_idx.cpu()], y[val_idx.cpu()])[0, 1]
    val_rmse = np.sqrt(((all_pred[val_idx.cpu()] - y[val_idx.cpu()]) ** 2).mean())

    print(f"\n  RESULTS:")
    print(f"    Full:  r = {full_corr:.3f}, RMSE = {full_rmse:.2f}")
    print(f"    Val:   r = {val_corr:.3f}, RMSE = {val_rmse:.2f}")

    # Feature importance
    print(f"\n--- Feature Importance ---")
    importances = []
    base = val_rmse
    for i, col in enumerate(feat_cols):
        X_perm = X_norm[val_idx.cpu()].copy()
        np.random.shuffle(X_perm[:, i])
        X_perm_t = torch.tensor(X_perm, device=DEVICE)
        with torch.no_grad():
            pp = model(X_perm_t).cpu().numpy()
        imp = np.sqrt(((pp - y[val_idx.cpu()]) ** 2).mean()) - base
        importances.append((col, imp))
    importances.sort(key=lambda x: -x[1])
    for col, imp in importances:
        bar = "#" * max(0, int(imp * 50))
        print(f"  {col:<22} +{imp:.3f}  {bar}")

    # MANAGER VALUE = actual residual minus predicted residual
    # What the model CAN'T explain from roster quality + tactical trends is
    # pure managerial skill.
    valid["predicted_pyth_resid"] = all_pred
    valid["mgr_value"] = valid.pyth_residual - valid.predicted_pyth_resid

    # Add close-game component as a second signal
    # Close-game residual is a known manager signal (bullpen + pinch hitting)
    valid["mgr_value_combined"] = (
        0.7 * valid.mgr_value + 0.3 * valid.close_residual / 5.0
    )

    # Career aggregates
    careers = valid.groupby("mgr_id").agg(
        name=("mgr_name", "first"),
        seasons=("year", "count"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        games=("games", "sum"),
        wins=("wins", "sum"),
        pyth_expected=("pyth_expected_W", "sum"),
        one_run_games=("one_run_games", "sum"),
        one_run_wins=("one_run_wins", "sum"),
        extra_inn_games=("extra_inn_games", "sum"),
        extra_inn_wins=("extra_inn_wins", "sum"),
        bullpen_aggr=("bullpen_aggr", "mean"),
        mgr_value_total=("mgr_value", "sum"),
        mgr_value_combined=("mgr_value_combined", "sum"),
        mgr_value_per_season=("mgr_value", "mean"),
        close_residual_total=("close_residual", "sum"),
    ).reset_index()

    careers["one_run_winpct"] = careers.one_run_wins / careers.one_run_games.clip(lower=1)
    careers["winpct"] = careers.wins / careers.games.clip(lower=1)
    careers_q = careers[careers.seasons >= 3].copy()

    print(f"\n{'=' * 72}")
    print("  BEST MANAGERS (2010-2025) — v3 Combined Score")
    print(f"{'=' * 72}")
    best = careers_q.sort_values("mgr_value_combined", ascending=False)
    print(f'{"#":>3} {"Name":<24} {"Yrs":>4} {"W-L":<10} '
          f'{"1-Run%":>7} {"Pen/G":>6} {"Value":>7}')
    print("-" * 70)
    for i, (_, r) in enumerate(best.head(25).iterrows()):
        wl = f"{int(r.wins)}-{int(r.games - r.wins)}"
        print(f'{i+1:>3} {r["name"]:<24} {int(r.seasons):>4} {wl:<10} '
              f'{r.one_run_winpct:>6.1%} {r.bullpen_aggr:>6.2f} '
              f'{r.mgr_value_combined:>+7.1f}')

    print(f"\n--- WORST MANAGERS ---")
    worst = careers_q.sort_values("mgr_value_combined")
    for i, (_, r) in enumerate(worst.head(10).iterrows()):
        wl = f"{int(r.wins)}-{int(r.games - r.wins)}"
        print(f'  {r["name"]:<24} {int(r.seasons)} yrs, {wl:<10} '
              f'1-run={r.one_run_winpct:.1%} value={r.mgr_value_combined:+.1f}')

    # Best single seasons
    print(f"\n--- BEST SINGLE SEASONS ---")
    valid_sorted = valid.sort_values("mgr_value_combined", ascending=False)
    print(f'{"Name":<22} {"Yr":>4} {"Tm":<4} {"W":>3} '
          f'{"Pyth+":>6} {"1-Run":>7} {"Value":>7}')
    for _, r in valid_sorted.head(15).iterrows():
        one_run_pct = r.one_run_wins / r.one_run_games if r.one_run_games > 0 else 0
        print(f'{r["mgr_name"]:<22} {int(r.year):>4} {r.team:<4} '
              f'{int(r.wins):>3} {r.pyth_residual:>+6.1f} '
              f'{one_run_pct:>6.1%} {r.mgr_value_combined:>+7.2f}')

    # Active managers
    print(f"\n--- ACTIVE MANAGERS (2023-2025) ---")
    recent = valid[valid.year >= 2023]
    recent_c = recent.groupby("mgr_id").agg(
        name=("mgr_name", "first"),
        seasons=("year", "count"),
        mgr_value=("mgr_value_combined", "sum"),
        one_run_wins=("one_run_wins", "sum"),
        one_run_games=("one_run_games", "sum"),
        pyth_residual=("pyth_residual", "sum"),
        bullpen_aggr=("bullpen_aggr", "mean"),
    ).reset_index()
    recent_c["one_run_winpct"] = recent_c.one_run_wins / recent_c.one_run_games.clip(lower=1)
    recent_c = recent_c.sort_values("mgr_value", ascending=False)

    print(f'{"Name":<24} {"Yrs":>4} {"1-Run%":>7} {"Pen/G":>6} {"Pyth+":>7} {"Value":>7}')
    print("-" * 62)
    for _, r in recent_c.head(25).iterrows():
        print(f'{r["name"]:<24} {int(r.seasons):>4} '
              f'{r.one_run_winpct:>6.1%} {r.bullpen_aggr:>6.2f} '
              f'{r.pyth_residual:>+7.1f} {r.mgr_value:>+7.1f}')

    # Save
    os.makedirs("models", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "feat_cols": feat_cols,
        "X_mean": X_mean.tolist(),
        "X_std": X_std.tolist(),
        "n_params": n_params,
        "val_rmse": float(val_rmse),
        "val_corr": float(val_corr),
    }, "models/manager_model_v3.pt")

    careers_q.to_csv("data/manager_careers_v3.csv", index=False)
    valid.to_csv("data/manager_seasons_v3.csv", index=False)

    print(f"\n  Saved models/manager_model_v3.pt ({n_params:,} params)")
    print(f"  Saved data/manager_careers_v3.csv")
    print(f"  Saved data/manager_seasons_v3.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
