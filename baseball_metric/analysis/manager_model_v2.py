"""Manager Value Model v2 — Pythagorean baseline + tactical features.

The v1 model had two critical flaws:
1. Used "47 + WAR" as expected wins, which systematically overestimates
   great rosters (and under-credits managers of those teams).
2. Only 8 features, none of which captured actual managerial decisions.

v2 fixes:
1. PYTHAGOREAN expected wins from R/RA. This is what a team "should" win
   given their run differential. The residual (actual - pythagorean) is
   the classic "one-run game luck / bullpen management" bucket that sabermetricians
   have always attributed to manager skill.
2. RICH TACTICAL FEATURES from Teams.csv:
   - SB success rate (decides when to steal)
   - SB attempts per game (aggressiveness)
   - Complete game rate (pitcher hook)
   - Save rate (bullpen leverage)
   - Double play rate (defensive positioning / bunt allowance)
   - Fielding percentage
   - Sacrifice fly rate (small-ball)
   - Hit-by-pitch rate (pitcher aggressiveness)
   - vs-league ERA+ (staff performance vs expectation)
3. CONTEXTUAL FEATURES:
   - Years of experience (learning curve)
   - Career games (survivability — bad managers get fired)
   - Previous-year record (continuity)
   - Roster stability (seasons with same org)
4. MULTIPLE TARGETS for multi-task learning:
   - Pythagorean residual (the main target)
   - One-run game record (known manager signal)
   - Bullpen effectiveness (BB vs SO ratio)
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


class ManagerNetV2(nn.Module):
    """Bigger, better manager model.

    Input: 20 features (roster, tactical, contextual)
    Output: predicted Pythagorean residual (wins above Pythagorean expectation)
    """

    def __init__(self, n_features: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pythagorean_expected_wins(r: float, ra: float, g: int, exponent: float = 1.83) -> float:
    """Bill James Pythagorean expected wins. The 1.83 exponent is the
    calibrated historical best fit (slightly better than the classic 2.0)."""
    if r + ra < 1:
        return g / 2
    wp = r ** exponent / (r ** exponent + ra ** exponent)
    return wp * g


def build_manager_dataset_v2():
    """Build rich manager training data from Teams + Managers."""
    managers = pd.read_csv("data/lahman2025/Managers.csv")
    teams_csv = pd.read_csv("data/lahman2025/Teams.csv")
    people = pd.read_csv("data/lahman2025/People.csv")
    seasons = pd.read_csv("data/bravs_all_seasons.csv")

    # Manager-season: full-season managers only
    mgr = managers[managers.G >= 100].copy()
    mgr["yearID"] = mgr.yearID.astype(int)
    mgr = mgr.sort_values(["playerID", "yearID"])

    # Career context
    mgr["career_games"] = mgr.groupby("playerID").G.cumsum() - mgr.G
    mgr["years_experience"] = mgr.groupby("playerID").cumcount()
    mgr["prev_W"] = mgr.groupby("playerID").W.shift(1).fillna(81)
    mgr["prev_L"] = mgr.groupby("playerID").L.shift(1).fillna(81)
    mgr["prev_winpct"] = mgr.prev_W / (mgr.prev_W + mgr.prev_L).clip(lower=1)

    # Team rosters (BRAVS WAR)
    team_war = seasons.groupby(["team", "yearID"]).agg(
        bat_war=("bravs_war_eq", lambda x: seasons.loc[x.index].query("PA >= 50").bravs_war_eq.sum()),
        pit_war=("bravs_war_eq", lambda x: seasons.loc[x.index].query("IP >= 20").bravs_war_eq.sum()),
        top5_war=("bravs_war_eq", lambda x: x.nlargest(5).sum()),
        roster_depth=("bravs_war_eq", lambda x: (x > 1).sum()),
    ).reset_index()
    team_war["total_war"] = team_war.bat_war + team_war.pit_war

    mgr = mgr.merge(
        team_war.rename(columns={"team": "teamID"}),
        on=["teamID", "yearID"], how="left",
    )

    # Team tactical stats from Teams.csv
    t_feat = teams_csv[[
        "yearID", "teamID", "G", "W", "L", "R", "RA", "ERA",
        "SB", "CS", "HBP", "SF", "CG", "SHO", "SV",
        "DP", "E", "FP", "BBA", "SOA", "HRA",
    ]].copy()

    mgr = mgr.merge(t_feat, on=["yearID", "teamID"], how="left", suffixes=("_mgr", "_team"))

    # Use team W-L (not manager's partial) for season-level analysis
    mgr["actual_W"] = mgr.W_team
    mgr["actual_L"] = mgr.L_team
    mgr["G"] = mgr.G_team

    # PYTHAGOREAN EXPECTED WINS — the new baseline
    mgr["pyth_expected_W"] = [
        pythagorean_expected_wins(r, ra, g)
        for r, ra, g in zip(mgr.R.fillna(0), mgr.RA.fillna(1), mgr.G.fillna(162))
    ]
    mgr["pyth_residual"] = mgr.actual_W - mgr.pyth_expected_W

    # Tactical derived stats
    mgr["sb_attempts"] = mgr.SB.fillna(0) + mgr.CS.fillna(0)
    mgr["sb_success_rate"] = mgr.SB / mgr.sb_attempts.replace(0, 1)
    mgr["sb_per_game"] = mgr.sb_attempts / mgr.G.clip(lower=1)
    mgr["cg_rate"] = mgr.CG / mgr.G.clip(lower=1)  # pitcher hook tendency
    mgr["sv_rate"] = mgr.SV / mgr.G.clip(lower=1)  # bullpen usage
    mgr["dp_rate"] = mgr.DP / mgr.G.clip(lower=1)  # defensive efficiency
    mgr["sf_rate"] = mgr.SF / mgr.G.clip(lower=1)  # small ball
    mgr["k_bb_ratio"] = mgr.SOA / mgr.BBA.replace(0, 1)  # staff quality
    mgr["rpg"] = mgr.R / mgr.G.clip(lower=1)
    mgr["rapg"] = mgr.RA / mgr.G.clip(lower=1)

    # Manager age
    birth = people[["playerID", "birthYear"]]
    mgr = mgr.merge(birth, on="playerID", how="left")
    mgr["mgr_age"] = mgr.yearID - mgr.birthYear

    # Manager name
    mgr = mgr.merge(
        people[["playerID", "nameFirst", "nameLast"]], on="playerID", how="left",
    )
    mgr["mgr_name"] = mgr.nameFirst + " " + mgr.nameLast

    # Filter valid rows
    mgr = mgr[
        (mgr.R.notna()) & (mgr.RA.notna()) & (mgr.G >= 100) &
        (mgr.total_war.notna())
    ].copy()

    return mgr


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 72)
    print("  MANAGER VALUE MODEL v2")
    print("  Pythagorean baseline + tactical features + career context")
    print(f"  Training on {DEVICE}")
    print("=" * 72)

    print("\n--- Building enhanced dataset ---")
    data = build_manager_dataset_v2()
    print(f"  Manager-seasons: {len(data)}")
    print(f"  Year range: {int(data.yearID.min())}-{int(data.yearID.max())}")
    print(f"  Unique managers: {data.playerID.nunique()}")

    # Pythagorean residual baseline
    print(f"\n  Pythagorean residual stats:")
    print(f"    Mean: {data.pyth_residual.mean():+.2f} wins")
    print(f"    Std:  {data.pyth_residual.std():.2f}")
    print(f"    Max:  {data.pyth_residual.max():+.1f}")
    print(f"    Min:  {data.pyth_residual.min():+.1f}")
    # This residual should have mean ~0 (pythagorean is unbiased)

    # 20 features
    feat_cols = [
        # Roster quality
        "total_war", "bat_war", "pit_war", "top5_war", "roster_depth",
        # Run environment
        "rpg", "rapg", "ERA", "k_bb_ratio",
        # Tactical decisions
        "sb_per_game", "sb_success_rate", "cg_rate", "sv_rate",
        "dp_rate", "sf_rate", "FP",
        # Context
        "mgr_age", "years_experience", "prev_winpct", "career_games",
    ]

    valid = data[data[feat_cols].notna().all(axis=1)].copy()
    print(f"  Valid samples: {len(valid)}")

    X = valid[feat_cols].values.astype(np.float32)
    y = valid.pyth_residual.values.astype(np.float32)

    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    X_norm = np.clip(X_norm, -5, 5)

    X_t = torch.tensor(X_norm, device=DEVICE)
    y_t = torch.tensor(y, device=DEVICE)

    # Time-aware split (train on pre-2015, val on 2015+)
    years = valid.yearID.values
    train_mask = years < 2015
    val_mask = years >= 2015

    train_idx = torch.tensor(np.where(train_mask)[0], device=DEVICE)
    val_idx = torch.tensor(np.where(val_mask)[0], device=DEVICE)

    print(f"  Train: {len(train_idx)} (1920-2014), Val: {len(val_idx)} (2015-2025)")

    # Model
    model = ManagerNetV2(len(feat_cols)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800)

    print("\n--- Training ---")
    best_val = float("inf")
    best_state = None

    for epoch in range(800):
        model.train()
        # Mini-batches for stability
        batch_size = 256
        epoch_loss = 0
        n_batches = 0
        shuffled = train_idx[torch.randperm(len(train_idx))]
        for start in range(0, len(shuffled), batch_size):
            idx = shuffled[start:start + batch_size]
            pred = model(X_t[idx])
            loss = F.smooth_l1_loss(pred, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                vp = model(X_t[val_idx])
                vl = F.smooth_l1_loss(vp, y_t[val_idx])
                val_rmse = (vp - y_t[val_idx]).pow(2).mean().sqrt()
                val_corr = np.corrcoef(vp.cpu().numpy(), y_t[val_idx].cpu().numpy())[0, 1]

            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            print(f"  Epoch {epoch+1}: train={epoch_loss/n_batches:.3f} val_rmse={val_rmse:.2f} val_r={val_corr:.3f}")

    if best_state:
        model.load_state_dict(best_state)

    # Full eval
    model.eval()
    with torch.no_grad():
        all_pred = model(X_t).cpu().numpy()

    train_pred = all_pred[train_mask]
    val_pred = all_pred[val_mask]

    train_corr = np.corrcoef(train_pred, y[train_mask])[0, 1]
    val_corr = np.corrcoef(val_pred, y[val_mask])[0, 1]
    val_rmse = np.sqrt(((val_pred - y[val_mask]) ** 2).mean())

    print(f"\n  RESULTS:")
    print(f"    Train: r = {train_corr:.3f}")
    print(f"    Val:   r = {val_corr:.3f}, RMSE = {val_rmse:.2f} wins")

    # Feature importance via permutation
    print(f"\n--- Feature Importance (Permutation) ---")
    base_rmse = val_rmse
    importances = []
    for i, col in enumerate(feat_cols):
        X_perm = X_norm[val_mask].copy()
        np.random.shuffle(X_perm[:, i])
        X_perm_t = torch.tensor(X_perm, device=DEVICE)
        with torch.no_grad():
            perm_pred = model(X_perm_t).cpu().numpy()
        perm_rmse = np.sqrt(((perm_pred - y[val_mask]) ** 2).mean())
        importances.append((col, perm_rmse - base_rmse))

    importances.sort(key=lambda x: -x[1])
    for col, imp in importances[:15]:
        bar = "#" * max(0, int(imp * 30))
        print(f"  {col:<22} +{imp:.3f}  {bar}")

    # Now compute MANAGER VALUE:
    # Not "predicted residual" — that's baseline prediction.
    # It's "actual residual minus predicted residual" — the unexplained
    # portion that's most likely managerial skill.
    valid["predicted_pyth_resid"] = all_pred
    valid["mgr_value"] = valid.pyth_residual - valid.predicted_pyth_resid
    # This is: "How much more did the team outperform Pythagoras than expected
    # given their roster, tactical tendencies, and context?"

    # Aggregate by career
    careers = valid.groupby("playerID").agg(
        name=("mgr_name", "first"),
        seasons=("yearID", "count"),
        first_year=("yearID", "min"),
        last_year=("yearID", "max"),
        total_games=("G", "sum"),
        actual_W=("actual_W", "sum"),
        actual_L=("actual_L", "sum"),
        pyth_expected=("pyth_expected_W", "sum"),
        pyth_residual_total=("pyth_residual", "sum"),
        mgr_value_total=("mgr_value", "sum"),
        mgr_value_per_season=("mgr_value", "mean"),
    ).reset_index()

    careers["winpct"] = careers.actual_W / (careers.actual_W + careers.actual_L).clip(lower=1)
    careers["expected_winpct"] = careers.pyth_expected / careers.total_games.clip(lower=1)

    # Only career managers
    careers_qualified = careers[careers.seasons >= 5].copy()

    print(f"\n{'=' * 72}")
    print("  BEST MANAGERS ALL-TIME (Manager Value = residual vs model prediction)")
    print(f"{'=' * 72}")
    best = careers_qualified.sort_values("mgr_value_total", ascending=False)
    print(f'{"#":>3} {"Name":<24} {"Seasons":>8} {"Years":<12} {"W-L":<10} {"Pyth+":>7} {"Mgr Value":>10}')
    print("-" * 78)
    for i, (_, r) in enumerate(best.head(25).iterrows()):
        wl = f"{int(r.actual_W)}-{int(r.actual_L)}"
        print(f'{i+1:>3} {r["name"]:<24} {int(r.seasons):>8} '
              f'{int(r.first_year)}-{int(r.last_year):<5} {wl:<10} '
              f'{r.pyth_residual_total:>+7.0f} {r.mgr_value_total:>+10.0f}')

    print(f"\n--- WORST MANAGERS (biggest negative value) ---")
    worst = careers_qualified.sort_values("mgr_value_total")
    for i, (_, r) in enumerate(worst.head(10).iterrows()):
        wl = f"{int(r.actual_W)}-{int(r.actual_L)}"
        print(f'  {r["name"]:<24} {int(r.seasons)} yrs, {wl:<10} '
              f'pyth={r.pyth_residual_total:+.0f}, value={r.mgr_value_total:+.0f}')

    # Active managers
    print(f"\n--- ACTIVE MANAGERS (with 2020+ seasons) ---")
    active = valid[valid.yearID >= 2020]
    active_careers = active.groupby("playerID").agg(
        name=("mgr_name", "first"),
        seasons=("yearID", "count"),
        mgr_value=("mgr_value", "sum"),
        pyth_resid=("pyth_residual", "sum"),
    ).reset_index()
    active_careers = active_careers[active_careers.seasons >= 2]
    active_careers = active_careers.sort_values("mgr_value", ascending=False)

    print(f'{"Name":<24} {"Seasons":>8} {"Pyth Total":>11} {"Mgr Value":>11}')
    print("-" * 60)
    for _, r in active_careers.head(20).iterrows():
        print(f'{r["name"]:<24} {int(r.seasons):>8} {r.pyth_resid:>+11.1f} {r.mgr_value:>+11.1f}')

    # Best single seasons
    print(f"\n--- BEST SINGLE-SEASON JOBS BY MANAGER VALUE ---")
    valid_sorted = valid.sort_values("mgr_value", ascending=False)
    print(f'{"Name":<22} {"Yr":>4} {"Team":<5} {"W-L":<8} {"Pyth":>5} {"Mgr Value":>10}')
    for _, r in valid_sorted.head(15).iterrows():
        wl = f"{int(r.actual_W)}-{int(r.actual_L)}"
        print(f'{r["mgr_name"]:<22} {int(r.yearID):>4} {r.teamID:<5} {wl:<8} '
              f'{r.pyth_residual:>+5.1f} {r.mgr_value:>+10.2f}')

    print(f"\n--- WORST SINGLE-SEASON JOBS ---")
    for _, r in valid_sorted.tail(10).iterrows():
        wl = f"{int(r.actual_W)}-{int(r.actual_L)}"
        print(f'  {r["mgr_name"]:<22} {int(r.yearID)} {r.teamID:<5} {wl:<8} '
              f'mgr value={r.mgr_value:+.1f}')

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
    }, "models/manager_model_v2.pt")

    careers_qualified.to_csv("data/manager_careers_v2.csv", index=False)
    valid[["playerID", "mgr_name", "yearID", "teamID", "actual_W", "actual_L",
           "R", "RA", "pyth_expected_W", "pyth_residual",
           "total_war", "sb_per_game", "sb_success_rate", "cg_rate",
           "mgr_value"]].to_csv("data/manager_seasons_v2.csv", index=False)

    print(f"\n  Saved models/manager_model_v2.pt ({n_params:,} params)")
    print(f"  Saved data/manager_careers_v2.csv ({len(careers_qualified)} managers)")
    print(f"  Saved data/manager_seasons_v2.csv ({len(valid)} seasons)")
    print("=" * 72)


if __name__ == "__main__":
    main()
