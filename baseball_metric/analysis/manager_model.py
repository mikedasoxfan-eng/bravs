"""Manager Value Model — quantifies manager contribution to wins.

Core idea: a manager's "value" = actual wins minus expected wins given roster talent.
Train a neural net on the residual: what about a manager predicts outperformance?

Features: manager age, years of experience, player retention rate, bullpen usage
patterns, in-game substitution tendencies (to the extent we can derive them).

Output: career WAR-above-expected (WAR-AE), best/worst managers ever,
and a "manager score" that teams could use to evaluate hiring decisions.
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


class ManagerNet(nn.Module):
    """Neural network predicting win delta from manager + roster features.

    Input: 12 features (team WAR, bat/pit split, manager experience, age, etc.)
    Output: predicted win delta (actual - expected wins)
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_manager_dataset():
    """Build training data: manager-seasons with team rosters and outcomes."""
    managers = pd.read_csv("data/lahman2025/Managers.csv")
    seasons = pd.read_csv("data/bravs_all_seasons.csv")
    teams_csv = pd.read_csv("data/lahman2025/Teams.csv")
    people = pd.read_csv("data/lahman2025/People.csv")

    # Filter to full-season managers (at least 100 games)
    mgr = managers[managers.G >= 100].copy()
    mgr["yearID"] = mgr.yearID.astype(int)

    # Build manager experience: career game count up to that year
    mgr = mgr.sort_values(["playerID", "yearID"])
    mgr["career_games"] = mgr.groupby("playerID").G.cumsum() - mgr.G
    mgr["years_experience"] = mgr.groupby("playerID").cumcount()

    # Get manager birth year for age
    birth = people[["playerID", "birthYear"]].rename(columns={"playerID": "mgr_id"})
    mgr = mgr.merge(birth.rename(columns={"mgr_id": "playerID"}), on="playerID", how="left")
    mgr["mgr_age"] = mgr.yearID - mgr.birthYear

    # Compute team WAR for each season
    team_war = seasons.groupby(["team", "yearID"]).agg(
        bat_war=("bravs_war_eq", lambda x: seasons.loc[x.index].query("PA >= 50").bravs_war_eq.sum()),
        pit_war=("bravs_war_eq", lambda x: seasons.loc[x.index].query("IP >= 20").bravs_war_eq.sum()),
        top5_war=("bravs_war_eq", lambda x: x.nlargest(5).sum()),
        roster_depth=("bravs_war_eq", lambda x: (x > 1).sum()),
    ).reset_index()
    team_war["total_war"] = team_war.bat_war + team_war.pit_war

    # Merge with manager data
    mgr = mgr.merge(
        team_war.rename(columns={"team": "teamID"}),
        on=["teamID", "yearID"], how="left"
    )

    # Expected wins from WAR
    mgr["expected_w"] = 47 + mgr.total_war.fillna(0)
    mgr["win_delta"] = mgr.W - mgr.expected_w

    # Filter to manager-seasons with roster data
    mgr = mgr[mgr.total_war.notna() & (mgr.G >= 100)].copy()

    # Get manager name
    people_name = people[["playerID", "nameFirst", "nameLast"]]
    mgr = mgr.merge(people_name, on="playerID", how="left")
    mgr["mgr_name"] = mgr.nameFirst + " " + mgr.nameLast

    return mgr


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 70)
    print("  BRAVS MANAGER MODEL")
    print(f"  Training on {DEVICE}")
    print("=" * 70)

    print("\n--- Building manager dataset ---")
    data = build_manager_dataset()
    print(f"  Manager-seasons: {len(data)}")
    print(f"  Unique managers: {data.playerID.nunique()}")
    print(f"  Year range: {int(data.yearID.min())}-{int(data.yearID.max())}")

    # Features
    feat_cols = [
        "total_war", "bat_war", "pit_war", "top5_war", "roster_depth",
        "mgr_age", "years_experience", "career_games",
    ]

    valid = data[data[feat_cols].notna().all(axis=1)].copy()
    print(f"  Valid samples: {len(valid)}")

    X = valid[feat_cols].values.astype(np.float32)
    y = valid.win_delta.values.astype(np.float32)

    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    X_t = torch.tensor(X_norm, device=DEVICE)
    y_t = torch.tensor(y, device=DEVICE)

    # Split
    n = len(X_t)
    n_val = int(n * 0.15)
    perm = torch.randperm(n)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    model = ManagerNet(len(feat_cols)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} parameters")

    print("\n--- Training ---")
    best_val = float("inf")
    best_state = None

    for epoch in range(500):
        model.train()
        pred = model(X_t[train_idx])
        loss = F.smooth_l1_loss(pred, y_t[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                vp = model(X_t[val_idx])
                vl = F.smooth_l1_loss(vp, y_t[val_idx])
                rmse = (vp - y_t[val_idx]).pow(2).mean().sqrt()

            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            print(f"  Epoch {epoch+1}: loss={loss.item():.3f} val={vl.item():.3f} rmse={rmse.item():.2f} wins")

    if best_state:
        model.load_state_dict(best_state)

    # Full eval
    model.eval()
    with torch.no_grad():
        all_pred = model(X_t).cpu().numpy()

    corr = np.corrcoef(all_pred, y)[0, 1]
    rmse = np.sqrt(((all_pred - y) ** 2).mean())
    print(f"\n  Full dataset:")
    print(f"    r = {corr:.3f}, RMSE = {rmse:.2f} wins")

    # Residuals = manager value above what the model predicts
    # (the model learns a baseline; the residual is pure manager skill)
    valid["predicted_delta"] = all_pred
    valid["mgr_value"] = valid.win_delta - valid.predicted_delta

    # Career manager value
    mgr_careers = valid.groupby("playerID").agg(
        name=("mgr_name", "first"),
        seasons=("yearID", "count"),
        first_year=("yearID", "min"),
        last_year=("yearID", "max"),
        total_games=("G", "sum"),
        total_wins=("W", "sum"),
        total_expected=("expected_w", "sum"),
        total_actual_delta=("win_delta", "sum"),
        total_mgr_value=("mgr_value", "sum"),
        avg_mgr_value=("mgr_value", "mean"),
    ).reset_index()

    mgr_careers["career_delta"] = mgr_careers.total_wins - mgr_careers.total_expected
    mgr_careers = mgr_careers[mgr_careers.seasons >= 3]

    print(f"\n--- BEST MANAGERS ALL-TIME (wins above expected, career) ---")
    best = mgr_careers.sort_values("career_delta", ascending=False)
    print(f'{"#":>3} {"Name":<24} {"Seasons":>8} {"Years":<12} {"Wins":>5} {"Expected":>9} {"Delta":>6}')
    print("-" * 70)
    for i, (_, r) in enumerate(best.head(20).iterrows()):
        print(f'{i+1:>3} {r["name"]:<24} {int(r.seasons):>8} '
              f'{int(r.first_year)}-{int(r.last_year):<5} '
              f'{int(r.total_wins):>5} {int(r.total_expected):>9} {int(r.career_delta):>+6}')

    print(f"\n--- WORST MANAGERS (biggest negative delta) ---")
    worst = mgr_careers.sort_values("career_delta")
    for i, (_, r) in enumerate(worst.head(10).iterrows()):
        print(f'  {r["name"]:<24} {int(r.seasons)} yrs, '
              f'{int(r.total_wins)}W vs {int(r.total_expected)} expected '
              f'({int(r.career_delta):+d})')

    # Best single seasons
    print(f"\n--- BEST SINGLE-SEASON MANAGERIAL JOBS ---")
    best_seasons = valid.sort_values("win_delta", ascending=False)
    print(f'{"Name":<22} {"Year":>4} {"Team":<5} {"W":>3} {"Exp":>4} {"Delta":>6} {"Team WAR":>9}')
    print("-" * 60)
    for _, r in best_seasons.head(15).iterrows():
        print(f'{r["mgr_name"]:<22} {int(r.yearID):>4} {r.teamID:<5} '
              f'{int(r.W):>3} {int(r.expected_w):>4} {int(r.win_delta):>+6} {r.total_war:>+9.1f}')

    # Worst single seasons
    print(f"\n--- WORST SINGLE-SEASON JOBS (managed underachievers) ---")
    worst_seasons = valid.sort_values("win_delta")
    for _, r in worst_seasons.head(10).iterrows():
        print(f'  {r["mgr_name"]:<22} {int(r.yearID)} {r.teamID:<4} '
              f'{int(r.W)}W vs {int(r.expected_w)} expected ({int(r.win_delta):+d}) '
              f'team WAR={r.total_war:+.1f}')

    # Save
    os.makedirs("models", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "feat_cols": feat_cols,
        "X_mean": X_mean.tolist(),
        "X_std": X_std.tolist(),
    }, "models/manager_model.pt")

    mgr_careers.to_csv("data/manager_careers.csv", index=False)
    valid[["playerID", "mgr_name", "yearID", "teamID", "G", "W", "L",
           "total_war", "expected_w", "win_delta", "mgr_value"]].to_csv(
        "data/manager_seasons.csv", index=False
    )

    print(f"\n  Saved models/manager_model.pt")
    print(f"  Saved data/manager_careers.csv ({len(mgr_careers)} managers)")
    print(f"  Saved data/manager_seasons.csv ({len(valid)} manager-seasons)")

    # Current managers (2025)
    print(f"\n--- ACTIVE MANAGERS (2020-2025) ---")
    recent = valid[valid.yearID >= 2020]
    recent_careers = recent.groupby("playerID").agg(
        name=("mgr_name", "first"),
        seasons=("yearID", "count"),
        total_delta=("win_delta", "sum"),
        avg_delta=("win_delta", "mean"),
    ).reset_index()
    recent_careers = recent_careers[recent_careers.seasons >= 2]
    recent_careers = recent_careers.sort_values("total_delta", ascending=False)

    print(f'{"Name":<24} {"Seasons":>8} {"Total Δ":>8} {"Per Yr":>7}')
    print("-" * 52)
    for _, r in recent_careers.head(15).iterrows():
        print(f'{r["name"]:<24} {int(r.seasons):>8} {int(r.total_delta):>+8} {r.avg_delta:>+7.1f}')

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
