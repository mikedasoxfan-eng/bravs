"""Win Probability Model — predict team wins from BRAVS components.

Uses historical team-seasons to learn the relationship between
BRAVS batting/pitching WAR and actual win totals. Incorporates
Pythagorean expectation, run differential, and BRAVS components.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WinModel(nn.Module):
    """Neural network predicting team wins from BRAVS roster composition."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_team_features(seasons: pd.DataFrame, teams_csv: pd.DataFrame,
                        year_range: tuple = (2000, 2025)) -> pd.DataFrame:
    """Build team-season features from BRAVS data."""
    features = []

    for year in range(year_range[0], year_range[1] + 1):
        yr = seasons[seasons.yearID == year]
        teams = yr.team.dropna().unique()

        for team in teams:
            td = yr[yr.team == team]
            batters = td[td.PA >= 50]
            pitchers = td[td.IP >= 20]

            if len(batters) < 5:
                continue

            bat_war = batters.bravs_war_eq.sum()
            pit_war = pitchers.bravs_war_eq.sum()

            # Component breakdowns
            hit_runs = batters.hitting_runs.sum()
            br_runs = batters.baserunning_runs.sum()
            fld_runs = batters.fielding_runs.sum()
            pit_runs = pitchers.get("pitching_runs", pd.Series(dtype=float)).sum()

            # Roster depth
            n_pos_above_avg = (batters.bravs_war_eq > 1.0).sum()
            n_pit_above_avg = (pitchers.bravs_war_eq > 1.0).sum()
            bat_std = batters.bravs_war_eq.std()
            best_batter = batters.bravs_war_eq.max()
            worst_batter = batters.bravs_war_eq.min()

            # Actual record
            actual = teams_csv[(teams_csv.yearID == year) & (teams_csv.teamID == team)]
            if len(actual) == 0:
                continue
            actual_w = int(actual.iloc[0].W)
            actual_l = int(actual.iloc[0].L)
            actual_r = int(actual.iloc[0].get("R", 0) or 0)
            actual_ra = int(actual.iloc[0].get("RA", 0) or 0)

            features.append({
                "team": team, "year": year,
                "bat_war": bat_war, "pit_war": pit_war,
                "total_war": bat_war + pit_war,
                "hit_runs": hit_runs, "br_runs": br_runs, "fld_runs": fld_runs,
                "n_pos_above_avg": n_pos_above_avg,
                "n_pit_above_avg": n_pit_above_avg,
                "bat_std": bat_std,
                "best_batter": best_batter,
                "worst_batter": worst_batter,
                "n_batters": len(batters),
                "n_pitchers": len(pitchers),
                "actual_w": actual_w,
                "actual_r": actual_r,
                "actual_ra": actual_ra,
            })

    return pd.DataFrame(features)


def train_win_model(df: pd.DataFrame) -> tuple:
    """Train win prediction model."""
    feature_cols = ["bat_war", "pit_war", "hit_runs", "br_runs", "fld_runs",
                    "n_pos_above_avg", "n_pit_above_avg", "bat_std",
                    "best_batter", "worst_batter", "n_batters", "n_pitchers"]

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df["actual_w"].values.astype(np.float32)

    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    X_t = torch.tensor(X_norm, device=DEVICE)
    y_t = torch.tensor(y, device=DEVICE)

    # Train/val split
    n = len(X)
    n_val = int(n * 0.15)
    perm = torch.randperm(n)
    X_train, y_train = X_t[perm[n_val:]], y_t[perm[n_val:]]
    X_val, y_val = X_t[perm[:n_val]], y_t[perm[:n_val]]

    model = WinModel(len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    best_val = float("inf")
    best_state = None

    for epoch in range(500):
        model.train()
        pred = model(X_train)
        loss = ((pred - y_train) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = ((val_pred - y_val) ** 2).mean()
                val_rmse = val_loss.sqrt()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            log.info("Epoch %d: train_rmse=%.2f val_rmse=%.2f",
                     epoch + 1, loss.sqrt().item(), val_rmse.item())

    if best_state:
        model.load_state_dict(best_state)

    # Full evaluation
    model.eval()
    with torch.no_grad():
        all_pred = model(X_t).cpu().numpy()

    corr = np.corrcoef(all_pred, y)[0, 1]
    rmse = np.sqrt(((all_pred - y) ** 2).mean())

    return model, feature_cols, X_mean, X_std, corr, rmse


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 70)
    print("  BRAVS WIN PREDICTION MODEL")
    print(f"  Training on {DEVICE}")
    print("=" * 70)

    seasons = pd.read_csv("data/bravs_all_seasons.csv")
    teams_csv = pd.read_csv("data/lahman2025/Teams.csv")

    print("\n--- Building team features ---")
    df = build_team_features(seasons, teams_csv, (2000, 2025))
    print(f"  Team-seasons: {len(df)}")

    print("\n--- Training win model ---")
    model, feature_cols, X_mean, X_std, corr, rmse = train_win_model(df)
    print(f"\n  Results:")
    print(f"    Correlation: r = {corr:.4f}")
    print(f"    RMSE: {rmse:.1f} wins")

    # Compare to simple WAR+47 baseline
    simple_pred = 47 + df.total_war
    simple_rmse = np.sqrt(((simple_pred - df.actual_w) ** 2).mean())
    simple_corr = np.corrcoef(simple_pred, df.actual_w)[0, 1]
    print(f"\n  Baseline (47 + total WAR):")
    print(f"    Correlation: r = {simple_corr:.4f}")
    print(f"    RMSE: {simple_rmse:.1f} wins")

    # Pythagorean baseline
    if "actual_r" in df.columns and "actual_ra" in df.columns:
        valid = df[(df.actual_r > 0) & (df.actual_ra > 0)]
        pyth_wp = valid.actual_r ** 2 / (valid.actual_r ** 2 + valid.actual_ra ** 2)
        pyth_w = pyth_wp * (valid.actual_w + (162 - valid.actual_w))  # approximate games
        pyth_corr = np.corrcoef(pyth_wp, valid.actual_w)[0, 1]
        print(f"\n  Pythagorean baseline:")
        print(f"    Correlation: r = {pyth_corr:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "X_mean": X_mean.tolist(),
        "X_std": X_std.tolist(),
    }, "models/win_model.pt")
    print(f"\n  Model saved to models/win_model.pt")

    # 2025 predictions
    print(f"\n--- 2025 Win Predictions ---")
    df_2025 = df[df.year == 2025].sort_values("actual_w", ascending=False)

    X_2025 = df_2025[feature_cols].fillna(0).values.astype(np.float32)
    X_2025_norm = (X_2025 - X_mean) / (X_std + 1e-8)
    X_2025_t = torch.tensor(X_2025_norm, device=DEVICE)

    model.eval()
    with torch.no_grad():
        pred_2025 = model(X_2025_t).cpu().numpy()

    df_2025 = df_2025.copy()
    df_2025["predicted_w"] = pred_2025.round().astype(int)

    print(f'  {"Team":<5} {"Pred W":>7} {"Act W":>6} {"Diff":>5} {"WAR":>6}')
    print("  " + "-" * 32)
    for _, r in df_2025.iterrows():
        diff = int(r.predicted_w - r.actual_w)
        print(f'  {r.team:<5} {int(r.predicted_w):>7} {r.actual_w:>6} {diff:>+5} {r.total_war:>+6.1f}')

    # Save predictions
    df.to_csv("data/team_win_predictions.csv", index=False)
    print(f"\n  Saved data/team_win_predictions.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
