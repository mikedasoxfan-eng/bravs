"""PyTorch Neural Network for Prospect MLB Projection.

Upgrades the GBM model with:
- Deeper feature engineering (level progression speed, age-relative-to-level)
- Neural network with uncertainty estimation
- GPU acceleration
- Better handling of the heavily right-skewed target (most prospects bust)
"""

from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEVEL_ORDER = {"RK": 0, "A-": 1, "A": 2, "A+": 3, "AA": 4, "AAA": 5, "WIN": -1}


class ProspectNet(nn.Module):
    """Neural network predicting MLB career WAR from MiLB stats.

    Architecture:
    - Input: 25+ features (MiLB stats, age, level progression)
    - 3 hidden layers with ReLU, dropout, batch norm
    - Dual output heads: mean prediction + log-variance (uncertainty)
    - Trained with Gaussian NLL for calibrated uncertainty
    """

    def __init__(self, n_features: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden // 2, 1)
        self.logvar_head = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1)
        return mean, logvar

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            mean, logvar = self(x)
            std = (logvar / 2).exp()
        return mean, std


def build_advanced_features(milb: pd.DataFrame, mlb_careers: pd.DataFrame,
                            crosswalk: pd.DataFrame, people: pd.DataFrame) -> tuple:
    """Build rich feature set for prospect prediction."""
    id_map = dict(zip(crosswalk.mlbam_id, crosswalk.lahman_id))

    milb_bat = milb[milb.PA > 0].copy()
    milb_bat["pid_int"] = milb_bat["playerID"].astype(float).astype(int)
    milb_bat["lahman_id"] = milb_bat["pid_int"].map(id_map)
    milb_bat = milb_bat[milb_bat.lahman_id.notna()].copy()

    # Get birth years for age computation
    people_sub = people[["playerID", "birthYear"]].drop_duplicates("playerID")
    birth_map = dict(zip(people_sub.playerID, people_sub.birthYear))

    # Aggregate per player with rich features
    players = []

    for lahman_id, group in milb_bat.groupby("lahman_id"):
        group = group.sort_values("yearID")
        birth = birth_map.get(lahman_id)
        if birth and not pd.isna(birth):
            group["age"] = group["yearID"] - int(birth)
        else:
            group["age"] = 22  # default

        # Basic aggregates
        total_pa = group.PA.sum()
        total_war = group.bravs_war_eq.sum()
        peak_war = group.bravs_war_eq.max()
        avg_war = group.bravs_war_eq.mean()
        avg_woba = group.wOBA.mean() if "wOBA" in group.columns else 0
        best_woba = group.wOBA.max() if "wOBA" in group.columns else 0
        n_seasons = len(group)

        # Level progression features
        levels_visited = group.level.map(LEVEL_ORDER).dropna()
        if len(levels_visited) > 0:
            highest_level = levels_visited.max()
            level_speed = highest_level / max(n_seasons, 1)  # how fast they climb
        else:
            highest_level = 0
            level_speed = 0

        # Age-relative-to-level (young for level = better)
        # Average age at each level, compare to league norms
        age_norms = {"RK": 19, "A-": 20, "A": 21, "A+": 22, "AA": 23, "AAA": 25}
        age_vs_level = []
        for _, row in group.iterrows():
            norm_age = age_norms.get(row.level, 22)
            age_diff = row.age - norm_age  # negative = young for level
            age_vs_level.append(age_diff)
        avg_age_vs_level = np.mean(age_vs_level) if age_vs_level else 0

        # First-season age
        first_age = group.age.min()

        # Performance by level
        level_features = {}
        for level, col in [("AAA", "aaa"), ("AA", "aa"), ("A+", "ap"), ("A", "a")]:
            ld = group[group.level == level]
            if len(ld) > 0:
                level_features[f"war_{col}"] = ld.bravs_war_eq.sum()
                level_features[f"woba_{col}"] = ld.wOBA.mean() if "wOBA" in ld.columns else 0
                level_features[f"pa_{col}"] = ld.PA.sum()
            else:
                level_features[f"war_{col}"] = 0
                level_features[f"woba_{col}"] = 0
                level_features[f"pa_{col}"] = 0

        # Trend: improving or declining?
        if n_seasons >= 2:
            war_trend = np.polyfit(range(n_seasons), group.bravs_war_eq.values, 1)[0]
        else:
            war_trend = 0

        # Power and speed
        total_hr = group.HR.sum() if "HR" in group.columns else 0
        total_sb = group.SB.sum() if "SB" in group.columns else 0
        hr_rate = total_hr / max(total_pa, 1) * 600
        sb_rate = total_sb / max(total_pa, 1) * 600

        hitting_runs = group.hitting_runs.sum() if "hitting_runs" in group.columns else 0

        feat = {
            "lahman_id": lahman_id,
            "name": group.name.iloc[0] if "name" in group.columns else "?",
            "total_pa": total_pa,
            "total_war": total_war,
            "peak_war": peak_war,
            "avg_war": avg_war,
            "avg_woba": avg_woba,
            "best_woba": best_woba,
            "n_seasons": n_seasons,
            "highest_level": highest_level,
            "level_speed": level_speed,
            "avg_age_vs_level": avg_age_vs_level,
            "first_age": first_age,
            "war_trend": war_trend,
            "hr_rate": hr_rate,
            "sb_rate": sb_rate,
            "hitting_runs": hitting_runs,
            **level_features,
        }
        players.append(feat)

    players_df = pd.DataFrame(players)

    # Link to MLB career outcomes
    mlb_careers["playerID"] = mlb_careers["playerID"].astype(str)
    linked = players_df.merge(
        mlb_careers[["playerID", "career_war_eq", "peak_bravs", "peak5_bravs", "seasons"]],
        left_on="lahman_id", right_on="playerID", how="inner",
    )

    feature_cols = [c for c in linked.columns if c not in
                    ["lahman_id", "name", "playerID", "career_war_eq", "peak_bravs", "peak5_bravs", "seasons"]]

    X = linked[feature_cols].fillna(0).values.astype(np.float32)
    y = linked["career_war_eq"].fillna(0).values.astype(np.float32)
    names = linked["name"].values

    return X, y, feature_cols, names, linked


def train_prospect_net(X: np.ndarray, y: np.ndarray, epochs: int = 300) -> ProspectNet:
    """Train the neural network on GPU."""
    n = X.shape[0]
    n_val = int(n * 0.15)

    # Shuffle
    perm = np.random.RandomState(42).permutation(n)
    X, y = X[perm], y[perm]

    X_train = torch.tensor(X[n_val:], device=DEVICE)
    y_train = torch.tensor(y[n_val:], device=DEVICE)
    X_val = torch.tensor(X[:n_val], device=DEVICE)
    y_val = torch.tensor(y[:n_val], device=DEVICE)

    model = ProspectNet(X.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        mean, logvar = model(X_train)
        # Gaussian NLL loss
        loss = 0.5 * (logvar + (y_train - mean) ** 2 / logvar.exp()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_mean, val_logvar = model(X_val)
                val_loss = 0.5 * (val_logvar + (y_val - val_mean) ** 2 / val_logvar.exp()).mean()
                val_rmse = ((val_mean - y_val) ** 2).mean().sqrt()
                val_corr = np.corrcoef(val_mean.cpu().numpy(), y_val.cpu().numpy())[0, 1]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            log.info("Epoch %d: train=%.4f val=%.4f rmse=%.2f r=%.3f",
                     epoch + 1, loss.item(), val_loss.item(), val_rmse.item(), val_corr)

    if best_state:
        model.load_state_dict(best_state)
    return model


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 70)
    print("  BRAVS Prospect Neural Network")
    print(f"  Training on {DEVICE}")
    print("=" * 70)

    milb = pd.read_csv("data/bravs_milb_seasons.csv")
    mlb_careers = pd.read_csv("data/bravs_careers.csv")
    crosswalk = pd.read_csv("data/id_crosswalk.csv")
    people = pd.read_csv("data/lahman2025/People.csv")

    print("\n--- Building features ---")
    X, y, feature_cols, names, linked = build_advanced_features(milb, mlb_careers, crosswalk, people)
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target mean: {y.mean():.1f} WAR, std: {y.std():.1f}")

    print("\n--- Training neural network ---")
    t0 = time.perf_counter()
    model = train_prospect_net(X, y, epochs=500)
    elapsed = time.perf_counter() - t0
    print(f"  Training: {elapsed:.1f}s")

    # Full evaluation
    model.eval()
    X_tensor = torch.tensor(X, device=DEVICE)
    with torch.no_grad():
        pred_mean, pred_std = model.predict(X_tensor)

    pred = pred_mean.cpu().numpy()
    corr = np.corrcoef(pred, y)[0, 1]
    rmse = np.sqrt(((pred - y) ** 2).mean())
    print(f"\n  Full dataset:")
    print(f"    Correlation: r = {corr:.4f}")
    print(f"    RMSE: {rmse:.1f} WAR")

    # Feature importance via permutation
    print("\n  Feature importance (permutation):")
    base_rmse = rmse
    importances = []
    for i, col in enumerate(feature_cols):
        X_perm = X.copy()
        np.random.shuffle(X_perm[:, i])
        X_perm_t = torch.tensor(X_perm, device=DEVICE)
        with torch.no_grad():
            perm_pred, _ = model.predict(X_perm_t)
        perm_rmse = np.sqrt(((perm_pred.cpu().numpy() - y) ** 2).mean())
        imp = perm_rmse - base_rmse
        importances.append((col, imp))

    importances.sort(key=lambda x: -x[1])
    for col, imp in importances[:12]:
        print(f"    {col:<25} +{imp:.3f} RMSE")

    # Validate known prospects
    print("\n--- Known Prospect Validation ---")
    known = {"Mike Trout": 85.4, "Juan Soto": 35.0, "Ronald Acuna": 25.0,
             "Bryce Harper": 62.5, "Mookie Betts": 67.7, "Julio Rodriguez": 10.0}
    for name, actual in known.items():
        mask = linked.name.str.contains(name.split()[-1], na=False)
        match = linked[mask].sort_values("career_war_eq", ascending=False)
        if len(match) > 0:
            idx = match.index[0]
            row_x = torch.tensor(X[linked.index.get_loc(idx)].reshape(1, -1), device=DEVICE)
            with torch.no_grad():
                p_mean, p_std = model.predict(row_x)
            print(f"  {name:<22} Pred={p_mean.item():+.1f} +/- {p_std.item():.1f}, "
                  f"Actual={actual:.1f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/prospect_net.pt")
    print(f"\n  Model saved to models/prospect_net.pt")

    # Project current prospects
    print("\n--- Current Prospect Projections (2022-2026 MiLB) ---")
    milb_recent = milb[(milb.yearID >= 2022) & (milb.PA > 0)].copy()
    milb_recent["pid_int"] = milb_recent["playerID"].astype(float).astype(int)

    people_birth = dict(zip(people.playerID, people.birthYear))
    crosswalk_map = dict(zip(crosswalk.mlbam_id, crosswalk.lahman_id))

    # Build features for each prospect
    prospect_feats = []
    for pid, group in milb_recent.groupby("pid_int"):
        lahman = crosswalk_map.get(pid)
        group = group.sort_values("yearID")

        birth = people_birth.get(lahman) if lahman else None
        if birth and not pd.isna(birth):
            group = group.copy()
            group["age"] = group["yearID"] - int(birth)
        else:
            group = group.copy()
            group["age"] = 22

        total_pa = group.PA.sum()
        if total_pa < 100:
            continue

        # Same feature engineering as training
        levels_visited = group.level.map(LEVEL_ORDER).dropna()
        highest_level = levels_visited.max() if len(levels_visited) > 0 else 0
        n_seasons = len(group)
        level_speed = highest_level / max(n_seasons, 1)

        age_norms = {"RK": 19, "A-": 20, "A": 21, "A+": 22, "AA": 23, "AAA": 25}
        age_diffs = [row.age - age_norms.get(row.level, 22) for _, row in group.iterrows()]
        avg_age_vs_level = np.mean(age_diffs) if age_diffs else 0

        war_trend = np.polyfit(range(n_seasons), group.bravs_war_eq.values, 1)[0] if n_seasons >= 2 else 0

        feat = {col: 0 for col in feature_cols}
        feat["total_pa"] = total_pa
        feat["total_war"] = group.bravs_war_eq.sum()
        feat["peak_war"] = group.bravs_war_eq.max()
        feat["avg_war"] = group.bravs_war_eq.mean()
        feat["avg_woba"] = group.wOBA.mean() if "wOBA" in group.columns else 0
        feat["best_woba"] = group.wOBA.max() if "wOBA" in group.columns else 0
        feat["n_seasons"] = n_seasons
        feat["highest_level"] = highest_level
        feat["level_speed"] = level_speed
        feat["avg_age_vs_level"] = avg_age_vs_level
        feat["first_age"] = group.age.min()
        feat["war_trend"] = war_trend
        feat["hr_rate"] = (group.HR.sum() if "HR" in group.columns else 0) / max(total_pa, 1) * 600
        feat["sb_rate"] = (group.SB.sum() if "SB" in group.columns else 0) / max(total_pa, 1) * 600
        feat["hitting_runs"] = group.hitting_runs.sum() if "hitting_runs" in group.columns else 0

        for level, col in [("AAA", "aaa"), ("AA", "aa"), ("A+", "ap"), ("A", "a")]:
            ld = group[group.level == level]
            if len(ld) > 0:
                feat[f"war_{col}"] = ld.bravs_war_eq.sum()
                feat[f"woba_{col}"] = ld.wOBA.mean() if "wOBA" in ld.columns else 0
                feat[f"pa_{col}"] = ld.PA.sum()

        feat["_name"] = group.name.iloc[0]
        feat["_pid"] = pid
        feat["_in_mlb"] = lahman in set(mlb_careers.playerID.astype(str)) if lahman else False
        prospect_feats.append(feat)

    prospect_df = pd.DataFrame(prospect_feats)
    X_prospect = prospect_df[feature_cols].fillna(0).values.astype(np.float32)
    X_pt = torch.tensor(X_prospect, device=DEVICE)

    with torch.no_grad():
        p_mean, p_std = model.predict(X_pt)

    prospect_df["projected_war"] = p_mean.cpu().numpy().round(1)
    prospect_df["projection_std"] = p_std.cpu().numpy().round(1)
    prospect_df = prospect_df.sort_values("projected_war", ascending=False)

    print(f"\n  Top 25 Prospects by Neural Net Projection:")
    print(f"  {'#':<4} {'Name':<24} {'Proj WAR':>9} {'Uncert':>7} {'Level':>6} {'wOBA':>6} {'PA':>5}")
    for i, (_, r) in enumerate(prospect_df.head(25).iterrows()):
        mlb_flag = "*" if r._in_mlb else " "
        print(f"  {i+1:<4} {str(r._name):<24} {r.projected_war:>+9.1f} {r.projection_std:>7.1f} "
              f"{int(r.highest_level):>6} {r.avg_woba:>6.3f} {int(r.total_pa):>5}{mlb_flag}")

    # Save
    prospect_df[["_name", "_pid", "projected_war", "projection_std", "total_pa",
                  "avg_woba", "highest_level", "_in_mlb"]].to_csv(
        "data/prospect_rankings_nn.csv", index=False
    )
    print(f"\n  Saved data/prospect_rankings_nn.csv")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
