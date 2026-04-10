"""Train the lineup value model and run optimization demo."""

import sys, os, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import torch
import pandas as pd
import numpy as np

from baseball_metric.lineup_optimizer.data_builder import (
    build_team_season_features, build_training_tensors,
)
from baseball_metric.lineup_optimizer.model import train_lineup_model
from baseball_metric.lineup_optimizer.optimizer import optimize_lineup


def main():
    print("=" * 70)
    print("  BRAVS LINEUP OPTIMIZER — Training & Demo")
    print("=" * 70)

    # Phase 1: Build training data
    print("\n--- Phase 1: Building training data ---")
    from baseball_metric.data import lahman
    team_features = build_team_season_features(
        "data/bravs_all_seasons.csv",
        str(lahman.DATA_DIR / "Teams.csv"),
    )
    print(f"  Team-season records: {len(team_features)}")
    print(f"  Year range: {team_features.yearID.min()}-{team_features.yearID.max()}")
    print(f"  Avg R/G: {(team_features.R / team_features.G.clip(lower=1)).mean():.2f}")

    # Save for later use
    team_features.to_csv("data/lineup_optimizer/team_features.csv", index=False)

    # Phase 2: Train model
    print("\n--- Phase 2: Training lineup value model ---")
    X, y = build_training_tensors(team_features)
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")

    model = train_lineup_model(X, y, epochs=500, lr=0.001)

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred_mean, pred_std = model.predict(X)
        rmse = ((pred_mean - y) ** 2).mean().sqrt()
        corr = np.corrcoef(pred_mean.cpu().numpy(), y.cpu().numpy())[0, 1]

    print(f"\n  Model performance:")
    print(f"    RMSE: {rmse:.3f} R/G")
    print(f"    Correlation: {corr:.3f}")
    print(f"    Mean prediction: {pred_mean.mean():.2f} R/G")
    print(f"    Mean actual: {y.mean():.2f} R/G")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lineup_value_model.pt")
    print(f"  Model saved to models/lineup_value_model.pt")

    # Phase 3: Demo optimization
    print("\n--- Phase 3: Lineup Optimization Demo ---")

    # Build a sample roster from 2025 data
    seasons = pd.read_csv("data/bravs_all_seasons.csv")
    # Use 2025 Yankees as demo
    nyy_2025 = seasons[(seasons.yearID == 2025) & (seasons.team == "NYA") & (seasons.PA >= 100)]
    if nyy_2025.empty:
        # Try another team
        teams = seasons[seasons.yearID == 2025].team.value_counts()
        best_team = teams.index[0]
        demo_roster = seasons[(seasons.yearID == 2025) & (seasons.team == best_team) & (seasons.PA >= 50)]
        team_name = best_team
    else:
        demo_roster = nyy_2025
        team_name = "NYY"

    if demo_roster.empty:
        print("  No 2025 roster data available for demo")
        return

    # Convert to roster dicts
    roster = []
    for _, r in demo_roster.iterrows():
        roster.append({
            "name": r.get("name", "Unknown"),
            "playerID": r.playerID,
            "position": r.position,
            "hitting_runs": float(r.hitting_runs),
            "baserunning_runs": float(r.baserunning_runs),
            "fielding_runs": float(r.fielding_runs),
            "positional_runs": float(r.positional_runs),
            "aqi_runs": float(r.get("aqi_runs", 0)),
            "HR": int(r.HR),
            "SB": int(r.SB),
            "PA": int(r.PA),
            "bravs_war_eq": float(r.bravs_war_eq),
        })

    print(f"\n  Optimizing lineup for {team_name} 2025 ({len(roster)} players)")

    # Run optimizer
    results = optimize_lineup(roster, n_candidates=50000, top_n=5)

    for i, config in enumerate(results):
        print(f"\n  -- Lineup #{i+1} (expected value: {config.expected_runs:.1f}) --")
        print(config.explanation)

    # Compare top vs worst
    if len(results) >= 2:
        diff = results[0].expected_runs - results[-1].expected_runs
        print(f"\n  Difference between best and 5th-best: {diff:.2f} run-value units")
        print(f"  Over 162 games: ~{diff * 162 / 10:.1f} extra wins from optimal ordering")


if __name__ == "__main__":
    main()
