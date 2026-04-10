"""Phase 1: Training data construction for lineup optimizer.

Builds lineup-outcome training data from Lahman + pre-computed BRAVS.
Since we don't have Retrosheet lineup cards, we use team-season-level
data combined with player BRAVS decompositions to learn lineup value.

The key insight: we don't need actual lineup cards to learn what makes
lineups valuable. We can learn from the relationship between the
BRAVS composition of a roster and the team's actual run production.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_team_season_features(seasons_csv: str, teams_csv: str) -> pd.DataFrame:
    """Build team-season feature vectors from BRAVS decompositions.

    For each team-season:
    - Aggregate player BRAVS components into team-level features
    - Include roster composition metrics (depth, flexibility, platoon capacity)
    - Target: actual team runs scored (from Teams table)

    Returns DataFrame ready for model training.
    """
    log.info("Building team-season training data...")

    seasons = pd.read_csv(seasons_csv)
    from baseball_metric.data import lahman
    teams = pd.read_csv(lahman.DATA_DIR / "Teams.csv")

    # Filter to 2000+ (modern era with reliable data)
    seasons = seasons[seasons.yearID >= 2000]
    teams = teams[teams.yearID >= 2000]

    features = []

    for _, team_row in teams.iterrows():
        yr = int(team_row.yearID)
        tid = team_row.teamID

        # Get all players on this team this year
        roster = seasons[(seasons.yearID == yr) & (seasons.team == tid)]
        if len(roster) < 5:
            continue

        # Position players (batters)
        batters = roster[roster.PA >= 100].sort_values("bravs_war_eq", ascending=False)
        pitchers = roster[roster.IP >= 30].sort_values("bravs_war_eq", ascending=False)

        if len(batters) < 5:
            continue

        # ── LINEUP QUALITY FEATURES ──
        # Top 9 batters (approximate starting lineup)
        top9 = batters.head(9)
        top5_pit = pitchers.head(5)

        feat = {
            "yearID": yr,
            "teamID": tid,

            # Target: actual team runs scored
            "R": int(team_row.get("R", 0) or 0),
            "W": int(team_row.get("W", 0) or 0),
            "L": int(team_row.get("L", 0) or 0),
            "G": int(team_row.get("G", 0) or 0),

            # Lineup hitting quality
            "lineup_hitting_sum": top9.hitting_runs.sum(),
            "lineup_hitting_mean": top9.hitting_runs.mean(),
            "lineup_hitting_std": top9.hitting_runs.std(),
            "lineup_hitting_best": top9.hitting_runs.max(),
            "lineup_hitting_worst": top9.hitting_runs.min(),

            # Lineup baserunning
            "lineup_br_sum": top9.baserunning_runs.sum(),
            "lineup_br_mean": top9.baserunning_runs.mean(),

            # Lineup fielding
            "lineup_fielding_sum": top9.fielding_runs.sum(),

            # Positional value
            "lineup_positional_sum": top9.positional_runs.sum(),

            # Total lineup BRAVS
            "lineup_bravs_sum": top9.bravs_war_eq.sum(),
            "lineup_bravs_mean": top9.bravs_war_eq.mean(),

            # Roster depth: how many useful batters (WAR-eq > 0)?
            "roster_depth": len(batters[batters.bravs_war_eq > 0]),

            # Power concentration: what fraction of HR come from top 3?
            "power_concentration": batters.head(3).HR.sum() / max(batters.HR.sum(), 1),

            # Speed: total SB in lineup
            "lineup_sb_total": top9.SB.sum(),

            # Pitching quality
            "rotation_bravs_sum": top5_pit.bravs_war_eq.sum() if len(top5_pit) > 0 else 0,
            "rotation_bravs_mean": top5_pit.bravs_war_eq.mean() if len(top5_pit) > 0 else 0,

            # Position diversity (how many positions covered by top 9)
            "positions_covered": top9.position.nunique(),
        }

        features.append(feat)

    df = pd.DataFrame(features)
    log.info("Built %d team-season training records", len(df))
    return df


def build_player_flexibility_profiles(seasons_csv: str) -> pd.DataFrame:
    """Build positional flexibility matrix for each player-season.

    Uses the Appearances table to determine how many games each player
    played at each position. Combined with BRAVS fielding component to
    estimate defensive value at each position.
    """
    from baseball_metric.data import lahman
    app = lahman._appearances()
    seasons = pd.read_csv(seasons_csv)

    app = app[app.yearID >= 2000]

    pos_cols = {
        "G_c": "C", "G_1b": "1B", "G_2b": "2B", "G_3b": "3B",
        "G_ss": "SS", "G_lf": "LF", "G_cf": "CF", "G_rf": "RF",
        "G_dh": "DH",
    }

    profiles = []
    for _, row in app.iterrows():
        pid = row.playerID
        yr = int(row.yearID)

        positions_played = {}
        total_g = 0
        for col, pos in pos_cols.items():
            g = int(row.get(col, 0) or 0)
            if g > 0:
                positions_played[pos] = g
                total_g += g

        if total_g < 20:
            continue

        # Multi-position flag
        meaningful_positions = [p for p, g in positions_played.items() if g >= 10]

        profiles.append({
            "playerID": pid,
            "yearID": yr,
            "primary_pos": max(positions_played, key=positions_played.get) if positions_played else "DH",
            "n_positions": len(meaningful_positions),
            "is_multi_position": len(meaningful_positions) >= 2,
            "positions": positions_played,
            "total_games": total_g,
        })

    df = pd.DataFrame(profiles)
    log.info("Built %d player flexibility profiles", len(df))
    return df


def build_training_tensors(team_features: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert team-season features into GPU tensors for training.

    X: feature matrix (n_teams × n_features)
    y: target (runs scored per game)
    """
    feature_cols = [c for c in team_features.columns
                    if c not in ("yearID", "teamID", "R", "W", "L", "G")]

    X = team_features[feature_cols].fillna(0).values.astype(np.float32)
    y = (team_features.R / team_features.G.clip(lower=1)).values.astype(np.float32)  # runs per game

    X_tensor = torch.tensor(X, device=DEVICE)
    y_tensor = torch.tensor(y, device=DEVICE)

    log.info("Training tensors: X=%s, y=%s, device=%s", X_tensor.shape, y_tensor.shape, DEVICE)
    return X_tensor, y_tensor
