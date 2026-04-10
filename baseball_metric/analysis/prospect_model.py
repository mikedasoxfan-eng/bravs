"""MiLB-to-MLB Prospect Projection Model.

Uses historical MiLB BRAVS performance to predict MLB outcomes.
Links minor league player-seasons to their eventual MLB careers
and builds a translation model.

Key insight: the translation rate from MiLB to MLB depends on:
- Level (AAA translates better than Rk)
- Age relative to level (young for level = better prospect)
- Rate stats vs counting stats (wOBA translates, raw HR don't)
- Position (catchers translate differently than corner OF)
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

log = logging.getLogger(__name__)

# Level hierarchy (higher = closer to MLB)
LEVEL_ORDER = {"RK": 0, "A-": 1, "A": 2, "A+": 3, "AA": 4, "AAA": 5}


def load_milb_mlb_linked() -> pd.DataFrame:
    """Load MiLB BRAVS and link to eventual MLB careers.

    Returns DataFrame with MiLB seasons joined to MLB career outcomes.
    """
    milb = pd.read_csv("data/bravs_milb_seasons.csv")
    mlb_careers = pd.read_csv("data/bravs_careers.csv")

    # Only batting prospects (position players)
    milb_bat = milb[milb.PA > 0].copy()

    # Get the Chadwick register to link MiLB playerIDs to MLB playerIDs
    # Both use MLB Advanced Media IDs (key_mlbam), so playerID should match
    # Load ID crosswalk (MLBAM -> Lahman)
    crosswalk = pd.read_csv("data/id_crosswalk.csv")
    id_map = dict(zip(crosswalk.mlbam_id, crosswalk.lahman_id))

    # Map MiLB MLBAM IDs to Lahman IDs
    milb_bat["playerID_mlbam"] = milb_bat["playerID"].astype(float).astype(int)
    milb_bat["playerID"] = milb_bat["playerID_mlbam"].map(id_map)
    milb_bat = milb_bat[milb_bat.playerID.notna()].copy()

    mlb_careers["playerID"] = mlb_careers["playerID"].astype(str)

    log.info("MiLB batters with Lahman ID: %d", len(milb_bat))

    # Aggregate MiLB career stats per player
    milb_career = milb_bat.groupby("playerID").agg(
        milb_seasons=("yearID", "count"),
        milb_first_year=("yearID", "min"),
        milb_last_year=("yearID", "max"),
        milb_total_pa=("PA", "sum"),
        milb_total_hr=("HR", "sum"),
        milb_avg_war=("bravs_war_eq", "mean"),
        milb_total_war=("bravs_war_eq", "sum"),
        milb_peak_war=("bravs_war_eq", "max"),
        milb_avg_woba=("wOBA", "mean"),
        milb_best_woba=("wOBA", "max"),
        milb_hitting_runs=("hitting_runs", "sum"),
        milb_highest_level=("level", lambda x: max(x, key=lambda l: LEVEL_ORDER.get(l, -1))),
        name=("name", "first"),
    ).reset_index()

    # Get best season at each level
    for level in ["AAA", "AA", "A+"]:
        level_data = milb_bat[milb_bat.level == level].groupby("playerID").agg(
            **{f"war_{level.lower().replace('+','p')}": ("bravs_war_eq", "max"),
               f"woba_{level.lower().replace('+','p')}": ("wOBA", "max"),
               f"pa_{level.lower().replace('+','p')}": ("PA", "sum")}
        ).reset_index()
        milb_career = milb_career.merge(level_data, on="playerID", how="left")

    # Link to MLB career outcomes
    linked = milb_career.merge(
        mlb_careers[["playerID", "career_war_eq", "total_PA", "total_HR",
                      "peak_bravs", "peak5_bravs", "seasons", "first_year", "last_year"]],
        on="playerID",
        how="inner",
        suffixes=("_milb", "_mlb"),
    )

    # Age at MiLB debut (estimate from first MiLB year vs MLB debut)
    linked["milb_debut_age"] = linked["milb_first_year"] - 18  # rough estimate
    linked["years_in_minors"] = linked["first_year"] - linked["milb_first_year"]

    log.info("Linked %d players with both MiLB and MLB careers", len(linked))
    return linked


def build_prospect_features(linked: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix for prospect projection model.

    Features:
    - MiLB BRAVS stats (total WAR, peak WAR, avg wOBA)
    - Level-specific performance (AAA WAR, AA WAR, A+ WAR)
    - Age/development indicators (years in minors, highest level)
    - Position category

    Target: MLB career WAR-eq
    """
    features = pd.DataFrame({
        "milb_total_war": linked["milb_total_war"].fillna(0),
        "milb_peak_war": linked["milb_peak_war"].fillna(0),
        "milb_avg_war": linked["milb_avg_war"].fillna(0),
        "milb_avg_woba": linked["milb_avg_woba"].fillna(0),
        "milb_best_woba": linked["milb_best_woba"].fillna(0),
        "milb_total_pa": linked["milb_total_pa"].fillna(0),
        "milb_seasons": linked["milb_seasons"].fillna(0),
        "milb_hitting_runs": linked["milb_hitting_runs"].fillna(0),
        "war_aaa": linked.get("war_aaa", pd.Series(0, index=linked.index)).fillna(0),
        "war_aa": linked.get("war_aa", pd.Series(0, index=linked.index)).fillna(0),
        "war_ap": linked.get("war_ap", pd.Series(0, index=linked.index)).fillna(0),
        "woba_aaa": linked.get("woba_aaa", pd.Series(0, index=linked.index)).fillna(0),
        "woba_aa": linked.get("woba_aa", pd.Series(0, index=linked.index)).fillna(0),
        "pa_aaa": linked.get("pa_aaa", pd.Series(0, index=linked.index)).fillna(0),
        "pa_aa": linked.get("pa_aa", pd.Series(0, index=linked.index)).fillna(0),
        "years_in_minors": linked["years_in_minors"].fillna(3).clip(0, 10),
        "highest_level": linked["milb_highest_level"].map(LEVEL_ORDER).fillna(0),
    })

    target = linked["career_war_eq"].fillna(0)

    return features, target


def train_prospect_model(features: pd.DataFrame, target: pd.Series) -> GradientBoostingRegressor:
    """Train GBM to predict MLB career WAR from MiLB stats."""
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )

    # Cross-validation
    scores = cross_val_score(model, features, target, cv=5, scoring="r2")
    log.info("Cross-val R²: %.3f ± %.3f", scores.mean(), scores.std())

    # Fit on all data
    model.fit(features, target)

    return model


def project_current_prospects(
    model: GradientBoostingRegressor,
    feature_names: list[str],
) -> pd.DataFrame:
    """Project MLB career WAR for current MiLB players who haven't reached MLB yet."""
    milb = pd.read_csv("data/bravs_milb_seasons.csv")
    mlb_careers = pd.read_csv("data/bravs_careers.csv")

    milb_bat = milb[milb.PA > 0].copy()

    # Map IDs
    crosswalk = pd.read_csv("data/id_crosswalk.csv")
    id_map = dict(zip(crosswalk.mlbam_id, crosswalk.lahman_id))
    milb_bat["playerID_mlbam"] = milb_bat["playerID"].astype(float).astype(int)
    milb_bat["lahman_id"] = milb_bat["playerID_mlbam"].map(id_map)

    mlb_ids = set(mlb_careers["playerID"].astype(str))

    # Focus on recent years (2022-2026) for current prospects
    recent = milb_bat[milb_bat.yearID >= 2022].copy()
    recent["playerID"] = recent["playerID"].astype(str)  # keep MLBAM ID for grouping

    # Aggregate per player
    prospect_career = recent.groupby("playerID").agg(
        milb_seasons=("yearID", "count"),
        milb_first_year=("yearID", "min"),
        milb_last_year=("yearID", "max"),
        milb_total_pa=("PA", "sum"),
        milb_total_hr=("HR", "sum"),
        milb_avg_war=("bravs_war_eq", "mean"),
        milb_total_war=("bravs_war_eq", "sum"),
        milb_peak_war=("bravs_war_eq", "max"),
        milb_avg_woba=("wOBA", "mean"),
        milb_best_woba=("wOBA", "max"),
        milb_hitting_runs=("hitting_runs", "sum"),
        milb_highest_level=("level", lambda x: max(x, key=lambda l: LEVEL_ORDER.get(l, -1))),
        name=("name", "first"),
    ).reset_index()

    # Build features
    for level, col_prefix in [("AAA", "aaa"), ("AA", "aa"), ("A+", "ap")]:
        level_data = recent[recent.level == level].groupby("playerID").agg(
            **{f"war_{col_prefix}": ("bravs_war_eq", "max"),
               f"woba_{col_prefix}": ("wOBA", "max"),
               f"pa_{col_prefix}": ("PA", "sum")}
        ).reset_index()
        prospect_career = prospect_career.merge(level_data, on="playerID", how="left")

    prospect_career["years_in_minors"] = prospect_career["milb_last_year"] - prospect_career["milb_first_year"] + 1
    prospect_career["highest_level"] = prospect_career["milb_highest_level"].map(LEVEL_ORDER).fillna(0)

    # Build feature matrix matching training features
    X = pd.DataFrame(index=prospect_career.index)
    for col in feature_names:
        if col in prospect_career.columns:
            X[col] = prospect_career[col].fillna(0)
        else:
            X[col] = 0

    # Predict
    prospect_career["projected_mlb_war"] = model.predict(X)
    # Check if player has reached MLB by mapping through crosswalk
    prospect_career["lahman_id"] = prospect_career["playerID"].astype(float).astype(int).map(id_map)
    prospect_career["in_mlb"] = prospect_career["lahman_id"].isin(mlb_ids)

    # Sort by projection
    result = prospect_career.sort_values("projected_mlb_war", ascending=False)

    return result


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 70)
    print("  BRAVS Prospect Projection Model")
    print("=" * 70)

    # Step 1: Link MiLB to MLB
    print("\n--- Loading and linking MiLB + MLB data ---")
    linked = load_milb_mlb_linked()
    print(f"  Linked players: {len(linked)}")
    print(f"  Avg MLB career WAR: {linked.career_war_eq.mean():.1f}")
    print(f"  Avg MiLB total WAR: {linked.milb_total_war.mean():.1f}")

    # Step 2: Build features
    print("\n--- Building features ---")
    features, target = build_prospect_features(linked)
    print(f"  Features: {list(features.columns)}")
    print(f"  Samples: {len(features)}")

    # Step 3: Train model
    print("\n--- Training GBM ---")
    model = train_prospect_model(features, target)

    # Feature importance
    importances = sorted(zip(features.columns, model.feature_importances_),
                        key=lambda x: -x[1])
    print("\n  Feature importance:")
    for feat, imp in importances[:10]:
        print(f"    {feat:<25} {imp:.3f}")

    # Step 4: Validate against known prospects
    print("\n--- Validation: Known prospects ---")
    known = {
        "Mike Trout": 85.4,
        "Juan Soto": 35.0,
        "Ronald Acuna": 25.0,
        "Julio Rodriguez": 10.0,
        "Gunnar Henderson": 12.0,
    }
    for name, actual_war in known.items():
        match = linked[linked.name.str.contains(name.split()[-1]) &
                       linked.name.str.contains(name.split()[0][:4])]
        if len(match) > 0:
            row = match.sort_values("career_war_eq", ascending=False).iloc[0]
            feat_row = features.loc[row.name:row.name]
            if len(feat_row) > 0:
                pred = model.predict(feat_row)[0]
                print(f"  {name:<22} MiLB WAR={row.milb_total_war:+.1f} -> "
                      f"Predicted MLB={pred:.1f}, Actual={actual_war:.1f}")
            else:
                print(f"  {name:<22} (features not found)")
        else:
            print(f"  {name:<22} (not in linked data)")

    # Step 5: Current prospect rankings
    print("\n--- Current Prospect Rankings (2022-2026 MiLB) ---")
    prospects = project_current_prospects(model, list(features.columns))

    # Filter to prospects not yet established in MLB (or very new)
    not_established = prospects[
        (~prospects.in_mlb) | (prospects.milb_total_pa >= 200)
    ].head(30)

    print(f"\n  Top 30 Prospects by Projected MLB Career WAR:")
    print(f"  {'#':<4} {'Name':<24} {'Level':<5} {'MiLB WAR':>9} {'Proj MLB':>9} {'wOBA':>6} {'PA':>5}")
    print("  " + "-" * 65)
    for i, (_, r) in enumerate(not_established.iterrows()):
        in_mlb = "*" if r.in_mlb else " "
        print(f"  {i+1:<4} {r['name']:<24} {r.milb_highest_level:<5} "
              f"{r.milb_total_war:>+9.1f} {r.projected_mlb_war:>+9.1f} "
              f"{r.milb_avg_woba:>6.3f} {int(r.milb_total_pa):>5}{in_mlb}")

    # Save prospect rankings
    prospects.to_csv("data/prospect_rankings.csv", index=False)
    print(f"\n  Saved data/prospect_rankings.csv ({len(prospects)} players)")

    # Step 6: MiLB-to-MLB translation validation
    print("\n--- MiLB-to-MLB Translation Rates ---")
    for level in ["AAA", "AA", "A+", "A", "RK"]:
        level_players = linked[linked.milb_highest_level == level]
        if len(level_players) > 10:
            reached_mlb = (level_players.career_war_eq > 0).sum()
            star = (level_players.career_war_eq > 10).sum()
            print(f"  {level:<4}: {len(level_players):>5} players, "
                  f"{reached_mlb:>4} reached MLB ({reached_mlb/len(level_players)*100:.0f}%), "
                  f"{star:>3} became stars (>10 WAR, {star/len(level_players)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("  Prospect model complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
