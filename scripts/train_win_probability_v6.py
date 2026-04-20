"""Win Probability Model v6 — State + pitcher/batter quality.

v5 was a pure lookup table over (inning, half, outs, diff, bases) and plateaued
at 74.87% accuracy because it treats every matchup as league-average. v6 adds
the one thing v5 couldn't know: WHO is hitting and WHO is pitching.

Approach:
  1. Reconstruct per-play state (reuse v5 logic).
  2. Attach v5 baseline WP as a feature so the model only has to learn the
     residual from pitcher/batter quality.
  3. Join Statcast season features via retroID -> playerID (Lahman People):
     batter:  xwoba, whiff_pct, barrel_pct, hard_hit_pct
     pitcher: xwoba_allowed, whiff_pct, velo_avg
  4. Train sklearn HistGradientBoosting on 2015-2023, validate on 2024.

Missing quality features fall back to league mean (bench/call-up plays stay at
the v5 baseline instead of being dropped).
"""

import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from scripts.train_win_probability_v5 import (
    reconstruct, load_real_outcomes, build_table, build_prior, lookup_wp
)


BAT_COLS = ["xwoba", "whiff_pct", "barrel_pct", "hard_hit_pct"]
PIT_COLS = ["xwoba_allowed", "whiff_pct", "velo_avg"]


def load_quality_tables():
    """Load batter/pitcher season features keyed by (retroID, year)."""
    people = pd.read_csv("data/lahman2025/People.csv",
                        usecols=["playerID", "retroID"], low_memory=False)
    people = people.dropna(subset=["retroID"]).drop_duplicates("retroID")

    bat = pd.read_csv("data/statcast/batter_season_features.csv")
    pit = pd.read_csv("data/statcast/pitcher_season_features.csv")

    bat = bat.merge(people, on="playerID", how="inner")
    pit = pit.merge(people, on="playerID", how="inner")

    bat = bat[["retroID", "yearID"] + BAT_COLS].rename(
        columns={c: f"bat_{c}" for c in BAT_COLS})
    pit = pit[["retroID", "yearID"] + PIT_COLS].rename(
        columns={c: f"pit_{c}" for c in PIT_COLS})

    bat_means = bat[[f"bat_{c}" for c in BAT_COLS]].mean().to_dict()
    pit_means = pit[[f"pit_{c}" for c in PIT_COLS]].mean().to_dict()
    return bat, pit, bat_means, pit_means


def build_feature_frame(plays, states, bat_q, pit_q, bat_means, pit_means):
    """Attach per-PA and team-game quality to each state row.

    Team-game aggregates are the crucial v6 feature: averaging the per-PA
    batter/pitcher quality across all plays by a given team in a given game
    gives a stable offense/defense score for that game. This captures
    "Skubal on a bad offense vs league-average arm on a bad offense" across
    every play, not just the one PA where Skubal happens to be on the mound.
    """
    print("\nBuilding feature frame...")
    t0 = time.perf_counter()

    plays = plays.copy()
    plays["half_ord"] = plays.half.map({"T": 0, "B": 1})
    plays = plays.sort_values(
        ["game_id", "inning", "half_ord", "outs_before"], kind="stable"
    ).reset_index(drop=True)
    plays["year"] = plays.date.str[:4].astype(int)

    assert len(plays) == len(states), (len(plays), len(states))
    states = states.reset_index(drop=True)

    df = pd.concat([
        states,
        plays[["batter_id", "pitcher_id", "year",
               "home_team", "away_team"]].reset_index(drop=True)
    ], axis=1)

    df = df.merge(bat_q, left_on=["batter_id", "year"],
                  right_on=["retroID", "yearID"], how="left"
                  ).drop(columns=["retroID", "yearID"])
    df = df.merge(pit_q, left_on=["pitcher_id", "year"],
                  right_on=["retroID", "yearID"], how="left"
                  ).drop(columns=["retroID", "yearID"])

    for col, mean in bat_means.items():
        df[col] = df[col].fillna(mean)
    for col, mean in pit_means.items():
        df[col] = df[col].fillna(mean)

    # Batting / pitching team IDs for this play
    is_bot = df.is_home_batting.values.astype(np.int32)
    df["bat_team"] = np.where(is_bot == 1, df.home_team, df.away_team)
    df["pit_team"] = np.where(is_bot == 1, df.away_team, df.home_team)

    # Team-game aggregates: mean of per-PA quality across all plays where
    # this team batted / pitched in this game. Merged back so every row has
    # the aggregate for its own batting team AND pitching team.
    print("  Computing team-game aggregates...")
    bat_agg_cols = [f"bat_{c}" for c in BAT_COLS]
    pit_agg_cols = [f"pit_{c}" for c in PIT_COLS]

    bat_agg = df.groupby(["game_id", "bat_team"], sort=False)[bat_agg_cols
                ].mean().reset_index()
    bat_agg = bat_agg.rename(columns={c: c + "_tg" for c in bat_agg_cols})
    pit_agg = df.groupby(["game_id", "pit_team"], sort=False)[pit_agg_cols
                ].mean().reset_index()
    pit_agg = pit_agg.rename(columns={c: c + "_tg" for c in pit_agg_cols})

    df = df.merge(bat_agg, on=["game_id", "bat_team"], how="left")
    df = df.merge(pit_agg, on=["game_id", "pit_team"], how="left")

    # Batting-team-centric diff
    sd = df.score_diff_home.values
    df["score_diff_bat"] = np.clip(
        np.where(is_bot == 1, sd, -sd), -10, 10
    ).astype(np.int32)
    df["inning_c"] = np.minimum(df.inning.values, 12).astype(np.int32)

    print(f"  Features ready in {time.perf_counter()-t0:.1f}s "
          f"({len(df):,} rows)")
    return df


def attach_baseline_wp(df, table, prior_map):
    print("  Computing v5 baseline WP for each state...")
    wp5 = np.empty(len(df), dtype=np.float32)
    inning = df.inning_c.values
    is_bot = df.is_home_batting.values.astype(np.int32)
    outs = df.outs.values.astype(np.int32)
    diff = df.score_diff_bat.values
    bi = df.bases_idx.values.astype(np.int32)
    for i in range(len(df)):
        wp5[i] = lookup_wp(table, prior_map,
                           int(inning[i]), int(is_bot[i]),
                           int(outs[i]), int(diff[i]), int(bi[i]))
    df = df.copy()
    df["baseline_wp_v5"] = wp5
    return df


def main():
    print("=" * 72)
    print("  WIN PROBABILITY MODEL v6 — State + Pitcher/Batter Quality")
    print("=" * 72)

    print("\nLoading events (2015-2024, Statcast-era)...")
    plays = pd.read_parquet("data/retrosheet/events_parsed.parquet")
    plays = plays[plays.event_simple.isin(
        ["HR", "HIT", "WALK", "HBP", "OUT", "K", "ERROR", "FC"]
    )].copy()
    plays["year"] = plays.date.str[:4].astype(int)
    plays = plays[plays.year.between(2015, 2024)].copy()
    print(f"  {len(plays):,} plays")

    outcomes = load_real_outcomes()
    plays = plays[plays.game_id.isin(outcomes)].copy()
    print(f"  Matched to real outcomes: {len(plays):,} plays, "
          f"{plays.game_id.nunique():,} games")

    states = reconstruct(plays)
    states["home_won"] = states.game_id.map(outcomes).astype(float)

    bat_q, pit_q, bat_means, pit_means = load_quality_tables()
    print(f"\nQuality tables: {len(bat_q):,} batter-seasons, "
          f"{len(pit_q):,} pitcher-seasons")

    df = build_feature_frame(plays, states,
                             bat_q, pit_q, bat_means, pit_means)

    # Build the v5 lookup table from TRAIN ONLY to avoid leaking 2024
    # outcomes into the baseline feature.
    print("\nBuilding v5 table from train slice (2015-2023)...")
    train_mask = df.year.values < 2024
    table = build_table(df.loc[train_mask])
    prior_map = build_prior(table)

    df = attach_baseline_wp(df, table, prior_map)

    # Target: did the batting team ultimately win?
    df["bat_won"] = np.where(df.is_home_batting.values == 1,
                             df.home_won.values,
                             1.0 - df.home_won.values).astype(np.float32)

    feature_cols = [
        "inning_c", "is_home_batting", "outs", "score_diff_bat", "bases_idx",
        "baseline_wp_v5",
        # Per-PA quality (who is up right now)
        *[f"bat_{c}" for c in BAT_COLS],
        *[f"pit_{c}" for c in PIT_COLS],
        # Team-game aggregate quality (offense/defense for this game)
        *[f"bat_{c}_tg" for c in BAT_COLS],
        *[f"pit_{c}_tg" for c in PIT_COLS],
    ]

    # Temporal split: 2015-2023 train, 2024 validate
    train = df[df.year < 2024]
    val = df[df.year == 2024]
    print(f"\nSplit: train={len(train):,}  val={len(val):,}")

    X_tr = train[feature_cols].values
    y_tr = train.bat_won.values
    X_va = val[feature_cols].values
    y_va = val.bat_won.values

    print("\nTraining HistGradientBoostingClassifier...")
    from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.perf_counter()
    clf = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    print(f"  Trained in {time.perf_counter()-t0:.1f}s "
          f"({clf.n_iter_} iterations)")

    # Evaluate
    def report(name, X, y):
        p = clf.predict_proba(X)[:, 1].astype(np.float32)
        acc = ((p > 0.5).astype(np.float32) == y).mean()
        brier = ((p - y) ** 2).mean()
        eps = 1e-5
        bce = -(y * np.log(p.clip(eps, 1 - eps)) +
                (1 - y) * np.log((1 - p).clip(eps, 1 - eps))).mean()
        print(f"  {name:<10} acc={acc:.3%}  brier={brier:.4f}  bce={bce:.4f}")
        return acc, brier, bce

    print("\n--- v6 Results ---")
    report("train", X_tr, y_tr)
    acc_v, brier_v, bce_v = report("val(2024)", X_va, y_va)

    # v5 baseline numbers on the same val slice
    wp5_va = val.baseline_wp_v5.values.astype(np.float32)
    v5_acc = ((wp5_va > 0.5).astype(np.float32) == y_va).mean()
    v5_brier = ((wp5_va - y_va) ** 2).mean()
    print(f"\n  v5 baseline on 2024: acc={v5_acc:.3%}  brier={v5_brier:.4f}")
    print(f"  v6 improvement:      acc +{(acc_v - v5_acc)*100:.2f}pp  "
          f"brier {brier_v - v5_brier:+.4f}")

    # Permutation importance on a 100k val sample (fast, model-agnostic)
    print("\n  Permutation importance (val, 100k sample):")
    try:
        from sklearn.inspection import permutation_importance
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X_va), size=min(100_000, len(X_va)), replace=False)
        pi = permutation_importance(
            clf, X_va[idx], y_va[idx],
            n_repeats=3, random_state=0, n_jobs=1,
            scoring="neg_log_loss",
        )
        order = np.argsort(pi.importances_mean)[::-1]
        for i in order:
            print(f"    {feature_cols[i]:<26} {pi.importances_mean[i]:+.5f}")
    except Exception as e:
        print(f"    (skipped: {e})")

    # Save
    os.makedirs("data/win_prob", exist_ok=True)
    with open("data/win_prob/wp_model_v6.pkl", "wb") as f:
        pickle.dump({
            "model": clf,
            "feature_cols": feature_cols,
            "bat_means": bat_means,
            "pit_means": pit_means,
            "v5_table": table,
            "v5_prior_map": prior_map,
            "val_acc": float(acc_v),
            "val_brier": float(brier_v),
            "v5_val_acc": float(v5_acc),
        }, f)
    print(f"\n  Saved v6 model -> data/win_prob/wp_model_v6.pkl")
    print("=" * 72)


if __name__ == "__main__":
    main()
