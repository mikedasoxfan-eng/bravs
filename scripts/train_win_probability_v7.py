"""Win Probability Model v7 — v6 + starters + park factors + regularization.

v6 closed the state-only ceiling by adding per-PA + team-game quality and
hit 76.95% val (vs v5 75.18%). But it still overfit (train 84%, val 77%),
leaving ~7pp of headroom. v7 tries to cash in by adding:

  1. Starting pitcher quality for both teams. The starter pitches most of
     the game, so "who is starting" is more stable signal than "who is
     currently on the mound" (which averages over SP + bullpen).
  2. Ballpark factor (runs/game at this park vs league mean for the year).
     Park effects are large (Coors +22% runs, Petco -10%) and shift what a
     "2-run lead in the 7th" means.
  3. Tighter regularization: fewer iterations, bigger leaves, more L2.

Same temporal split as v6 (2015-2023 train, 2024 val) so numbers are
directly comparable.
"""

import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from scripts.train_win_probability_v5 import (
    reconstruct, load_real_outcomes, build_table, build_prior,
)
from scripts.train_win_probability_v6 import (
    load_quality_tables, build_feature_frame, attach_baseline_wp,
    BAT_COLS, PIT_COLS,
)


SP_COLS = ["xwoba_allowed", "whiff_pct", "velo_avg"]


def load_starters():
    """Return a DataFrame with one row per (game_id, home/away) and the
    starter's Statcast season quality merged in."""
    gl = pd.read_csv("data/retrosheet/game_logs_parsed.csv")
    gl["date_str"] = gl.date.astype(str)
    gl = gl.sort_values(["date_str", "home_team"]).reset_index(drop=True)
    counts = gl.groupby(["date_str", "home_team"]).cumcount()
    totals = gl.groupby(["date_str", "home_team"])["home_team"].transform("size")
    gl["game_num"] = np.where(totals >= 2, counts + 1, 0)
    gl["game_id"] = gl.home_team + gl.date_str + gl.game_num.astype(str)

    people = pd.read_csv("data/lahman2025/People.csv",
                        usecols=["playerID", "retroID"], low_memory=False)
    people = people.dropna(subset=["retroID"]).drop_duplicates("retroID")

    pit = pd.read_csv("data/statcast/pitcher_season_features.csv")
    pit = pit.merge(people, on="playerID", how="inner")
    pit = pit[["retroID", "yearID"] + SP_COLS]

    gl = gl[["game_id", "year", "home_sp_id", "vis_sp_id"]]
    # Merge home starter quality
    hq = pit.rename(columns={c: f"hsp_{c}" for c in SP_COLS})
    vq = pit.rename(columns={c: f"vsp_{c}" for c in SP_COLS})
    gl = gl.merge(hq, left_on=["home_sp_id", "year"],
                  right_on=["retroID", "yearID"], how="left"
                  ).drop(columns=["retroID", "yearID"])
    gl = gl.merge(vq, left_on=["vis_sp_id", "year"],
                  right_on=["retroID", "yearID"], how="left"
                  ).drop(columns=["retroID", "yearID"])

    # League means for missing
    means = {
        f"hsp_{c}": pit[c].mean() for c in SP_COLS
    } | {
        f"vsp_{c}": pit[c].mean() for c in SP_COLS
    }
    for col, m in means.items():
        gl[col] = gl[col].fillna(m)

    return gl[["game_id"] + list(means.keys())], means


def load_park_factors():
    """Return (year, home_team) -> runs-per-game ratio vs league."""
    gl = pd.read_csv("data/retrosheet/game_logs_parsed.csv")
    gl["tot"] = gl.vis_runs + gl.home_runs
    pf = gl.groupby(["year", "home_team"]).tot.mean().reset_index()
    lg = gl.groupby("year").tot.mean().reset_index().rename(columns={"tot": "lg"})
    pf = pf.merge(lg, on="year")
    pf["park_factor"] = pf.tot / pf.lg
    return pf[["year", "home_team", "park_factor"]]


def main():
    print("=" * 72)
    print("  WIN PROBABILITY MODEL v7 \u2014 v6 + Starters + Parks")
    print("=" * 72)

    print("\nLoading events 2015-2024...")
    plays = pd.read_parquet("data/retrosheet/events_parsed.parquet")
    plays = plays[plays.event_simple.isin(
        ["HR", "HIT", "WALK", "HBP", "OUT", "K", "ERROR", "FC"]
    )].copy()
    plays["year"] = plays.date.str[:4].astype(int)
    plays = plays[plays.year.between(2015, 2024)].copy()
    print(f"  {len(plays):,} plays")

    outcomes = load_real_outcomes()
    plays = plays[plays.game_id.isin(outcomes)].copy()
    print(f"  Matched: {len(plays):,} plays, {plays.game_id.nunique():,} games")

    states = reconstruct(plays)
    states["home_won"] = states.game_id.map(outcomes).astype(float)

    bat_q, pit_q, bat_means, pit_means = load_quality_tables()
    df = build_feature_frame(plays, states, bat_q, pit_q, bat_means, pit_means)

    # Build v5 table on train slice only
    print("\nBuilding v5 table (train 2015-2023)...")
    train_mask = df.year.values < 2024
    table = build_table(df.loc[train_mask])
    prior_map = build_prior(table)
    df = attach_baseline_wp(df, table, prior_map)

    # v7 additions
    print("\nLoading starter quality...")
    starters, sp_means = load_starters()
    df = df.merge(starters, on="game_id", how="left")
    for col, m in sp_means.items():
        df[col] = df[col].fillna(m)
    print(f"  Merged {len(starters):,} starter-games with {len(sp_means)} cols")

    print("\nLoading park factors...")
    pf = load_park_factors()
    df = df.merge(pf, on=["year", "home_team"], how="left")
    df["park_factor"] = df.park_factor.fillna(1.0)

    feature_cols = [
        # State
        "inning_c", "is_home_batting", "outs", "score_diff_bat", "bases_idx",
        "baseline_wp_v5",
        # v6 per-PA quality
        *[f"bat_{c}" for c in BAT_COLS],
        *[f"pit_{c}" for c in PIT_COLS],
        # v6 team-game aggregates
        *[f"bat_{c}_tg" for c in BAT_COLS],
        *[f"pit_{c}_tg" for c in PIT_COLS],
        # v7 starters
        *[f"hsp_{c}" for c in SP_COLS],
        *[f"vsp_{c}" for c in SP_COLS],
        # v7 park
        "park_factor",
    ]
    print(f"\nFeatures: {len(feature_cols)}")

    df["bat_won"] = np.where(df.is_home_batting.values == 1,
                             df.home_won.values, 1.0 - df.home_won.values
                             ).astype(np.float32)

    train = df[df.year < 2024]
    val = df[df.year == 2024]
    print(f"Split: train={len(train):,}  val={len(val):,}")

    X_tr = train[feature_cols].values
    y_tr = train.bat_won.values
    X_va = val[feature_cols].values
    y_va = val.bat_won.values

    print("\nTraining HistGradientBoostingClassifier (regularized)...")
    from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.perf_counter()
    clf = HistGradientBoostingClassifier(
        max_iter=600,
        learning_rate=0.04,
        max_leaf_nodes=31,          # smaller leaves (v6 was 63)
        min_samples_leaf=500,       # more regularization (v6 was 200)
        l2_regularization=3.0,      # heavier L2 (v6 was 1.0)
        max_depth=8,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    print(f"  Trained in {time.perf_counter()-t0:.1f}s ({clf.n_iter_} iters)")

    def report(name, X, y):
        p = clf.predict_proba(X)[:, 1].astype(np.float32)
        acc = ((p > 0.5).astype(np.float32) == y).mean()
        brier = ((p - y) ** 2).mean()
        eps = 1e-5
        bce = -(y * np.log(p.clip(eps, 1 - eps)) +
                (1 - y) * np.log((1 - p).clip(eps, 1 - eps))).mean()
        print(f"  {name:<10} acc={acc:.3%}  brier={brier:.4f}  bce={bce:.4f}")
        return acc, brier, bce

    print("\n--- v7 Results ---")
    report("train", X_tr, y_tr)
    acc_v, brier_v, bce_v = report("val(2024)", X_va, y_va)

    wp5_va = val.baseline_wp_v5.values.astype(np.float32)
    v5_acc = ((wp5_va > 0.5).astype(np.float32) == y_va).mean()
    v5_brier = ((wp5_va - y_va) ** 2).mean()
    print(f"\n  v5 baseline on 2024: acc={v5_acc:.3%}  brier={v5_brier:.4f}")
    print(f"  v7 improvement:      acc +{(acc_v - v5_acc)*100:.2f}pp  "
          f"brier {brier_v - v5_brier:+.4f}")

    # Permutation importance
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

    os.makedirs("data/win_prob", exist_ok=True)
    with open("data/win_prob/wp_model_v7.pkl", "wb") as f:
        pickle.dump({
            "model": clf,
            "feature_cols": feature_cols,
            "bat_means": bat_means,
            "pit_means": pit_means,
            "sp_means": sp_means,
            "v5_table": table,
            "v5_prior_map": prior_map,
            "val_acc": float(acc_v),
            "val_brier": float(brier_v),
            "v5_val_acc": float(v5_acc),
        }, f)
    print("\n  Saved v7 -> data/win_prob/wp_model_v7.pkl")
    print("=" * 72)


if __name__ == "__main__":
    main()
