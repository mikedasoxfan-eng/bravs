"""Pitcher fatigue curve model — built on Statcast pitch-by-pitch 2021-2025.

Fatigue manifests as velocity decay, spin decay, and whiff-rate decay as a
pitcher accumulates pitches within a single appearance. We also measure the
Times-Through-Order (TTO) penalty: the third time a batter sees a pitcher is
famously worse than the first time.

Outputs (data/fatigue/):
  league_curve.csv      League-average delta_velo/delta_spin/delta_whiff by
                        pitches_thrown bucket (1-10, 11-20, ..., 101+).
  tto_penalty.csv       League-average delta_metric by TTO bucket (1, 2, 3+).
  pitcher_profiles.csv  Per-pitcher fatigue coefficients (base_velo,
                        velo_decay_rate, tto_penalty, n_appearances).
  fatigue_model.pkl     Pickled dict with league_curve + tto + pitcher map
                        for inference use.

Inference:
  predicted_velo = base_velo - velo_decay_curve[bucket] - tto_penalty[tto]
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "fatigue")
os.makedirs(OUT_DIR, exist_ok=True)

# Years to use — tune for runtime vs sample size
YEARS = [2021, 2022, 2023, 2024, 2025]

# Columns we need from pitch-by-pitch parquet (tiny fraction of the 118)
NEEDED = [
    "pitcher", "player_name", "game_pk", "at_bat_number", "pitch_number",
    "release_speed", "release_spin_rate", "description",
    "inning", "n_thruorder_pitcher", "pitch_type",
]

BUCKET_EDGES = [0, 10, 20, 30, 40, 50, 60, 75, 90, 110, 9999]
BUCKET_LABELS = ["1-10", "11-20", "21-30", "31-40", "41-50",
                 "51-60", "61-75", "76-90", "91-110", "111+"]


def load_pitches(years: list[int]) -> pd.DataFrame:
    frames = []
    for y in years:
        path = f"data/statcast/pbp/statcast_{y}.parquet"
        print(f"  loading {y}...", end=" ", flush=True)
        t0 = time.time()
        df = pd.read_parquet(path, columns=NEEDED)
        # Drop pitches with no velocity reading (a handful per year)
        df = df.dropna(subset=["release_speed", "pitcher", "game_pk",
                               "at_bat_number", "pitch_number"])
        # Cast to compact dtypes
        df["pitcher"] = df["pitcher"].astype(np.int32)
        df["game_pk"] = df["game_pk"].astype(np.int32)
        df["at_bat_number"] = df["at_bat_number"].astype(np.int16)
        df["pitch_number"] = df["pitch_number"].astype(np.int16)
        df["release_speed"] = df["release_speed"].astype(np.float32)
        # spin may have NaN
        df["release_spin_rate"] = df["release_spin_rate"].astype("Float32")
        # tto may have NaN for some rare edge rows
        df["n_thruorder_pitcher"] = df["n_thruorder_pitcher"].fillna(1).astype(np.int8)
        frames.append(df)
        print(f"{len(df):,} pitches ({time.time()-t0:.1f}s)")
    out = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(out):,} pitches")
    return out


def add_pitches_thrown(df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative pitches_thrown per pitcher per game."""
    # Sort so that cumcount gives pitches-thrown-so-far
    df = df.sort_values(["game_pk", "pitcher", "at_bat_number", "pitch_number"],
                        kind="stable").reset_index(drop=True)
    df["pitches_thrown"] = df.groupby(["game_pk", "pitcher"]).cumcount() + 1
    df["pitches_thrown"] = df["pitches_thrown"].astype(np.int16)
    return df


def compute_league_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate velocity/spin/whiff by pitches_thrown bucket.

    Normalize velo by subtracting each pitcher's own pitches-1-10 baseline,
    otherwise high-velo starters dominate the raw curve.
    """
    # Per-pitcher baseline (first 10 pitches of each appearance, averaged
    # across all that pitcher's appearances)
    baselines = (
        df[df.pitches_thrown <= 10]
        .groupby("pitcher")
        .agg(
            base_velo=("release_speed", "mean"),
            base_spin=("release_spin_rate", "mean"),
        )
        .reset_index()
    )

    df = df.merge(baselines, on="pitcher", how="left")
    # Drop pitchers with missing baseline (insufficient early-pitch samples)
    df = df.dropna(subset=["base_velo"])
    df["delta_velo"] = df["release_speed"] - df["base_velo"]
    df["delta_spin"] = (df["release_spin_rate"] - df["base_spin"]).astype("float32")

    # Whiff = swinging_strike / swinging_strike_blocked
    df["is_whiff"] = df["description"].isin(
        ["swinging_strike", "swinging_strike_blocked"]
    ).astype(np.float32)

    # Bucket
    df["bucket"] = pd.cut(
        df.pitches_thrown, bins=BUCKET_EDGES, labels=BUCKET_LABELS, right=True
    )

    curve = (
        df.groupby("bucket", observed=True)
        .agg(
            n=("delta_velo", "size"),
            delta_velo=("delta_velo", "mean"),
            delta_spin=("delta_spin", "mean"),
            whiff_rate=("is_whiff", "mean"),
        )
        .reset_index()
    )
    curve["bucket"] = curve.bucket.astype(str)
    return curve, df  # return enriched df for further analysis


def compute_tto_curve(df: pd.DataFrame) -> pd.DataFrame:
    """TTO = Times Through the Order. Penalty grows each time through."""
    # Treat TTO >=3 as one bucket since data thins out
    df = df.copy()
    df["tto_b"] = df["n_thruorder_pitcher"].clip(1, 3)
    curve = (
        df.groupby("tto_b", observed=True)
        .agg(
            n=("delta_velo", "size"),
            delta_velo=("delta_velo", "mean"),
            delta_spin=("delta_spin", "mean"),
            whiff_rate=("is_whiff", "mean"),
        )
        .reset_index()
        .rename(columns={"tto_b": "tto"})
    )
    return curve


def fit_pitcher_profiles(df: pd.DataFrame, min_pitches: int = 300) -> pd.DataFrame:
    """Per-pitcher linear fit: velocity drop vs log(1+pitches_thrown/20).

    Keeps only pitchers with enough sample. Reports:
      base_velo        — mean of first 10 pitches
      decay_rate       — slope of delta_velo vs log1p(pitches/20)
      max_drop         — predicted velo drop at pitch 100
      tto3_penalty     — additional velo drop on TTO=3+ vs TTO=1
      n_pitches, name
    """
    counts = df.groupby("pitcher").size()
    keep = counts[counts >= min_pitches].index
    sub = df[df.pitcher.isin(keep)]

    # log transform
    sub = sub.copy()
    sub["lp"] = np.log1p(sub.pitches_thrown / 20.0)

    rows = []
    for pid, grp in sub.groupby("pitcher"):
        lp = grp.lp.values
        dv = grp.delta_velo.values
        # Quick closed-form OLS: delta_velo = a + b*lp. Intercept should be ~0.
        if len(lp) < 50:
            continue
        lp_mean = lp.mean()
        dv_mean = dv.mean()
        cov = ((lp - lp_mean) * (dv - dv_mean)).mean()
        var = ((lp - lp_mean) ** 2).mean()
        if var < 1e-8:
            continue
        b = cov / var
        a = dv_mean - b * lp_mean

        # TTO3 penalty: avg delta_velo at TTO=3+ minus avg at TTO=1
        tto1 = grp[grp.n_thruorder_pitcher == 1].delta_velo.mean()
        tto3 = grp[grp.n_thruorder_pitcher >= 3].delta_velo.mean()
        tto3_pen = (tto3 - tto1) if not (np.isnan(tto1) or np.isnan(tto3)) else np.nan

        # Predicted max drop at pitch 100
        lp100 = np.log1p(100 / 20.0)
        max_drop = a + b * lp100

        rows.append({
            "pitcher": int(pid),
            "player_name": grp.player_name.iloc[0],
            "n_pitches": len(grp),
            "base_velo": float(grp.release_speed[grp.pitches_thrown <= 10].mean()),
            "decay_intercept": float(a),
            "decay_slope": float(b),
            "max_drop_p100": float(max_drop),
            "tto3_penalty": float(tto3_pen) if not np.isnan(tto3_pen) else None,
        })

    return pd.DataFrame(rows).sort_values("n_pitches", ascending=False)


def fit_league_parametric(league_curve: pd.DataFrame) -> dict:
    """Fit a simple parametric curve delta_velo = -alpha * log1p(p/beta)
    using the bucket centers. Returns (alpha, beta, rmse).
    """
    # Approximate bucket centers
    centers = {
        "1-10": 5, "11-20": 15, "21-30": 25, "31-40": 35, "41-50": 45,
        "51-60": 55, "61-75": 68, "76-90": 83, "91-110": 100, "111+": 120,
    }
    lc = league_curve.copy()
    lc["center"] = lc["bucket"].map(centers)
    lc = lc.dropna(subset=["center", "delta_velo"])
    p = lc["center"].values.astype(float)
    y = lc["delta_velo"].values.astype(float)

    # Grid search beta, solve alpha in closed form for each beta
    best = (np.inf, 0, 0)
    for beta in np.linspace(5, 80, 60):
        x = np.log1p(p / beta)
        # Fit y ~= -alpha * x  (no intercept)
        alpha = -(x * y).sum() / (x * x).sum()
        resid = y - (-alpha * x)
        rmse = float(np.sqrt((resid ** 2).mean()))
        if rmse < best[0]:
            best = (rmse, alpha, beta)
    return {"alpha": best[1], "beta": best[2], "rmse": best[0]}


def main():
    print("=" * 72)
    print("  PITCHER FATIGUE MODEL — Statcast 2021-2025")
    print("=" * 72)

    print("\nLoading pitch-by-pitch data...")
    t0 = time.time()
    df = load_pitches(YEARS)
    print(f"  Load time: {time.time() - t0:.1f}s")

    print("\nComputing cumulative pitches per appearance...")
    df = add_pitches_thrown(df)
    print(f"  Max pitches in one appearance: {df.pitches_thrown.max()}")
    print(f"  Unique pitcher-game combos: {df.groupby(['game_pk','pitcher']).ngroups:,}")

    print("\nComputing league-wide fatigue curve...")
    league_curve, enriched = compute_league_curve(df)
    print("\n  Pitches      n           d_velo    d_spin     whiff%")
    for _, r in league_curve.iterrows():
        spin = f"{r.delta_spin:+5.0f}" if not np.isnan(r.delta_spin) else "   -"
        print(f"  {r['bucket']:<10} {int(r['n']):>10,}   {r.delta_velo:+5.2f}  "
              f"{spin}    {r.whiff_rate:.1%}")

    print("\nComputing TTO (Times Through Order) penalty...")
    tto_curve = compute_tto_curve(enriched)
    print("\n  TTO   n            d_velo    d_spin     whiff%")
    for _, r in tto_curve.iterrows():
        spin = f"{r.delta_spin:+5.0f}" if not np.isnan(r.delta_spin) else "   -"
        print(f"  {int(r.tto):<3}   {int(r.n):>10,}   {r.delta_velo:+5.2f}  "
              f"{spin}    {r.whiff_rate:.1%}")

    print("\nFitting league parametric curve d_velo = -alpha * log(1 + p/beta) ...")
    params = fit_league_parametric(league_curve)
    print(f"  alpha = {params['alpha']:.4f}")
    print(f"  beta  = {params['beta']:.2f}")
    print(f"  RMSE  = {params['rmse']:.4f} mph")
    for p in [10, 30, 50, 75, 100]:
        pred = -params["alpha"] * np.log1p(p / params["beta"])
        print(f"    p={p:>3}: predicted d_velo = {pred:+.2f} mph")

    print("\nFitting per-pitcher profiles (min 300 pitches) ...")
    profiles = fit_pitcher_profiles(enriched, min_pitches=300)
    print(f"  {len(profiles)} pitchers profiled")

    # Notable examples
    print("\n--- Highest velocity base (avg first 10 pitches) ---")
    top = profiles.sort_values("base_velo", ascending=False).head(10)
    for _, r in top.iterrows():
        drop = r.max_drop_p100 if not pd.isna(r.max_drop_p100) else 0
        print(f"  {r.player_name:<25} base={r.base_velo:.1f}  "
              f"drop@p100={drop:+.2f}  n={r.n_pitches:,}")

    print("\n--- Largest velocity decay (most fatigue) ---")
    fatig = profiles.dropna(subset=["max_drop_p100"])
    fatig = fatig[fatig.n_pitches >= 800].sort_values("max_drop_p100")
    for _, r in fatig.head(10).iterrows():
        print(f"  {r.player_name:<25} base={r.base_velo:.1f}  "
              f"drop@p100={r.max_drop_p100:+.2f}  n={r.n_pitches:,}")

    print("\n--- Most stamina (smallest velocity decay) ---")
    for _, r in fatig.tail(10).iloc[::-1].iterrows():
        print(f"  {r.player_name:<25} base={r.base_velo:.1f}  "
              f"drop@p100={r.max_drop_p100:+.2f}  n={r.n_pitches:,}")

    print("\n--- Worst TTO3 penalty (third time through order) ---")
    tto_bad = profiles.dropna(subset=["tto3_penalty"])
    tto_bad = tto_bad[tto_bad.n_pitches >= 800].sort_values("tto3_penalty")
    for _, r in tto_bad.head(10).iterrows():
        print(f"  {r.player_name:<25} TTO3 d_velo={r.tto3_penalty:+.2f}  n={r.n_pitches:,}")

    print("\nSaving outputs...")
    league_curve.to_csv(os.path.join(OUT_DIR, "league_curve.csv"), index=False)
    tto_curve.to_csv(os.path.join(OUT_DIR, "tto_penalty.csv"), index=False)
    profiles.to_csv(os.path.join(OUT_DIR, "pitcher_profiles.csv"), index=False)

    with open(os.path.join(OUT_DIR, "fatigue_model.pkl"), "wb") as f:
        pickle.dump({
            "league_curve": league_curve.to_dict(orient="records"),
            "tto_curve": tto_curve.to_dict(orient="records"),
            "league_params": params,
            "pitcher_profiles": profiles.set_index("pitcher").to_dict(orient="index"),
            "years": YEARS,
        }, f)

    print(f"  -> {OUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
