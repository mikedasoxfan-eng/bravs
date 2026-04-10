"""Statcast-BRAVS Blend: enhance BRAVS with Statcast expected batting and catcher framing.

For 2015-2025 player-seasons:
  1. Blends BRAVS hitting runs with Statcast xwOBA-based luck adjustment
  2. Replaces the flat 2.0 catcher framing proxy with real Statcast framing runs
  3. Recomputes bravs_war_eq with improved components
  4. Validates against known fWAR values

Output: data/bravs_statcast_blend.csv
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from baseball_metric.data.statcast_overlay import (
    get_statcast_hitting_adjustment,
    get_statcast_stats,
)
from baseball_metric.data.catcher_framing_data import get_framing_runs

# ---------------------------------------------------------------------------
# Curated fWAR benchmarks for validation (2015-2025)
# Source: FanGraphs. A diverse sample of hitters, pitchers, and catchers.
# ---------------------------------------------------------------------------
FWAR_BENCHMARKS: dict[tuple[str, int], float] = {
    # -- MVP-caliber hitters --
    ("troutmi01", 2016): 10.5,
    ("troutmi01", 2018): 9.8,
    ("troutmi01", 2019): 8.3,
    ("bettsmo01", 2018): 10.9,
    ("bettsmo01", 2020): 3.5,
    ("judgeaa01", 2017): 8.1,
    ("judgeaa01", 2022): 11.4,
    ("ohtansh01", 2021): 9.1,
    ("ohtansh01", 2023): 10.0,
    ("sotoju01", 2020): 4.9,
    ("sotoju01", 2024): 7.8,
    ("harpebr03", 2015): 9.3,
    ("harpebr03", 2021): 5.9,
    ("freemfr01", 2020): 3.7,
    ("freemfr01", 2022): 6.9,
    ("arenano01", 2015): 5.3,
    ("arenano01", 2018): 5.6,
    ("goldspa01", 2015): 8.8,
    ("lindofr01", 2018): 7.6,
    ("yelicch01", 2018): 7.6,
    ("yelicch01", 2019): 3.6,
    ("bellico01", 2019): 5.4,
    ("turnetr01", 2021): 6.9,
    ("alvaryo01", 2022): 6.0,
    ("acunaro01", 2023): 5.5,
    ("acunaro01", 2019): 5.9,
    # -- Starting pitchers --
    ("degroja01", 2018): 9.6,
    ("degroja01", 2019): 7.2,
    ("scherma01", 2015): 7.5,
    ("scherma01", 2018): 8.3,
    ("verlaju01", 2019): 7.8,
    ("verlaju01", 2022): 5.5,
    ("colege01", 2019): 7.4,
    ("colege01", 2023): 5.2,
    ("buehlwa01", 2021): 5.2,
    ("biebesh01", 2020): 3.6,
    ("alcansa01", 2019): 6.1,
    ("burneco01", 2021): 7.5,
    # -- Catchers --
    ("grandya01", 2019): 4.9,
    ("grandya01", 2015): 3.5,
    ("realmjt01", 2018): 4.1,
    ("contrwi01", 2022): 3.2,
    # -- Role players / DH --
    ("cruzne02", 2015): 3.9,
    ("cruzne02", 2019): 3.2,
    ("martijd02", 2015): 4.8,
}


def load_bravs_seasons() -> pd.DataFrame:
    """Load the pre-computed BRAVS all-seasons CSV."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "bravs_all_seasons.csv")
    df = pd.read_csv(path)
    return df


def compute_blend(df: pd.DataFrame) -> pd.DataFrame:
    """Enhance BRAVS with Statcast expected batting and catcher framing.

    For each player-season in the 2015-2025 window:
      - Blend hitting runs: 0.7 * bravs_hitting + 0.3 * statcast_adjusted_hitting
      - Replace flat catcher proxy (2.0 * games_frac) with real framing runs
      - Recompute total runs and bravs_war_eq
    """
    # Work on a copy; only modify 2015-2025 rows
    out = df.copy()

    # Track statistics
    n_hitting_blended = 0
    n_framing_replaced = 0
    n_total_statcast = 0
    n_catchers_statcast = 0

    statcast_mask = (out["yearID"] >= 2015) & (out["yearID"] <= 2025)
    statcast_rows = out[statcast_mask].index

    # New columns for diagnostics
    out["hitting_runs_orig"] = out["hitting_runs"]
    out["catcher_runs_orig"] = out["catcher_runs"]
    out["statcast_adj_runs"] = 0.0
    out["framing_runs_real"] = np.nan
    out["bravs_war_eq_orig"] = out["bravs_war_eq"]

    for idx in statcast_rows:
        pid = out.at[idx, "playerID"]
        yr = int(out.at[idx, "yearID"])
        n_total_statcast += 1

        # --- Hitting blend ---
        adj = get_statcast_hitting_adjustment(pid, yr)
        if adj is not None:
            bravs_hitting = out.at[idx, "hitting_runs"]
            # Statcast-adjusted hitting = bravs_hitting + luck adjustment
            statcast_adj_hitting = bravs_hitting + adj
            # Blend: 70% BRAVS, 30% Statcast-adjusted
            blended = 0.7 * bravs_hitting + 0.3 * statcast_adj_hitting
            out.at[idx, "hitting_runs"] = round(blended, 1)
            out.at[idx, "statcast_adj_runs"] = round(adj, 2)
            n_hitting_blended += 1

        # --- Catcher framing ---
        pos = out.at[idx, "position"]
        if pos == "C":
            n_catchers_statcast += 1
            framing = get_framing_runs(pid, yr)
            if framing is not None:
                out.at[idx, "catcher_runs"] = round(framing, 1)
                out.at[idx, "framing_runs_real"] = framing
                n_framing_replaced += 1

    # --- Recompute total runs and bravs_war_eq for blended rows ---
    # Total runs = sum of all component columns
    component_cols = [
        "hitting_runs", "pitching_runs", "baserunning_runs",
        "fielding_runs", "positional_runs", "durability_runs",
        "aqi_runs", "leverage_runs", "catcher_runs",
    ]
    out["total_runs"] = out[component_cols].sum(axis=1)

    # bravs (raw) = total_runs / rpw
    out["bravs"] = (out["total_runs"] / out["rpw"]).round(2)

    # bravs_era_std = total_runs / 5.90 (standard era RPW)
    out["bravs_era_std"] = (out["total_runs"] / 5.90).round(2)

    # Recompute bravs_war_eq using the same calibration factors from gpu_engine_v3
    # Position-specific + era-adjusted calibration (v3.8)
    ip = out["IP"]
    pa = out["PA"].clip(lower=1)
    years = out["yearID"]
    is_mainly_pitcher = (ip >= pa * 0.3).astype(float)
    is_modern_pitcher = is_mainly_pitcher * (years >= 1985).astype(float)
    is_old_pitcher = is_mainly_pitcher * (years < 1985).astype(float)
    is_modern_hitter = (1 - is_mainly_pitcher) * (years >= 2000).astype(float)
    is_classic_hitter = (1 - is_mainly_pitcher) * (years < 2000).astype(float)

    cal_factor = (
        is_old_pitcher * 0.580
        + is_modern_pitcher * 0.680
        + is_classic_hitter * 0.690
        + is_modern_hitter * 0.695
    )
    out["bravs_war_eq"] = (out["bravs"] * cal_factor).round(2)

    # Drop the helper column
    out.drop(columns=["total_runs"], inplace=True)

    print(f"\n  Statcast blend summary (2015-2025):")
    print(f"    Total player-seasons in range:     {n_total_statcast:,}")
    print(f"    Hitting blended with xwOBA:        {n_hitting_blended:,}")
    print(f"    Catchers in range:                 {n_catchers_statcast:,}")
    print(f"    Catcher framing replaced:          {n_framing_replaced:,}")

    return out


def validate_against_fwar(df: pd.DataFrame) -> None:
    """Compare blended bravs_war_eq to known fWAR benchmarks for 2015-2025."""
    matched = []
    for (pid, yr), fwar in FWAR_BENCHMARKS.items():
        row = df[(df["playerID"] == pid) & (df["yearID"] == yr)]
        if len(row) == 0:
            continue
        bravs_eq = row.iloc[0]["bravs_war_eq"]
        bravs_eq_orig = row.iloc[0]["bravs_war_eq_orig"]
        matched.append({
            "playerID": pid,
            "yearID": yr,
            "fWAR": fwar,
            "bravs_war_eq_blend": bravs_eq,
            "bravs_war_eq_orig": bravs_eq_orig,
        })

    if not matched:
        print("\n  No fWAR benchmarks matched in the dataset.")
        return

    comp = pd.DataFrame(matched)
    n = len(comp)

    # Correlation: blended vs fWAR
    corr_blend = comp["fWAR"].corr(comp["bravs_war_eq_blend"])
    corr_orig = comp["fWAR"].corr(comp["bravs_war_eq_orig"])

    # RMSE
    rmse_blend = np.sqrt(((comp["fWAR"] - comp["bravs_war_eq_blend"]) ** 2).mean())
    rmse_orig = np.sqrt(((comp["fWAR"] - comp["bravs_war_eq_orig"]) ** 2).mean())

    # MAE
    mae_blend = (comp["fWAR"] - comp["bravs_war_eq_blend"]).abs().mean()
    mae_orig = (comp["fWAR"] - comp["bravs_war_eq_orig"]).abs().mean()

    print(f"\n  fWAR validation ({n} benchmark player-seasons, 2015-2025):")
    print(f"  {'Metric':<30s} {'Original':>10s} {'Blended':>10s} {'Change':>10s}")
    print(f"  {'-' * 62}")
    print(f"  {'Pearson r vs fWAR':<30s} {corr_orig:>10.4f} {corr_blend:>10.4f} {corr_blend - corr_orig:>+10.4f}")
    print(f"  {'RMSE vs fWAR':<30s} {rmse_orig:>10.3f} {rmse_blend:>10.3f} {rmse_blend - rmse_orig:>+10.3f}")
    print(f"  {'MAE vs fWAR':<30s} {mae_orig:>10.3f} {mae_blend:>10.3f} {mae_blend - mae_orig:>+10.3f}")

    # Show biggest movers
    comp["delta"] = comp["bravs_war_eq_blend"] - comp["bravs_war_eq_orig"]
    biggest = comp.reindex(comp["delta"].abs().sort_values(ascending=False).index).head(10)
    print(f"\n  Top 10 biggest changes from Statcast blend:")
    print(f"  {'Player':<16s} {'Year':>5s} {'fWAR':>6s} {'Orig':>6s} {'Blend':>7s} {'Delta':>7s}")
    print(f"  {'-' * 50}")
    for _, r in biggest.iterrows():
        print(f"  {r['playerID']:<16s} {int(r['yearID']):>5d} {r['fWAR']:>6.1f} "
              f"{r['bravs_war_eq_orig']:>6.2f} {r['bravs_war_eq_blend']:>7.2f} {r['delta']:>+7.2f}")


def main():
    t_start = time.perf_counter()

    print("=" * 72)
    print("  STATCAST-BRAVS BLEND: xwOBA + Catcher Framing Enhancement")
    print("=" * 72)

    # 1. Load BRAVS all-seasons
    print("\n  Loading BRAVS all-seasons...")
    df = load_bravs_seasons()
    print(f"    Loaded {len(df):,} player-seasons ({df['yearID'].min()}-{df['yearID'].max()})")

    # 2. Compute blend for 2015-2025
    print("\n  Computing Statcast-BRAVS blend...")
    blended = compute_blend(df)

    # 3. Save output
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "bravs_statcast_blend.csv")
    blended.to_csv(out_path, index=False)
    print(f"\n  Saved blended data to {os.path.abspath(out_path)}")

    # 4. Validate against fWAR
    validate_against_fwar(blended)

    elapsed = time.perf_counter() - t_start
    print(f"\n  Done in {elapsed:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
