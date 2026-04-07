"""Verify AQI is properly orthogonalized against the hitting component.

Computes BRAVS for a set of player-seasons and regresses AQI runs
against hitting runs. If the correlation is > 0.15, the proxy model
is still leaking value from the hitting component and needs further
dampening.

Expected result: r < 0.15 (preferably < 0.10)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy import stats as sp_stats

from baseball_metric.core.model import compute_bravs
from baseball_metric.run import NOTABLE_SEASONS


def main() -> None:
    hitting_runs = []
    aqi_runs = []
    names = []

    print("Computing BRAVS for notable seasons...\n")

    for player in NOTABLE_SEASONS:
        if player.pa < 100:
            continue  # Skip pitchers without batting

        result = compute_bravs(player, fast=True)

        h = result.components.get("hitting")
        a = result.components.get("approach_quality")
        if h is None or a is None:
            continue

        hitting_runs.append(h.runs_mean)
        aqi_runs.append(a.runs_mean)
        names.append(f"{player.player_name} {player.season}")

    h_arr = np.array(hitting_runs)
    a_arr = np.array(aqi_runs)

    # Pearson correlation
    r, p_value = sp_stats.pearsonr(h_arr, a_arr)

    print(f"{'Player':<30} {'Hitting':>8} {'AQI':>8}")
    print("-" * 50)
    for name, h, a in zip(names, hitting_runs, aqi_runs):
        print(f"{name:<30} {h:>8.1f} {a:>8.1f}")

    print(f"\n{'=' * 50}")
    print(f"  AQI-Hitting Correlation: r = {r:.3f} (p = {p_value:.4f})")
    print(f"  N = {len(hitting_runs)} player-seasons")
    print(f"{'=' * 50}")

    if abs(r) < 0.10:
        print("  PASS: AQI is well-orthogonalized (|r| < 0.10)")
    elif abs(r) < 0.20:
        print("  MARGINAL: AQI has slight correlation (0.10 < |r| < 0.20)")
        print("  Consider further dampening proxy coefficients")
    else:
        print(f"  FAIL: AQI still correlates with hitting (|r| = {abs(r):.3f})")
        print("  The proxy model needs further orthogonalization")
        print("  Recommended: reduce proxy scale factor or add wOBA-residual penalty")

    # Regression details
    slope, intercept, _, _, stderr = sp_stats.linregress(h_arr, a_arr)
    print(f"\n  Regression: AQI = {slope:.4f} * Hitting + {intercept:.2f}")
    print(f"  Slope SE: {stderr:.4f}")
    print(f"  For a 50-run hitter: predicted AQI = {slope * 50 + intercept:.1f}")
    print(f"  For a 100-run hitter: predicted AQI = {slope * 100 + intercept:.1f}")

    # Save to log
    with open("logs/aqi_orthogonality_check.log", "w") as f:
        f.write(f"AQI Orthogonality Verification\n")
        f.write(f"Correlation: r = {r:.3f}, p = {p_value:.4f}\n")
        f.write(f"N = {len(hitting_runs)}\n")
        f.write(f"Regression: AQI = {slope:.4f} * Hitting + {intercept:.2f}\n")
        f.write(f"Status: {'PASS' if abs(r) < 0.10 else 'MARGINAL' if abs(r) < 0.20 else 'FAIL'}\n")


if __name__ == "__main__":
    main()
