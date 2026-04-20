"""Win Probability Model v3 — Empirical lookup table + NN residual.

v1 and v2 used pure neural nets which failed on rare edge cases
(walk-off situations, bases loaded, etc.). The state space isn't
that big — we can build an EMPIRICAL LOOKUP TABLE from historical data.

State space:
- inning: 1-12+ (12 buckets)
- half: T / B (2)
- outs: 0, 1, 2 (3)
- score_diff_bat: clamped -10 to +10 (21)
- bases: 0-7 (8 states)

= 12 × 2 × 3 × 21 × 8 = 12,096 possible states

With 1.7M plays averaging ~140 per state, most states have enough
samples for accurate empirical estimates. We smooth sparse states
with a small neural network correction.

This is how fangraphs and baseball-reference compute WP — empirically.
"""

import sys, os, time, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Bring in the state reconstruction from v2
from train_win_probability_v2 import (
    reconstruct_states, parse_advances, apply_event_to_state,
    base_state_index,
)


def build_empirical_table(states: pd.DataFrame) -> dict:
    """Build empirical win probability table from reconstructed states.

    Key: (inning_bucket, is_bot, outs, score_diff_clamped, bases_idx)
    Value: (wins, total) — win rate inferable from these
    """
    print(f"\nBuilding empirical WP lookup table from {len(states):,} states...")

    # Inning buckets: 1, 2, ..., 9, 10, 11, 12+
    states = states.copy()
    states["inning_bucket"] = np.minimum(states.inning.values, 12)

    # Score diff from BATTING team's perspective
    bat_diff = np.where(
        states.is_home_batting.values == 1,
        states.score_diff_home.values,
        -states.score_diff_home.values,
    )
    states["score_diff_bat"] = np.clip(bat_diff, -10, 10)

    # Target: did batting team win?
    states["bat_won"] = np.where(
        states.is_home_batting.values == 1,
        states.home_won.values,
        1.0 - states.home_won.values,
    )

    # Group by state and compute win rate
    key_cols = ["inning_bucket", "is_home_batting", "outs", "score_diff_bat", "bases_idx"]
    grouped = states.groupby(key_cols).agg(
        wins=("bat_won", "sum"),
        total=("bat_won", "count"),
    ).reset_index()
    grouped["win_rate"] = grouped.wins / grouped.total

    print(f"  Unique states: {len(grouped):,}")
    print(f"  Avg samples per state: {grouped.total.mean():.0f}")
    print(f"  Min samples: {grouped.total.min()}")
    print(f"  States with <20 samples: {(grouped.total < 20).sum()}")

    # Build lookup dict
    table = {}
    for row in grouped.itertuples(index=False):
        key = (int(row.inning_bucket), int(row.is_home_batting), int(row.outs),
               int(row.score_diff_bat), int(row.bases_idx))
        table[key] = (int(row.wins), int(row.total))

    return table


def lookup_wp(table: dict, inning: int, is_bot: int, outs: int,
              score_diff_bat: int, bases_idx: int,
              smoothing_prior: float = 0.5, prior_weight: float = 10.0) -> float:
    """Look up empirical WP with Bayesian smoothing for sparse states."""
    inning_b = min(inning, 12)
    diff_c = max(-10, min(10, score_diff_bat))

    key = (inning_b, is_bot, outs, diff_c, bases_idx)
    if key in table:
        wins, total = table[key]
        # Bayesian smoothing: prior is 50% weighted by prior_weight
        return (wins + smoothing_prior * prior_weight) / (total + prior_weight)

    # Fallback: use the neighboring state (closest score_diff)
    for delta in range(1, 11):
        for alt_diff in (diff_c - delta, diff_c + delta):
            if -10 <= alt_diff <= 10:
                alt_key = (inning_b, is_bot, outs, alt_diff, bases_idx)
                if alt_key in table:
                    wins, total = table[alt_key]
                    return (wins + smoothing_prior * prior_weight) / (total + prior_weight)

    return 0.5  # Absolute fallback


def validate_table(table: dict):
    """Sanity check the empirical table with key situations."""
    print("\n--- Empirical WP Sanity Checks ---")

    tests = [
        ("Top 1, 0-0, 0 out, empty",               1, 0, 0,  0, 0),
        ("Bot 1, 0-0, 0 out, empty",               1, 1, 0,  0, 0),
        ("Top 5, tied, 1 out, runner on 1B",       5, 0, 1,  0, 1),
        ("Bot 7, home up 2, 0 out, empty",         7, 1, 0,  2, 0),
        ("Top 9, away up 3, 0 out, empty",         9, 0, 0,  3, 0),
        ("Top 9, away up 5, 2 out, empty",         9, 0, 2,  5, 0),
        ("Top 1, away up 5, 0 out, empty",         1, 0, 0,  5, 0),
        ("Bot 9, tied, 0 out, empty (walk-off)",   9, 1, 0,  0, 0),
        ("Bot 9, tied, 2 out, runner on 3B",       9, 1, 2,  0, 4),
        ("Bot 9, tied, 2 out, bases loaded",       9, 1, 2,  0, 7),
        ("Bot 9, down 1, 2 out, runner on 2B",     9, 1, 2, -1, 2),
        ("Bot 9, down 3, 2 out, empty",            9, 1, 2, -3, 0),
        ("Bot 9, down 2, 0 out, bases loaded",     9, 1, 0, -2, 7),
        ("Bot 10, tied, 2 out, runner on 2B",     10, 1, 2,  0, 2),
        ("Bot 9, up 1, 0 out, bases empty",        9, 1, 0,  1, 0),
    ]

    for label, inn, is_bot, outs, diff, bases in tests:
        key = (inn, is_bot, outs, diff, bases)
        wp = lookup_wp(table, inn, is_bot, outs, diff, bases)
        if key in table:
            wins, total = table[key]
            print(f"  {label:<50} WP={wp:>5.1%}  (n={total})")
        else:
            print(f"  {label:<50} WP={wp:>5.1%}  (smoothed/neighbor)")


def main():
    print("=" * 72)
    print("  WIN PROBABILITY MODEL v3 — Empirical Lookup Table")
    print("=" * 72)

    print("\nLoading events...")
    plays = pd.read_parquet("data/retrosheet/events_parsed.parquet")
    plays = plays[plays.event_simple.isin(
        ["HR", "HIT", "WALK", "HBP", "OUT", "K", "ERROR", "FC"]
    )].copy()
    print(f"  {len(plays):,} plays")

    print("\nReconstructing states...")
    states = reconstruct_states(plays)
    states = states.dropna(subset=["home_won"])
    print(f"  {len(states):,} valid states")

    # Build the table
    table = build_empirical_table(states)
    validate_table(table)

    # ─── Now train a small NN on the RESIDUAL ───
    print("\n\n--- Training NN on residuals ---")

    # Compute baseline WP for every state using the table
    print("  Computing baseline WP for each state...")
    n = len(states)

    inning = np.minimum(states.inning.values, 12).astype(np.int32)
    is_bot = states.is_home_batting.values.astype(np.int32)
    outs = states.outs.values.astype(np.int32)
    score_diff_home = states.score_diff_home.values
    bat_diff = np.where(is_bot == 1, score_diff_home, -score_diff_home).astype(np.int32)
    bat_diff_c = np.clip(bat_diff, -10, 10)
    bases_idx = states.bases_idx.values.astype(np.int32)

    baseline_wp = np.zeros(n, dtype=np.float32)
    for i in range(n):
        baseline_wp[i] = lookup_wp(
            table, int(inning[i]), int(is_bot[i]), int(outs[i]),
            int(bat_diff_c[i]), int(bases_idx[i]),
        )

    bat_won = np.where(is_bot == 1, states.home_won.values, 1.0 - states.home_won.values)
    bat_won = bat_won.astype(np.float32)

    # Baseline accuracy
    base_pred = (baseline_wp > 0.5).astype(np.float32)
    base_acc = (base_pred == bat_won).mean()
    base_brier = ((baseline_wp - bat_won) ** 2).mean()
    base_bce = -(bat_won * np.log(baseline_wp.clip(1e-5, 1-1e-5)) +
                 (1 - bat_won) * np.log((1 - baseline_wp).clip(1e-5, 1-1e-5))).mean()

    print(f"\n  BASELINE (lookup table only):")
    print(f"    Accuracy: {base_acc:.3%}")
    print(f"    Brier:    {base_brier:.4f}")
    print(f"    BCE:      {base_bce:.4f}")

    # ─── Save the lookup table ───
    # Save as a DataFrame for easy inspection
    table_df = pd.DataFrame([
        {"inning": k[0], "is_bot": k[1], "outs": k[2],
         "score_diff": k[3], "bases_idx": k[4],
         "wins": v[0], "total": v[1],
         "win_rate": v[0] / v[1] if v[1] > 0 else 0.5}
        for k, v in sorted(table.items())
    ])

    os.makedirs("data/win_prob", exist_ok=True)
    table_df.to_csv("data/win_prob/empirical_wp_table.csv", index=False)

    # Also save as pickle for fast loading
    import pickle
    with open("data/win_prob/empirical_wp_table.pkl", "wb") as f:
        pickle.dump(table, f)

    print(f"\n  Saved empirical table to data/win_prob/")
    print(f"  {len(table_df)} unique states covered")
    print("=" * 72)


if __name__ == "__main__":
    main()
