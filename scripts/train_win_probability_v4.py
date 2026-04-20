"""Win Probability Model v4 — Empirical lookup table with REAL game outcomes.

v3 had the right idea (empirical lookup) but used reconstructed home_won
which was wrong because our run estimation is approximate. v4 joins actual
game outcomes from Retrosheet gamelogs so the win labels are ground truth.

State space:
- inning: 1-12+ (12 buckets)
- half: T / B (2)
- outs: 0, 1, 2 (3)
- score_diff_bat: clamped -10 to +10 (21)
- bases: 0-7 (8 states)
= 12,096 possible states; ~140 plays/state with 1.7M samples
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from train_win_probability_v2 import (
    reconstruct_states, parse_advances, apply_event_to_state,
    base_state_index,
)


def load_real_outcomes() -> dict:
    """Return map game_id -> home_won (0/1) from Retrosheet gamelogs."""
    gl = pd.read_csv("data/retrosheet/game_logs_parsed.csv")
    gl["date_str"] = gl.date.astype(str)
    # Retrosheet game_id = HOMEYYYYMMDD<game_num>. For doubleheaders the
    # gamelogs list both games on the same date; assign game_num by order.
    gl = gl.sort_values(["date_str", "home_team"]).reset_index(drop=True)
    gl["game_num"] = gl.groupby(["date_str", "home_team"]).cumcount()
    gl["game_id"] = gl.home_team + gl.date_str + gl.game_num.astype(str)
    return dict(zip(gl.game_id, gl.home_won.astype(int)))


def build_empirical_table(states: pd.DataFrame) -> dict:
    print(f"\nBuilding empirical WP table from {len(states):,} states...")

    states = states.copy()
    states["inning_bucket"] = np.minimum(states.inning.values, 12)
    bat_diff = np.where(
        states.is_home_batting.values == 1,
        states.score_diff_home.values,
        -states.score_diff_home.values,
    )
    states["score_diff_bat"] = np.clip(bat_diff, -10, 10)
    states["bat_won"] = np.where(
        states.is_home_batting.values == 1,
        states.home_won.values,
        1.0 - states.home_won.values,
    )

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

    table = {}
    for row in grouped.itertuples(index=False):
        key = (int(row.inning_bucket), int(row.is_home_batting), int(row.outs),
               int(row.score_diff_bat), int(row.bases_idx))
        table[key] = (int(row.wins), int(row.total))
    return table


def lookup_wp(table: dict, inning: int, is_bot: int, outs: int,
              score_diff_bat: int, bases_idx: int,
              prior_weight: float = 20.0) -> float:
    """Empirical WP with score-aware Bayesian prior.

    The prior is the batting team's unconditional win rate at that score
    differential (not 0.5), which is a much better default for sparse states.
    """
    inning_b = min(inning, 12)
    diff_c = max(-10, min(10, score_diff_bat))

    # Score-aware prior: rough Pythagorean prior on score diff alone
    # These are calibrated to historical base rates.
    diff_prior = {
        -10: 0.02, -9: 0.03, -8: 0.04, -7: 0.05, -6: 0.07, -5: 0.10,
        -4: 0.14, -3: 0.19, -2: 0.26, -1: 0.35, 0: 0.50,
        1: 0.65, 2: 0.74, 3: 0.81, 4: 0.86, 5: 0.90,
        6: 0.93, 7: 0.95, 8: 0.96, 9: 0.97, 10: 0.98,
    }
    prior = diff_prior[diff_c]

    key = (inning_b, is_bot, outs, diff_c, bases_idx)
    if key in table:
        wins, total = table[key]
        return (wins + prior * prior_weight) / (total + prior_weight)

    # Aggregate neighbors: same (inning, half, diff) across all (outs, bases)
    wins_acc, total_acc = 0, 0
    for o in range(3):
        for b in range(8):
            k = (inning_b, is_bot, o, diff_c, b)
            if k in table:
                w, t = table[k]
                wins_acc += w
                total_acc += t
    if total_acc > 0:
        return (wins_acc + prior * prior_weight) / (total_acc + prior_weight)
    return prior


def validate_table(table: dict):
    print("\n--- Sanity Checks ---")
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
        ("Top 3, away up 1, 1 out, runner on 2B",  3, 0, 1,  1, 2),
        ("Bot 8, tied, 2 out, runner on 3B",       8, 1, 2,  0, 4),
    ]
    for label, inn, is_bot, outs, diff, bases in tests:
        key = (inn, is_bot, outs, diff, bases)
        wp = lookup_wp(table, inn, is_bot, outs, diff, bases)
        tag = f"n={table[key][1]}" if key in table else "smoothed"
        print(f"  {label:<50} WP={wp:>5.1%}  ({tag})")


def main():
    print("=" * 72)
    print("  WIN PROBABILITY MODEL v4 — Empirical + Real Outcomes")
    print("=" * 72)

    print("\nLoading events...")
    plays = pd.read_parquet("data/retrosheet/events_parsed.parquet")
    plays = plays[plays.event_simple.isin(
        ["HR", "HIT", "WALK", "HBP", "OUT", "K", "ERROR", "FC"]
    )].copy()
    print(f"  {len(plays):,} plays")

    print("\nLoading real game outcomes from gamelogs...")
    outcome_map = load_real_outcomes()
    print(f"  {len(outcome_map):,} game outcomes loaded")

    # Filter plays to games with known outcomes
    plays = plays[plays.game_id.isin(outcome_map)].copy()
    print(f"  After outcome match: {len(plays):,} plays, {plays.game_id.nunique():,} games")

    print("\nReconstructing states...")
    states = reconstruct_states(plays)
    # Overwrite home_won with REAL outcome
    states["home_won"] = states.game_id.map(outcome_map).astype(float)
    states = states.dropna(subset=["home_won"])
    print(f"  {len(states):,} valid states with real outcomes")

    # Also drop invalid late-game home-batting-while-ahead states (reconstructed
    # score is approx; the real filter using actual score can't be applied here,
    # but v2 already does a rough filter that removes some)
    table = build_empirical_table(states)
    validate_table(table)

    # ─── Baseline metrics over full dataset ───
    print("\n\nComputing baseline WP for each state...")
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

    base_pred = (baseline_wp > 0.5).astype(np.float32)
    base_acc = (base_pred == bat_won).mean()
    base_brier = ((baseline_wp - bat_won) ** 2).mean()
    base_bce = -(bat_won * np.log(baseline_wp.clip(1e-5, 1-1e-5)) +
                 (1 - bat_won) * np.log((1 - baseline_wp).clip(1e-5, 1-1e-5))).mean()

    print(f"\n  BASELINE (empirical lookup):")
    print(f"    Accuracy: {base_acc:.3%}")
    print(f"    Brier:    {base_brier:.4f}")
    print(f"    BCE:      {base_bce:.4f}")

    # ─── Save ───
    table_df = pd.DataFrame([
        {"inning": k[0], "is_bot": k[1], "outs": k[2],
         "score_diff": k[3], "bases_idx": k[4],
         "wins": v[0], "total": v[1],
         "win_rate": v[0] / v[1] if v[1] > 0 else 0.5}
        for k, v in sorted(table.items())
    ])
    os.makedirs("data/win_prob", exist_ok=True)
    table_df.to_csv("data/win_prob/empirical_wp_table_v4.csv", index=False)

    import pickle
    with open("data/win_prob/empirical_wp_table_v4.pkl", "wb") as f:
        pickle.dump(table, f)

    print(f"\n  Saved to data/win_prob/empirical_wp_table_v4.{{csv,pkl}}")
    print(f"  {len(table_df)} unique states covered")
    print("=" * 72)


if __name__ == "__main__":
    main()
