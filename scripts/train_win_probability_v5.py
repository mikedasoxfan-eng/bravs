"""Win Probability Model v5 — Strict run counting + real outcomes.

v4 improvements:
- Runs counted ONLY from advance tokens (.X-H) + HR batter rule
  This captures 98.5% of actual runs vs ~70% with heuristics
- Real game outcomes from Retrosheet gamelogs
- Base state still tracked via advance parsing + heuristics (for lookup key)
- Outs tracked correctly via event type

State space unchanged: 12 × 2 × 3 × 21 × 8 = 12,096 keys
"""

import sys, os, time, re, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

ADV_SCORE = re.compile(r"([B123])-H")
ADV_ALL = re.compile(r"([B123])-([123H])")


def runs_on_play(event_raw: str, event_simple: str) -> int:
    """Strictly count runs scored on this play using advance tokens + HR rule."""
    r = 0
    if event_simple == "HR":
        r += 1  # batter scores on HR
    if "." in event_raw:
        adv_block = event_raw.split(".", 1)[1]
        r += len(ADV_SCORE.findall(adv_block))
    return r


def update_bases(event_raw: str, event_simple: str, bases: list[int]) -> list[int]:
    """Compute new base state from advance tokens + event fallback."""
    new_bases = [0, 0, 0]

    # Parse advance tokens if present
    advances = []
    if "." in event_raw:
        adv_block = event_raw.split(".", 1)[1]
        advances = ADV_ALL.findall(adv_block)

    # Track which prior runners moved
    moved = {"1": False, "2": False, "3": False}
    batter_placed = False

    for runner, dest in advances:
        if runner == "B":
            if dest != "H":
                new_bases[int(dest) - 1] = 1
            batter_placed = True
        else:
            moved[runner] = True
            if dest != "H":
                new_bases[int(dest) - 1] = 1

    # Any prior runners not explicitly moved stay put
    for i, label in enumerate(("1", "2", "3")):
        if not moved[label] and bases[i] == 1:
            # Don't clobber a newly placed runner
            if new_bases[i] == 0:
                new_bases[i] = 1

    # Place batter if not already placed via advance token
    if not batter_placed:
        if event_simple == "HR":
            pass  # batter scored
        elif event_simple in ("HIT", "WALK", "HBP", "ERROR", "FC"):
            # Default: batter to 1B (singles, walks, HBP, errors, FC)
            # For walk/HBP, force runners if needed
            if event_simple in ("WALK", "HBP"):
                # Walk forces only if 1B occupied
                if bases[0] == 1 and new_bases[0] == 1:
                    # 1B already has runner who stayed — walk forces
                    new_bases[0] = 1
                    if bases[1] == 1 and new_bases[1] == 1:
                        new_bases[1] = 1
                        if bases[2] == 1 and new_bases[2] == 1:
                            # bases loaded walk — runner on 3B scores (counted via advance ideally)
                            pass
                        else:
                            new_bases[2] = 1
                    else:
                        new_bases[1] = 1
                new_bases[0] = 1
            elif event_simple == "HIT":
                # If no advance string but it's a hit, infer base from core token
                core = event_raw.split("/")[0].strip()
                if core.startswith("D"):
                    new_bases[1] = 1
                elif core.startswith("T"):
                    new_bases[2] = 1
                else:
                    new_bases[0] = 1
            else:
                new_bases[0] = 1

    return new_bases


def bases_idx(bases: list[int]) -> int:
    return bases[0] + bases[1] * 2 + bases[2] * 4


def reconstruct(plays: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct state for each play using strict advance-based run counting."""
    print(f"Reconstructing {len(plays):,} plays with strict advance parsing...")
    t0 = time.perf_counter()

    plays = plays.copy()
    plays["half_ord"] = plays.half.map({"T": 0, "B": 1})
    plays = plays.sort_values(
        ["game_id", "inning", "half_ord", "outs_before"], kind="stable"
    )

    rows = []
    for game_id, gp in plays.groupby("game_id", sort=False):
        home_score = 0
        away_score = 0
        bases = [0, 0, 0]
        prev_key = None

        for row in gp.itertuples(index=False):
            k = (row.inning, row.half)
            if k != prev_key:
                bases = [0, 0, 0]
                prev_key = k

            is_home_bat = 1 if row.half == "B" else 0

            # Pre-play state
            rows.append({
                "game_id": game_id,
                "inning": int(row.inning),
                "is_home_batting": is_home_bat,
                "outs": int(row.outs_before),
                "score_diff_home": home_score - away_score,
                "bases_idx": bases_idx(bases),
            })

            # Apply event
            r = runs_on_play(row.event_raw, row.event_simple)
            if is_home_bat:
                home_score += r
            else:
                away_score += r
            bases = update_bases(row.event_raw, row.event_simple, bases)

    print(f"  {len(rows):,} states in {time.perf_counter()-t0:.1f}s")
    return pd.DataFrame(rows)


def load_real_outcomes() -> dict:
    gl = pd.read_csv("data/retrosheet/game_logs_parsed.csv")
    gl["date_str"] = gl.date.astype(str)
    gl = gl.sort_values(["date_str", "home_team"]).reset_index(drop=True)
    # Retrosheet convention: single game -> suffix '0', DH -> '1' and '2'.
    counts = gl.groupby(["date_str", "home_team"]).cumcount()
    totals = gl.groupby(["date_str", "home_team"])["home_team"].transform("size")
    gl["game_num"] = np.where(totals >= 2, counts + 1, 0)
    gl["game_id"] = gl.home_team + gl.date_str + gl.game_num.astype(str)
    return dict(zip(gl.game_id, gl.home_won.astype(int)))


def build_table(states: pd.DataFrame) -> dict:
    print(f"\nBuilding table from {len(states):,} states...")
    s = states.copy()
    s["inning_b"] = np.minimum(s.inning.values, 12)
    bat_diff = np.where(
        s.is_home_batting.values == 1,
        s.score_diff_home.values,
        -s.score_diff_home.values,
    )
    s["score_diff_bat"] = np.clip(bat_diff, -10, 10)
    s["bat_won"] = np.where(
        s.is_home_batting.values == 1,
        s.home_won.values,
        1.0 - s.home_won.values,
    )

    g = s.groupby(
        ["inning_b", "is_home_batting", "outs", "score_diff_bat", "bases_idx"]
    ).agg(wins=("bat_won", "sum"), total=("bat_won", "count")).reset_index()

    print(f"  Unique states: {len(g):,}")
    print(f"  Avg samples: {g.total.mean():.0f}")
    print(f"  States <20 samples: {(g.total < 20).sum()}")

    table = {}
    for row in g.itertuples(index=False):
        key = (int(row.inning_b), int(row.is_home_batting), int(row.outs),
               int(row.score_diff_bat), int(row.bases_idx))
        table[key] = (int(row.wins), int(row.total))
    return table


def build_prior(table: dict) -> dict:
    """Aggregate win rate by (inning, half, diff) across all outs/bases.

    This gives a context-aware smoothing target that captures home-field
    advantage and inning-specific dynamics, unlike a static diff-only prior.
    """
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for (inn, is_bot, outs, diff, b), (w, t) in table.items():
        k = (inn, is_bot, diff)
        agg[k][0] += w
        agg[k][1] += t
    # Also per (inning, diff) ignoring half, for double-fallback
    agg_diff = defaultdict(lambda: [0, 0])
    for (inn, is_bot, diff), (w, t) in agg.items():
        agg_diff[diff][0] += w
        agg_diff[diff][1] += t
    return {
        "ihd": {k: v[0] / v[1] for k, v in agg.items() if v[1] > 0},
        "d":   {k: v[0] / v[1] for k, v in agg_diff.items() if v[1] > 0},
    }


def lookup_wp(table, prior_map, inning, is_bot, outs, diff, bases,
              prior_weight=15.0):
    inning_b = min(inning, 12)
    diff_c = max(-10, min(10, diff))

    # Context-aware prior
    p = prior_map["ihd"].get((inning_b, is_bot, diff_c))
    if p is None:
        p = prior_map["d"].get(diff_c, 0.5)

    key = (inning_b, is_bot, outs, diff_c, bases)
    if key in table:
        w, t = table[key]
        return (w + p * prior_weight) / (t + prior_weight)

    # Fallback: aggregate across outs+bases at same (inning, half, diff)
    wa, ta = 0, 0
    for o in range(3):
        for b in range(8):
            k = (inning_b, is_bot, o, diff_c, b)
            if k in table:
                w, t = table[k]
                wa += w
                ta += t
    if ta > 0:
        return (wa + p * prior_weight) / (ta + prior_weight)
    return p


def validate(table, prior_map):
    print("\n--- Sanity Checks ---")
    tests = [
        ("Top 1, 0-0, 0 out, empty",               1, 0, 0,  0, 0),
        ("Bot 1, 0-0, 0 out, empty",               1, 1, 0,  0, 0),
        ("Top 5, tied, 1 out, runner on 1B",       5, 0, 1,  0, 1),
        ("Bot 7, home up 2, 0 out, empty",         7, 1, 0,  2, 0),
        ("Top 9, away up 3, 0 out, empty",         9, 0, 0,  3, 0),
        ("Top 9, away up 5, 2 out, empty",         9, 0, 2,  5, 0),
        ("Top 1, away up 5, 0 out, empty",         1, 0, 0,  5, 0),
        ("Bot 9, tied, 0 out, empty",              9, 1, 0,  0, 0),
        ("Bot 9, tied, 0 out, runner on 1B",       9, 1, 0,  0, 1),
        ("Bot 9, tied, 2 out, runner on 3B",       9, 1, 2,  0, 4),
        ("Bot 9, tied, 2 out, bases loaded",       9, 1, 2,  0, 7),
        ("Bot 9, tied, 1 out, bases loaded",       9, 1, 1,  0, 7),
        ("Bot 9, down 1, 2 out, runner on 2B",     9, 1, 2, -1, 2),
        ("Bot 9, down 3, 2 out, empty",            9, 1, 2, -3, 0),
        ("Bot 9, down 2, 0 out, bases loaded",     9, 1, 0, -2, 7),
        ("Bot 10, tied, 2 out, runner on 2B",     10, 1, 2,  0, 2),
        ("Top 3, away up 1, 1 out, runner on 2B",  3, 0, 1,  1, 2),
        ("Bot 8, tied, 2 out, runner on 3B",       8, 1, 2,  0, 4),
    ]
    for label, inn, is_bot, outs, diff, b in tests:
        k = (inn, is_bot, outs, diff, b)
        wp = lookup_wp(table, prior_map, inn, is_bot, outs, diff, b)
        tag = f"n={table[k][1]}" if k in table else "fallback"
        print(f"  {label:<48} WP={wp:>5.1%}  ({tag})")


def main():
    print("=" * 72)
    print("  WIN PROBABILITY MODEL v5 — Strict Reconstruction")
    print("=" * 72)

    print("\nLoading events...")
    plays = pd.read_parquet("data/retrosheet/events_parsed.parquet")
    plays = plays[plays.event_simple.isin(
        ["HR", "HIT", "WALK", "HBP", "OUT", "K", "ERROR", "FC"]
    )].copy()
    print(f"  {len(plays):,} plays")

    print("\nLoading real game outcomes...")
    outcomes = load_real_outcomes()
    plays = plays[plays.game_id.isin(outcomes)].copy()
    print(f"  Matched: {len(plays):,} plays, {plays.game_id.nunique():,} games")

    states = reconstruct(plays)
    states["home_won"] = states.game_id.map(outcomes).astype(float)

    # Validate reconstruction: compare final scores where possible (via gamelogs)
    # We can't directly but can check total runs
    print("\n  Verifying run counts...")
    total_runs = 0
    for game_id, gp in states.groupby("game_id"):
        # Get the last play in each half-inning approximately; safer to just
        # pick max of home_score+away_score... but we only have pre-play states.
        pass
    # Instead, verify via the plays themselves
    plays["runs"] = [runs_on_play(e, s) for e, s in zip(plays.event_raw, plays.event_simple)]
    gl = pd.read_csv("data/retrosheet/game_logs_parsed.csv")
    gl = gl[gl.year.between(2015, 2024)]
    real_total = gl.vis_runs.sum() + gl.home_runs.sum()
    parsed_total = plays.runs.sum()
    print(f"  Parsed runs: {parsed_total:,} / Real runs: {real_total:,} ({parsed_total/real_total:.1%})")

    table = build_table(states)
    prior_map = build_prior(table)
    print(f"  Context-aware priors: {len(prior_map['ihd'])} (inning,half,diff) cells")
    validate(table, prior_map)

    # Full metrics
    print("\nComputing baseline WP for each state...")
    n = len(states)
    inning = np.minimum(states.inning.values, 12).astype(np.int32)
    is_bot = states.is_home_batting.values.astype(np.int32)
    outs = states.outs.values.astype(np.int32)
    sd = states.score_diff_home.values
    bat_diff = np.where(is_bot == 1, sd, -sd).astype(np.int32)
    bat_diff_c = np.clip(bat_diff, -10, 10)
    bi = states.bases_idx.values.astype(np.int32)

    wp_arr = np.empty(n, dtype=np.float32)
    for i in range(n):
        wp_arr[i] = lookup_wp(table, prior_map, int(inning[i]), int(is_bot[i]),
                              int(outs[i]), int(bat_diff_c[i]), int(bi[i]))

    bat_won = np.where(is_bot == 1, states.home_won.values,
                       1.0 - states.home_won.values).astype(np.float32)

    acc = ((wp_arr > 0.5).astype(np.float32) == bat_won).mean()
    brier = ((wp_arr - bat_won) ** 2).mean()
    bce = -(bat_won * np.log(wp_arr.clip(1e-5, 1-1e-5)) +
            (1 - bat_won) * np.log((1 - wp_arr).clip(1e-5, 1-1e-5))).mean()
    print(f"\n  BASELINE:")
    print(f"    Accuracy: {acc:.3%}")
    print(f"    Brier:    {brier:.4f}")
    print(f"    BCE:      {bce:.4f}")

    # Save
    os.makedirs("data/win_prob", exist_ok=True)
    with open("data/win_prob/empirical_wp_table_v5.pkl", "wb") as f:
        pickle.dump({"table": table, "prior_map": prior_map}, f)

    df = pd.DataFrame([
        {"inning": k[0], "is_bot": k[1], "outs": k[2], "score_diff": k[3],
         "bases_idx": k[4], "wins": v[0], "total": v[1],
         "win_rate": v[0]/v[1] if v[1] else 0.5}
        for k, v in sorted(table.items())
    ])
    df.to_csv("data/win_prob/empirical_wp_table_v5.csv", index=False)
    print(f"\n  Saved v5 table ({len(df)} states)")
    print("=" * 72)


if __name__ == "__main__":
    main()
