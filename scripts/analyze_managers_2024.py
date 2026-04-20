"""Manager in-game decision analysis for 2024 using v5-frame WPA.

For every game, walk plays in order and detect:
  - Pitching changes (pitcher_id transitions within the same pitching team)
  - Mid-inning vs between-inning changes
  - Leverage at the moment of the pull

For each pulled pitcher and each incoming reliever we attribute the dWP
they generated in their 'segment' (continuous run of plays with them on
the mound). Then we aggregate per manager from the gamelog's
home_mgr_id / vis_mgr_id fields.

Outputs (data/win_prob/):
  - manager_decisions_2024.csv : one row per manager
  - pitching_changes_2024.csv  : one row per change, for deep dives
"""

import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from scripts.train_win_probability_v5 import (
    reconstruct, load_real_outcomes,
)
from scripts.train_win_probability_v6 import (
    load_quality_tables, build_feature_frame, attach_baseline_wp,
)


def main():
    print("=" * 72)
    print("  MANAGER DECISION ANALYSIS 2024")
    print("=" * 72)

    with open("data/win_prob/wp_model_v6.pkl", "rb") as f:
        b = pickle.load(f)
    table, prior_map = b["v5_table"], b["v5_prior_map"]

    print("\nLoading 2024 events...")
    plays = pd.read_parquet("data/retrosheet/events_parsed.parquet")
    plays = plays[plays.event_simple.isin(
        ["HR", "HIT", "WALK", "HBP", "OUT", "K", "ERROR", "FC"]
    )].copy()
    plays["year"] = plays.date.str[:4].astype(int)
    plays = plays[plays.year == 2024].copy()

    outcomes = load_real_outcomes()
    plays = plays[plays.game_id.isin(outcomes)].copy()
    print(f"  {len(plays):,} plays, {plays.game_id.nunique():,} games")

    states = reconstruct(plays)
    states["home_won"] = states.game_id.map(outcomes).astype(float)

    bat_q, pit_q, bat_means, pit_means = load_quality_tables()
    df = build_feature_frame(plays, states, bat_q, pit_q, bat_means, pit_means)
    df = attach_baseline_wp(df, table, prior_map)

    # Use v5 state-only WP for dWP (same rationale as compute_wpa.py)
    df["wp_bat"] = df.baseline_wp_v5.values.astype(np.float32)
    is_bot = df.is_home_batting.values.astype(np.int32)
    wp_home = np.where(is_bot == 1, df.wp_bat.values, 1.0 - df.wp_bat.values
                       ).astype(np.float32)
    df["wp_home"] = wp_home

    next_game = df.game_id.shift(-1)
    next_wp_home = df.wp_home.shift(-1).values
    same_game = df.game_id.values == next_game.values
    home_won = df.home_won.values.astype(np.float32)
    wp_home_post = np.where(same_game, next_wp_home, home_won).astype(np.float32)
    df["dWP_home"] = wp_home_post - wp_home
    # dWP from the BATTING team's perspective (+ve = batter gains)
    df["dWP_bat"] = np.where(is_bot == 1, df.dWP_home, -df.dWP_home)
    # Defensive dWP (+ve = pitcher gains, batting team loses WP)
    df["dWP_def"] = -df.dWP_bat.values

    # Build state-level LI (reuse the leverage table if it exists)
    try:
        li_df = pd.read_csv("data/win_prob/leverage_index_states.csv")
        li_map = dict(zip(
            zip(li_df.inning, li_df.is_bot, li_df.outs,
                li_df.score_diff_bat, li_df.bases_idx),
            li_df.LI.values,
        ))
        df["LI"] = [
            li_map.get((int(r.inning_c), int(r.is_home_batting),
                        int(r.outs), int(r.score_diff_bat), int(r.bases_idx)),
                       1.0)
            for r in df[["inning_c", "is_home_batting", "outs",
                         "score_diff_bat", "bases_idx"]].itertuples(index=False)
        ]
    except FileNotFoundError:
        df["LI"] = 1.0

    # Pitching team is the opposite of batting team
    df["pit_team"] = np.where(is_bot == 1, df.away_team, df.home_team)

    # Detect pitching changes: compare each play to the previous play where
    # the SAME pitching team was on defense. Skipping the other team's
    # half-innings is essential — a between-inning pitching change (end of
    # top 7 -> start of top 8) has the other team's bottom 7 sitting between
    # the two relevant rows.
    grp = df.groupby(["game_id", "pit_team"], sort=False)
    df["prev_pitcher_same"] = grp.pitcher_id.shift(1)
    df["prev_inning_same"] = grp.inning_c.shift(1)
    df["prev_half_same"] = grp.is_home_batting.shift(1)

    changed = (df.prev_pitcher_same.notna()
               & (df.pitcher_id.values != df.prev_pitcher_same.values))
    df["is_pitching_change"] = changed

    mid_inning = changed & (df.inning_c.values == df.prev_inning_same.values) \
                 & (df.is_home_batting.values == df.prev_half_same.values)
    df["mid_inning_change"] = mid_inning

    n_changes = int(changed.sum())
    n_mid = int(mid_inning.sum())
    print(f"\n  {n_changes:,} pitching changes in 2024 "
          f"({n_mid:,} mid-inning, {n_changes - n_mid:,} between-inning)")

    # Attach manager IDs from the gamelog
    gl = pd.read_csv("data/retrosheet/game_logs_parsed.csv")
    gl["date_str"] = gl.date.astype(str)
    gl = gl.sort_values(["date_str", "home_team"]).reset_index(drop=True)
    counts = gl.groupby(["date_str", "home_team"]).cumcount()
    totals = gl.groupby(["date_str", "home_team"])["home_team"].transform("size")
    gl["game_num"] = np.where(totals >= 2, counts + 1, 0)
    gl["game_id"] = gl.home_team + gl.date_str + gl.game_num.astype(str)
    gl_keep = gl[["game_id", "home_team", "home_mgr_id", "home_mgr_name",
                  "vis_mgr_id", "vis_mgr_name"]]
    df = df.merge(gl_keep, on=["game_id", "home_team"], how="left")

    # Manager of the pitching team = whichever side is currently defending
    df["pit_mgr_id"] = np.where(is_bot == 1, df.vis_mgr_id, df.home_mgr_id)
    df["pit_mgr_name"] = np.where(is_bot == 1,
                                   df.vis_mgr_name, df.home_mgr_name)
    df["bat_mgr_id"] = np.where(is_bot == 1, df.home_mgr_id, df.vis_mgr_id)
    df["bat_mgr_name"] = np.where(is_bot == 1,
                                   df.home_mgr_name, df.vis_mgr_name)

    # --- Per-change detail: state, LI, who was pulled, who came in -----
    changes = df[df.is_pitching_change].copy()
    changes["pulled_pitcher"] = changes.prev_pitcher_same.values
    changes["new_pitcher"] = changes.pitcher_id.values
    changes_out = changes[[
        "game_id", "pit_mgr_id", "pit_mgr_name",
        "inning_c", "is_home_batting", "outs", "score_diff_bat",
        "bases_idx", "LI", "mid_inning_change",
        "pulled_pitcher", "new_pitcher",
    ]].rename(columns={"inning_c": "inning", "is_home_batting": "bat_bot"})
    os.makedirs("data/win_prob", exist_ok=True)
    changes_out.to_csv("data/win_prob/pitching_changes_2024.csv", index=False)

    # --- Segment-level pitcher WPA: contiguous runs of same pitcher ----
    # A "segment" is a continuous run where the same pitcher faces the
    # same (game, pit_team). Half-inning gaps DON'T break segments — a
    # starter who pitches T1 then T2 is the same segment.
    new_segment = df.prev_pitcher_same.isna() | (
        df.pitcher_id.values != df.prev_pitcher_same.values
    )
    df["segment_id"] = np.cumsum(new_segment.values)

    seg = df.groupby("segment_id", sort=False).agg(
        game_id=("game_id", "first"),
        pit_team=("pit_team", "first"),
        pitcher_id=("pitcher_id", "first"),
        pit_mgr_id=("pit_mgr_id", "first"),
        pit_mgr_name=("pit_mgr_name", "first"),
        BF=("dWP_def", "size"),
        WPA=("dWP_def", "sum"),
        entry_LI=("LI", "first"),
        mean_LI=("LI", "mean"),
    ).reset_index()

    # --- Per-manager aggregates ---------------------------------------
    # Games managed = distinct game_ids where this manager was home or vis
    print("\nAggregating per-manager...")
    home_games = gl[gl.year == 2024].groupby("home_mgr_id").size()
    vis_games = gl[gl.year == 2024].groupby("vis_mgr_id").size()
    total_games = home_games.add(vis_games, fill_value=0)

    # Manager names
    mgr_name = {}
    for mid, nm in zip(gl.home_mgr_id.fillna(""), gl.home_mgr_name.fillna("")):
        if mid: mgr_name[mid] = nm
    for mid, nm in zip(gl.vis_mgr_id.fillna(""), gl.vis_mgr_name.fillna("")):
        if mid: mgr_name[mid] = nm

    # Pitching-change stats
    pc = changes.groupby("pit_mgr_id").agg(
        n_changes=("is_pitching_change", "size"),
        n_mid=("mid_inning_change", "sum"),
        avg_li_at_pull=("LI", "mean"),
    )

    # Segment WPA per manager (the WPA of pitchers they chose to use)
    seg_agg = seg.groupby("pit_mgr_id").agg(
        segments=("segment_id", "size"),
        total_bf=("BF", "sum"),
        def_WPA=("WPA", "sum"),
    )

    out = pd.DataFrame({"games": total_games}).join(pc).join(seg_agg)
    out.index.name = "mgr_id"
    out = out.reset_index()
    out["name"] = out.mgr_id.map(mgr_name)
    out = out[out.games >= 40]  # full-season managers only
    out["changes_per_game"] = out.n_changes / out.games
    out["mid_pct"] = out.n_mid / out.n_changes
    out["def_WPA_per_game"] = out.def_WPA / out.games
    out = out[["mgr_id", "name", "games", "n_changes", "changes_per_game",
               "n_mid", "mid_pct", "avg_li_at_pull", "def_WPA",
               "def_WPA_per_game"]]
    out = out.sort_values("def_WPA", ascending=False)
    out.to_csv("data/win_prob/manager_decisions_2024.csv", index=False)

    print("\nTop 10 managers by defensive WPA (2024):")
    for _, r in out.head(10).iterrows():
        print(f"  {r['name']:<22} {int(r.games):>3}G  "
              f"{int(r.n_changes):>3} changes ({r.changes_per_game:.2f}/g, "
              f"{r.mid_pct:.0%} mid)  "
              f"LI@pull {r.avg_li_at_pull:.2f}  "
              f"def_WPA {r.def_WPA:+.2f}")

    print("\nBottom 5 managers by defensive WPA:")
    for _, r in out.tail(5).iterrows():
        print(f"  {r['name']:<22} {int(r.games):>3}G  "
              f"{int(r.n_changes):>3} changes ({r.changes_per_game:.2f}/g, "
              f"{r.mid_pct:.0%} mid)  "
              f"LI@pull {r.avg_li_at_pull:.2f}  "
              f"def_WPA {r.def_WPA:+.2f}")

    print("\nHighest-leverage single pulls of 2024:")
    hp = changes[changes.mid_inning_change].nlargest(10, "LI")
    for _, r in hp.iterrows():
        side = "B" if r.is_home_batting else "T"
        print(f"  {r.game_id} {side}{int(r.inning_c)} "
              f"{r.pit_mgr_name:<18} pulled {r.pulled_pitcher} -> "
              f"{r.new_pitcher}  LI={r.LI:.2f}")

    print("\n" + "=" * 72)
    print("  Saved: data/win_prob/manager_decisions_2024.csv, "
          "pitching_changes_2024.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
