"""Compute WPA leaderboards from the v6 win-probability model.

For every play 2015-2024:
  1. Predict pre-play WP from batting team POV using v6.
  2. Compute post-play WP = next row's pre-play WP (flipped if half changed)
     or the real game outcome if this is the last play of the game.
  3. dWP = post - pre; credit +dWP to the batter, -dWP to the pitcher.
  4. Build a state-level Leverage Index (mean |dWP| for that state /
     global mean |dWP|) so we can flag high-leverage plays.

Outputs (data/win_prob/):
  - top_plays_2015_2024.csv    : 100 biggest single-play swings
  - wpa_season_batters.csv     : per (playerID, year) batter WPA leaderboard
  - wpa_season_pitchers.csv    : per (playerID, year) pitcher WPA leaderboard
  - wpa_career_batters.csv     : sum over 2015-2024 (career window)
  - wpa_career_pitchers.csv    : sum over 2015-2024 (career window)
  - leverage_index_states.csv  : per-state LI table
"""

import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from scripts.train_win_probability_v5 import reconstruct, load_real_outcomes
from scripts.train_win_probability_v6 import (
    load_quality_tables, build_feature_frame, attach_baseline_wp,
    BAT_COLS, PIT_COLS,
)


def load_v6():
    with open("data/win_prob/wp_model_v6.pkl", "rb") as f:
        b = pickle.load(f)
    return b


def main():
    print("=" * 72)
    print("  WPA LEADERBOARDS from v6")
    print("=" * 72)

    bundle = load_v6()
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    table = bundle["v5_table"]
    prior_map = bundle["v5_prior_map"]
    print(f"\nLoaded v6 model ({len(feature_cols)} features)")

    print("\nLoading events 2015-2024...")
    plays = pd.read_parquet("data/retrosheet/events_parsed.parquet")
    plays = plays[plays.event_simple.isin(
        ["HR", "HIT", "WALK", "HBP", "OUT", "K", "ERROR", "FC"]
    )].copy()
    plays["year"] = plays.date.str[:4].astype(int)
    plays = plays[plays.year.between(2015, 2024)].copy()

    outcomes = load_real_outcomes()
    plays = plays[plays.game_id.isin(outcomes)].copy()
    print(f"  {len(plays):,} plays, {plays.game_id.nunique():,} games")

    states = reconstruct(plays)
    states["home_won"] = states.game_id.map(outcomes).astype(float)

    bat_q, pit_q, bat_means, pit_means = load_quality_tables()
    df = build_feature_frame(plays, states, bat_q, pit_q, bat_means, pit_means)
    df = attach_baseline_wp(df, table, prior_map)

    # Predict WP from batting team POV using the STATE-ONLY v5 table.
    # We deliberately use v5 (not v6) for WPA so that half-inning boundaries
    # are continuous in the home frame. v6's team-game features create
    # discontinuities at half boundaries (bat_team/pit_team flip + model
    # recalibration) that produce phantom dWP credited to whichever batter
    # happens to be at the plate at the moment of the jump. State-only WP
    # is the standard WPA convention for this reason.
    print("\nAssigning state-only WP (v5 table) for dWP computation...")
    df["wp_bat"] = df.baseline_wp_v5.values.astype(np.float32)

    # Play sort is already (game_id, inning, half_ord, outs_before) from
    # build_feature_frame. We need the batter_id / pitcher_id on each row too.
    # The concat in build_feature_frame already added them.
    assert "batter_id" in df.columns and "pitcher_id" in df.columns

    # Compute dWP per play in a FIXED home-team frame so the sum telescopes
    # cleanly across half-inning boundaries (no perspective flips in the
    # sum). Then assign signed WPA to batter/pitcher based on who's batting.
    print("\nComputing dWP per play...")
    is_bot = df.is_home_batting.values.astype(np.int32)
    wp_bat = df.wp_bat.values
    wp_home = np.where(is_bot == 1, wp_bat, 1.0 - wp_bat).astype(np.float32)
    df["wp_home"] = wp_home

    next_game = df.game_id.shift(-1)
    next_wp_home = df.wp_home.shift(-1).values
    same_game = (df.game_id.values == next_game.values)

    # wp_home_post: next row's home wp (same frame, no flip), or actual
    # home_won at game end.
    home_won = df.home_won.values.astype(np.float32)
    wp_home_post = np.where(same_game, next_wp_home, home_won).astype(np.float32)
    df["wp_home_post"] = wp_home_post
    dWP_home = (wp_home_post - wp_home).astype(np.float32)
    df["dWP_home"] = dWP_home

    # Batter WPA: the change in WP from BATTER's perspective. If home is
    # batting, dWP_home aligns with batter; if away is batting, flip.
    df["dWP"] = np.where(is_bot == 1, dWP_home, -dWP_home).astype(np.float32)
    df["wp_post"] = np.where(is_bot == 1, wp_home_post, 1.0 - wp_home_post
                             ).astype(np.float32)

    print(f"  mean |dWP|: {np.abs(df.dWP.values).mean():.4f}")
    print(f"  max |dWP|:  {np.abs(df.dWP.values).max():.4f}")
    # Telescoping sanity: sum of dWP_home per game should equal
    # home_won - wp_home[first_play_of_game].
    per_game = df.groupby("game_id", sort=False).agg(
        s=("dWP_home", "sum"),
        w=("home_won", "first"),
        w0=("wp_home", "first"),
    )
    residual = (per_game.s - (per_game.w - per_game.w0)).abs().mean()
    print(f"  telescoping residual per game: {residual:.6f} (should be ~0)")

    # --- Leverage Index per state --------------------------------------
    print("\nComputing state Leverage Index...")
    state_keys = pd.Series(list(zip(
        df.inning_c.values, df.is_home_batting.values.astype(int),
        df.outs.values.astype(int), df.score_diff_bat.values.astype(int),
        df.bases_idx.values.astype(int),
    )), index=df.index)
    abs_dwp = np.abs(df.dWP.values)
    state_li = (pd.DataFrame({"k": state_keys, "a": abs_dwp})
                .groupby("k")["a"].agg(["mean", "count"]))
    global_mean = abs_dwp.mean()
    state_li["LI"] = state_li["mean"] / global_mean
    state_li = state_li.reset_index()
    state_li[["inning", "is_bot", "outs", "score_diff_bat", "bases_idx"]] = \
        pd.DataFrame(state_li["k"].tolist(), index=state_li.index)
    state_li = state_li[["inning", "is_bot", "outs", "score_diff_bat",
                         "bases_idx", "mean", "count", "LI"]]
    state_li.to_csv("data/win_prob/leverage_index_states.csv", index=False)
    li_map = dict(zip(
        zip(state_li.inning, state_li.is_bot, state_li.outs,
            state_li.score_diff_bat, state_li.bases_idx),
        state_li.LI.values,
    ))
    df["LI"] = [li_map.get(k, 1.0) for k in state_keys]
    print(f"  {len(state_li):,} unique states, mean |dWP| = {global_mean:.4f}")

    # --- Player name lookup -------------------------------------------
    print("\nLoading player names...")
    people = pd.read_csv("data/lahman2025/People.csv",
                        usecols=["playerID", "retroID", "nameFirst",
                                 "nameLast"], low_memory=False)
    people = people.dropna(subset=["retroID"]).drop_duplicates("retroID")
    people["name"] = people.nameFirst.fillna("") + " " + people.nameLast.fillna("")
    retro_to_player = dict(zip(people.retroID, people.playerID))
    retro_to_name = dict(zip(people.retroID, people.name))

    df["batter_playerID"] = df.batter_id.map(retro_to_player)
    df["pitcher_playerID"] = df.pitcher_id.map(retro_to_player)
    df["batter_name"] = df.batter_id.map(retro_to_name)
    df["pitcher_name"] = df.pitcher_id.map(retro_to_name)

    # --- Top single-play swings ---------------------------------------
    print("\nTop 20 single-play WP swings 2015-2024:")
    top = df.assign(abs_dwp=np.abs(df.dWP.values)) \
            .nlargest(100, "abs_dwp")
    cols = ["year", "game_id", "inning", "is_home_batting", "outs",
            "score_diff_bat", "bases_idx", "wp_bat", "wp_post", "dWP",
            "LI", "batter_name", "pitcher_name", "event_raw", "event_simple"]
    # event_raw / event_simple are on plays, not df. Re-attach:
    plays_sorted = plays.copy()
    plays_sorted["half_ord"] = plays_sorted.half.map({"T": 0, "B": 1})
    plays_sorted = plays_sorted.sort_values(
        ["game_id", "inning", "half_ord", "outs_before"], kind="stable"
    ).reset_index(drop=True)
    df["event_raw"] = plays_sorted.event_raw.values
    df["event_simple"] = plays_sorted.event_simple.values
    top = df.assign(abs_dwp=np.abs(df.dWP.values)).nlargest(100, "abs_dwp")
    top[cols].to_csv("data/win_prob/top_plays_2015_2024.csv", index=False)
    for _, r in top.head(15).iterrows():
        side = "B" if r.is_home_batting else "T"
        print(f"  {r.year} {r.game_id} {side}{int(r.inning)} "
              f"{r.batter_name:>18} vs {r.pitcher_name:<18} "
              f"{r.event_simple:<5} dWP={r.dWP:+.3f} LI={r.LI:.2f}")

    # --- Season WPA leaderboards --------------------------------------
    print("\nBuilding season WPA leaderboards...")
    df["neg_dWP"] = -df.dWP.values
    df["high_lev"] = (df.LI.values >= 1.5).astype(np.int8)
    df["clutch_dWP"] = df.dWP.values * df.high_lev.values
    df["clutch_neg_dWP"] = df.neg_dWP.values * df.high_lev.values

    bat_season = df.dropna(subset=["batter_playerID"]).groupby(
        ["batter_playerID", "year"], sort=False).agg(
        name=("batter_name", "first"),
        PA=("dWP", "size"),
        WPA=("dWP", "sum"),
        clutch_WPA=("clutch_dWP", "sum"),
        mean_LI=("LI", "mean"),
    ).reset_index().rename(columns={"batter_playerID": "playerID"})
    bat_season.to_csv("data/win_prob/wpa_season_batters.csv", index=False)

    pit_season = df.dropna(subset=["pitcher_playerID"]).groupby(
        ["pitcher_playerID", "year"], sort=False).agg(
        name=("pitcher_name", "first"),
        BF=("neg_dWP", "size"),
        WPA=("neg_dWP", "sum"),
        clutch_WPA=("clutch_neg_dWP", "sum"),
        mean_LI=("LI", "mean"),
    ).reset_index().rename(columns={"pitcher_playerID": "playerID"})
    pit_season.to_csv("data/win_prob/wpa_season_pitchers.csv", index=False)

    # --- Career (2015-2024 sum) ---------------------------------------
    bat_career = bat_season.groupby(["playerID", "name"], sort=False).agg(
        years=("year", "nunique"),
        PA=("PA", "sum"),
        WPA=("WPA", "sum"),
        clutch_WPA=("clutch_WPA", "sum"),
    ).reset_index().sort_values("WPA", ascending=False)
    bat_career.to_csv("data/win_prob/wpa_career_batters.csv", index=False)

    pit_career = pit_season.groupby(["playerID", "name"], sort=False).agg(
        years=("year", "nunique"),
        BF=("BF", "sum"),
        WPA=("WPA", "sum"),
        clutch_WPA=("clutch_WPA", "sum"),
    ).reset_index().sort_values("WPA", ascending=False)
    pit_career.to_csv("data/win_prob/wpa_career_pitchers.csv", index=False)

    print("\nTop 10 batter career WPA (2015-2024):")
    for _, r in bat_career.head(10).iterrows():
        print(f"  {r['name']:<22} {r.PA:>6} PA  WPA {r.WPA:+.2f}  "
              f"clutch {r.clutch_WPA:+.2f}")

    print("\nTop 10 pitcher career WPA (2015-2024):")
    for _, r in pit_career.head(10).iterrows():
        print(f"  {r['name']:<22} {r.BF:>6} BF  WPA {r.WPA:+.2f}  "
              f"clutch {r.clutch_WPA:+.2f}")

    # --- Reliever leverage --------------------------------------------
    # Rough reliever filter: pitchers whose mean LI > 1.1 AND BF < 400 in a season
    print("\nTop 10 reliever seasons by high-leverage WPA:")
    rel = pit_season[(pit_season.mean_LI > 1.1) & (pit_season.BF < 400)
                     & (pit_season.BF > 80)]
    rel = rel.sort_values("clutch_WPA", ascending=False)
    rel.to_csv("data/win_prob/wpa_relievers.csv", index=False)
    for _, r in rel.head(10).iterrows():
        print(f"  {int(r.year)} {r['name']:<22} {int(r.BF):>4} BF  "
              f"LI {r.mean_LI:.2f}  clutch WPA {r.clutch_WPA:+.2f}")

    print("\n" + "=" * 72)
    print("  Saved: data/win_prob/{top_plays,wpa_season_*,wpa_career_*,"
          "wpa_relievers,leverage_index_states}.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
