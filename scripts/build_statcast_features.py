"""Compute per-player-season Statcast features and merge to BRAVS playerID.

Statcast pitch-by-pitch has MLBAM integer IDs; BRAVS uses Lahman string
playerIDs. We bridge via Chadwick register (MLBAM -> bbrefID) and treat
bbrefID as playerID (100% match for modern players in our Lahman People).

Per batter-season:
  pa, bip, exit_velo_avg, launch_angle_avg, hard_hit_pct, barrel_pct,
  xwoba, whiff_pct, k_pct, bb_pct

Per pitcher-season:
  bf, velo_avg, spin_avg, xwoba_allowed, k_pct, bb_pct, whiff_pct, hard_hit_allowed

Output: data/statcast/batter_season_features.csv
        data/statcast/pitcher_season_features.csv
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "statcast")
YEARS = list(range(2015, 2026))

BATTER_COLS = [
    "game_year", "batter", "player_name", "events", "description",
    "launch_speed", "launch_angle", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom", "zone", "type",
]

PITCHER_COLS = [
    "game_year", "pitcher", "events", "description",
    "release_speed", "release_spin_rate",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
    "launch_speed", "zone", "type",
]

WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked"}
SWING_DESC = WHIFF_DESC | {"foul", "foul_tip", "hit_into_play", "foul_bunt"}
CHASE_ZONES = {11, 12, 13, 14}  # zones outside strike zone


def load_all(cols: list[str]) -> pd.DataFrame:
    frames = []
    for y in YEARS:
        path = f"data/statcast/pbp/statcast_{y}.parquet"
        print(f"  {y}...", end=" ", flush=True)
        t0 = time.time()
        df = pd.read_parquet(path, columns=cols)
        frames.append(df)
        print(f"{len(df):,} ({time.time()-t0:.1f}s)")
    return pd.concat(frames, ignore_index=True)


def build_mlbam_to_lahman(mlbam_ids: list[int]) -> dict[int, str]:
    """Bridge MLBAM -> Lahman playerID via Chadwick register."""
    from pybaseball import playerid_reverse_lookup
    print(f"  Bridging {len(mlbam_ids):,} MLBAM ids via Chadwick...")
    df = playerid_reverse_lookup(mlbam_ids, key_type="mlbam")
    # bbref id matches Lahman playerID in ~100% of cases for modern players
    out = {}
    for _, r in df.iterrows():
        m = r.get("key_mlbam")
        b = r.get("key_bbref")
        if pd.notna(m) and isinstance(b, str) and b:
            out[int(m)] = b
    return out


def batter_features(df: pd.DataFrame) -> pd.DataFrame:
    """Season-level batter features from every pitch they faced."""
    df = df.copy()
    # Whiff / swing flags
    df["is_swing"] = df["description"].isin(SWING_DESC)
    df["is_whiff"] = df["description"].isin(WHIFF_DESC)
    df["is_bip"] = df["type"] == "X"  # ball in play
    df["is_chase"] = df["is_swing"] & df["zone"].isin(CHASE_ZONES)
    # Barrel: launch_speed >= 98 and launch_angle between ~26 and 30 (approx)
    df["is_barrel"] = (
        (df.launch_speed >= 98) & (df.launch_angle.between(26, 30))
    )
    df["is_hard_hit"] = df.launch_speed >= 95

    # Season-end PA from woba_denom > 0
    df["pa_flag"] = df["woba_denom"].fillna(0) > 0

    grp = df.groupby(["game_year", "batter"])
    out = grp.agg(
        pitches=("description", "size"),
        pa=("pa_flag", "sum"),
        bip=("is_bip", "sum"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        chases=("is_chase", "sum"),
        hard_hit=("is_hard_hit", "sum"),
        barrels=("is_barrel", "sum"),
        exit_velo_avg=("launch_speed", "mean"),
        launch_angle_avg=("launch_angle", "mean"),
        xwoba_num=("estimated_woba_using_speedangle", "sum"),
        woba_num=("woba_value", "sum"),
        woba_denom=("woba_denom", "sum"),
    ).reset_index()

    out["whiff_pct"] = out.whiffs / out.swings.replace(0, np.nan)
    out["chase_pct"] = out.chases / out.swings.replace(0, np.nan)
    out["hard_hit_pct"] = out.hard_hit / out.bip.replace(0, np.nan)
    out["barrel_pct"] = out.barrels / out.bip.replace(0, np.nan)
    out["xwoba"] = out.xwoba_num / out.woba_denom.replace(0, np.nan)
    out["woba"] = out.woba_num / out.woba_denom.replace(0, np.nan)
    return out


def pitcher_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_swing"] = df["description"].isin(SWING_DESC)
    df["is_whiff"] = df["description"].isin(WHIFF_DESC)
    df["is_hard_hit"] = df.launch_speed >= 95
    df["pa_flag"] = df["woba_denom"].fillna(0) > 0

    grp = df.groupby(["game_year", "pitcher"])
    out = grp.agg(
        pitches=("description", "size"),
        bf=("pa_flag", "sum"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        hard_hit_allowed=("is_hard_hit", "sum"),
        velo_avg=("release_speed", "mean"),
        spin_avg=("release_spin_rate", "mean"),
        xwoba_allowed_num=("estimated_woba_using_speedangle", "sum"),
        woba_allowed_num=("woba_value", "sum"),
        woba_denom=("woba_denom", "sum"),
    ).reset_index()

    out["whiff_pct"] = out.whiffs / out.swings.replace(0, np.nan)
    out["xwoba_allowed"] = out.xwoba_allowed_num / out.woba_denom.replace(0, np.nan)
    out["woba_allowed"] = out.woba_allowed_num / out.woba_denom.replace(0, np.nan)
    return out


def main():
    print("=" * 72)
    print("  STATCAST SEASON FEATURES")
    print("=" * 72)

    print("\n[Batters] Loading...")
    bat = load_all(BATTER_COLS)
    print(f"  Total: {len(bat):,} pitches")

    print("\n[Batters] Computing features...")
    bat_feats = batter_features(bat)
    print(f"  {len(bat_feats):,} batter-seasons")

    print("\n[Pitchers] Loading...")
    pit = load_all(PITCHER_COLS)
    print(f"  Total: {len(pit):,} pitches")

    print("\n[Pitchers] Computing features...")
    pit_feats = pitcher_features(pit)
    print(f"  {len(pit_feats):,} pitcher-seasons")

    # Bridge MLBAM -> Lahman
    all_mlbam = sorted(set(bat_feats.batter.unique()) | set(pit_feats.pitcher.unique()))
    mapping = build_mlbam_to_lahman(all_mlbam)
    print(f"  Bridged {len(mapping):,} / {len(all_mlbam):,} players")

    bat_feats["playerID"] = bat_feats.batter.map(mapping)
    pit_feats["playerID"] = pit_feats.pitcher.map(mapping)

    n_bat_matched = bat_feats.playerID.notna().sum()
    n_pit_matched = pit_feats.playerID.notna().sum()
    print(f"  Batter-seasons with playerID: {n_bat_matched:,} / {len(bat_feats):,}")
    print(f"  Pitcher-seasons with playerID: {n_pit_matched:,} / {len(pit_feats):,}")

    # Drop unmatched, keep only the features we care about
    bat_out = bat_feats.dropna(subset=["playerID"]).rename(columns={"game_year": "yearID"})[
        ["playerID", "yearID", "pa", "pitches", "exit_velo_avg", "launch_angle_avg",
         "whiff_pct", "chase_pct", "hard_hit_pct", "barrel_pct", "xwoba", "woba"]
    ]
    pit_out = pit_feats.dropna(subset=["playerID"]).rename(columns={"game_year": "yearID"})[
        ["playerID", "yearID", "bf", "pitches", "velo_avg", "spin_avg",
         "whiff_pct", "xwoba_allowed", "woba_allowed"]
    ]

    bat_path = os.path.join(OUT_DIR, "batter_season_features.csv")
    pit_path = os.path.join(OUT_DIR, "pitcher_season_features.csv")
    bat_out.to_csv(bat_path, index=False, encoding="utf-8-sig")
    pit_out.to_csv(pit_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved: {bat_path} ({len(bat_out):,} rows)")
    print(f"  Saved: {pit_path} ({len(pit_out):,} rows)")

    # Spot check
    print("\n  Top batters by xwOBA (min 300 PA, 2024):")
    top = bat_out[(bat_out.yearID == 2024) & (bat_out.pa >= 300)].nlargest(10, "xwoba")
    for _, r in top.iterrows():
        print(f"    {r.playerID:<12} xwOBA={r.xwoba:.3f}  exit_velo={r.exit_velo_avg:.1f}  barrel%={r.barrel_pct:.1%}")

    print("\n  Top pitchers by xwOBA allowed (min 500 BF, 2024):")
    top = pit_out[(pit_out.yearID == 2024) & (pit_out.bf >= 500)].nsmallest(10, "xwoba_allowed")
    for _, r in top.iterrows():
        print(f"    {r.playerID:<12} xwOBA_allowed={r.xwoba_allowed:.3f}  velo={r.velo_avg:.1f}  whiff%={r.whiff_pct:.1%}")

    print("=" * 72)


if __name__ == "__main__":
    main()
