"""Unified data access for all BRAVS apps.

Builds single-season tables for batters and pitchers with all stats merged
(traditional + statcast + BRAVS). Everything is lazy-loaded and lru-cached.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATCAST = os.path.join(ROOT, "data", "statcast")
LAHMAN = os.path.join(ROOT, "data", "lahman2025")


# ----------------------------------------------------------------------
# Primitive loaders
# ----------------------------------------------------------------------

def _safe_read(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


@lru_cache(maxsize=1)
def crosswalk() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(ROOT, "data", "id_crosswalk.csv"))
    df["mlbam_id"] = df["mlbam_id"].astype(int)
    return df


@lru_cache(maxsize=1)
def mlb_to_lahman() -> dict[int, str]:
    df = crosswalk()
    return dict(zip(df["mlbam_id"], df["lahman_id"].astype(str)))


@lru_cache(maxsize=1)
def lahman_to_mlb() -> dict[str, int]:
    df = crosswalk()
    return dict(zip(df["lahman_id"].astype(str), df["mlbam_id"].astype(int)))


@lru_cache(maxsize=1)
def people() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(LAHMAN, "People.csv"))
    df["full_name"] = (df["nameFirst"].fillna("") + " " + df["nameLast"].fillna("")).str.strip()
    return df[["playerID", "full_name", "birthYear"]]


@lru_cache(maxsize=1)
def teams() -> pd.DataFrame:
    t = pd.read_csv(os.path.join(LAHMAN, "Teams.csv"))
    return t[["yearID", "teamID", "name", "lgID"]]


# ----------------------------------------------------------------------
# Raw season tables
# ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def batting_raw() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(LAHMAN, "Batting.csv"))
    df = df.groupby(["playerID", "yearID"], as_index=False).agg({
        "G": "sum", "AB": "sum", "R": "sum", "H": "sum", "X2B": "sum", "X3B": "sum",
        "HR": "sum", "RBI": "sum", "SB": "sum", "CS": "sum",
        "BB": "sum", "SO": "sum", "IBB": "sum", "HBP": "sum", "SH": "sum", "SF": "sum",
    })
    df["PA"] = df["AB"] + df["BB"].fillna(0) + df["HBP"].fillna(0) + df["SF"].fillna(0) + df["SH"].fillna(0)
    df["1B"] = df["H"] - df["X2B"] - df["X3B"] - df["HR"]
    df["TB"] = df["1B"] + 2*df["X2B"] + 3*df["X3B"] + 4*df["HR"]
    ab = df["AB"].replace(0, np.nan)
    df["AVG"] = df["H"] / ab
    df["OBP"] = (df["H"] + df["BB"] + df["HBP"]) / (df["AB"] + df["BB"] + df["HBP"] + df["SF"]).replace(0, np.nan)
    df["SLG"] = df["TB"] / ab
    df["OPS"] = df["OBP"] + df["SLG"]
    df["ISO"] = df["SLG"] - df["AVG"]
    pa = df["PA"].replace(0, np.nan)
    df["K_pct"] = df["SO"] / pa * 100
    df["BB_pct"] = df["BB"] / pa * 100
    return df


@lru_cache(maxsize=1)
def pitching_raw() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(LAHMAN, "Pitching.csv"))
    df = df.groupby(["playerID", "yearID"], as_index=False).agg({
        "W": "sum", "L": "sum", "G": "sum", "GS": "sum", "CG": "sum", "SHO": "sum",
        "SV": "sum", "IPouts": "sum", "H": "sum", "ER": "sum", "HR": "sum",
        "BB": "sum", "SO": "sum", "HBP": "sum", "BFP": "sum", "GF": "sum", "R": "sum",
    })
    df["IP"] = df["IPouts"] / 3.0
    ip = df["IP"].replace(0, np.nan)
    df["ERA"] = df["ER"] * 9 / ip
    df["WHIP"] = (df["BB"] + df["H"]) / ip
    df["K9"] = df["SO"] * 9 / ip
    df["BB9"] = df["BB"] * 9 / ip
    df["HR9"] = df["HR"] * 9 / ip
    bfp = df["BFP"].replace(0, np.nan)
    df["K_pct"] = df["SO"] / bfp * 100
    df["BB_pct"] = df["BB"] / bfp * 100
    df["K_BB_pct"] = df["K_pct"] - df["BB_pct"]
    # Simple FIP: ((13*HR + 3*(BB+HBP) - 2*SO) / IP) + constant
    df["FIP"] = (13*df["HR"] + 3*(df["BB"] + df["HBP"].fillna(0)) - 2*df["SO"]) / ip + 3.10
    return df


@lru_cache(maxsize=1)
def bravs() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ROOT, "data", "bravs_all_seasons.csv"))


@lru_cache(maxsize=1)
def batter_team_year() -> pd.DataFrame:
    """Primary team per (playerID, yearID)."""
    df = pd.read_csv(os.path.join(LAHMAN, "Batting.csv"))
    # Pick the stint with most PA as primary
    df["pa_stint"] = df["AB"].fillna(0) + df["BB"].fillna(0)
    idx = df.groupby(["playerID", "yearID"])["pa_stint"].idxmax()
    return df.loc[idx, ["playerID", "yearID", "teamID", "lgID"]]


@lru_cache(maxsize=1)
def pitcher_team_year() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(LAHMAN, "Pitching.csv"))
    df["_ip"] = df["IPouts"].fillna(0)
    idx = df.groupby(["playerID", "yearID"])["_ip"].idxmax()
    return df.loc[idx, ["playerID", "yearID", "teamID", "lgID"]]


# ----------------------------------------------------------------------
# Statcast tables
# ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def exit_velocity() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "exit_velocity_all.csv"))


@lru_cache(maxsize=1)
def expected_batting() -> pd.DataFrame:
    frames = []
    for y in range(2015, 2026):
        p = os.path.join(STATCAST, f"expected_batting_{y}.csv")
        if os.path.exists(p):
            df = _safe_read(p)
            if "year" not in df.columns:
                df["year"] = y
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@lru_cache(maxsize=1)
def expected_pitching() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "expected_pitching_all.csv"))


@lru_cache(maxsize=1)
def sprint_speed() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "sprint_speed_all.csv"))


@lru_cache(maxsize=1)
def oaa_table() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "outs_above_average.csv"))


@lru_cache(maxsize=1)
def arm_strength() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "arm_strength_all.csv"))


@lru_cache(maxsize=1)
def batter_features() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "batter_season_features.csv"))


@lru_cache(maxsize=1)
def pitcher_features() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "pitcher_season_features.csv"))


@lru_cache(maxsize=1)
def savant_pct_batter() -> pd.DataFrame:
    df = _safe_read(os.path.join(STATCAST, "percentile_rankings_batter_all.csv"))
    if not df.empty:
        df["player_id"] = df["player_id"].astype(int)
    return df


@lru_cache(maxsize=1)
def savant_pct_pitcher() -> pd.DataFrame:
    df = _safe_read(os.path.join(STATCAST, "percentile_rankings_pitcher_all.csv"))
    if not df.empty:
        df["player_id"] = df["player_id"].astype(int)
    return df


@lru_cache(maxsize=1)
def pitch_arsenal() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "pitch_arsenal_stats.csv"))


# ----------------------------------------------------------------------
# Merged master tables
# ----------------------------------------------------------------------

def _attach_mlbam(df: pd.DataFrame) -> pd.DataFrame:
    cw = crosswalk()
    out = df.merge(cw, left_on="playerID", right_on="lahman_id", how="left")
    return out


@lru_cache(maxsize=1)
def batter_master() -> pd.DataFrame:
    """One row per (playerID, yearID) batter, with all joinable stats."""
    bat = batting_raw().copy()
    nm = people()
    bat = bat.merge(nm, on="playerID", how="left")
    team = batter_team_year()
    bat = bat.merge(team, on=["playerID", "yearID"], how="left")
    bat = _attach_mlbam(bat)

    # BRAVS
    brv = bravs()[["playerID", "yearID", "bravs", "bravs_era_std", "bravs_war_eq",
                   "hitting_runs", "baserunning_runs", "fielding_runs", "positional_runs",
                   "durability_runs", "leverage_runs", "position"]]
    bat = bat.merge(brv, on=["playerID", "yearID"], how="left")

    # Statcast
    ev = exit_velocity()
    if not ev.empty:
        ev_s = ev[["player_id", "year", "avg_hit_speed", "max_hit_speed",
                   "ev95percent", "brl_percent", "anglesweetspotpercent", "attempts"]].rename(
            columns={"player_id": "mlbam_id", "year": "yearID",
                     "avg_hit_speed": "ev_avg", "max_hit_speed": "ev_max",
                     "ev95percent": "hardhit_pct", "brl_percent": "barrel_pct",
                     "anglesweetspotpercent": "sweetspot_pct"})
        bat = bat.merge(ev_s, on=["mlbam_id", "yearID"], how="left")

    xb = expected_batting()
    if not xb.empty:
        xb_s = xb[["player_id", "year", "est_ba", "est_slg", "est_woba", "woba", "pa"]].rename(
            columns={"player_id": "mlbam_id", "year": "yearID", "pa": "pa_sc",
                     "est_ba": "xBA", "est_slg": "xSLG", "est_woba": "xwOBA", "woba": "wOBA"})
        bat = bat.merge(xb_s, on=["mlbam_id", "yearID"], how="left")

    bf = batter_features()
    if not bf.empty:
        bf_s = bf[["playerID", "yearID", "chase_pct", "whiff_pct"]].rename(
            columns={"chase_pct": "chase_rate", "whiff_pct": "whiff_rate"})
        bat = bat.merge(bf_s, on=["playerID", "yearID"], how="left")
        bat["chase_rate"] = bat["chase_rate"] * 100
        bat["whiff_rate"] = bat["whiff_rate"] * 100

    sp = sprint_speed()
    if not sp.empty:
        sp_s = sp[["player_id", "year", "sprint_speed"]].rename(
            columns={"player_id": "mlbam_id", "year": "yearID"})
        bat = bat.merge(sp_s, on=["mlbam_id", "yearID"], how="left")

    oa = oaa_table()
    if not oa.empty:
        oa_s = oa[["player_id", "year", "outs_above_average", "fielding_runs_prevented"]].rename(
            columns={"player_id": "mlbam_id", "year": "yearID", "outs_above_average": "OAA",
                     "fielding_runs_prevented": "field_rv"})
        bat = bat.merge(oa_s, on=["mlbam_id", "yearID"], how="left")

    return bat


@lru_cache(maxsize=1)
def pitcher_master() -> pd.DataFrame:
    pit = pitching_raw().copy()
    nm = people()
    pit = pit.merge(nm, on="playerID", how="left")
    team = pitcher_team_year()
    pit = pit.merge(team, on=["playerID", "yearID"], how="left")
    pit = _attach_mlbam(pit)

    brv = bravs()[["playerID", "yearID", "bravs", "bravs_era_std", "bravs_war_eq",
                   "pitching_runs", "leverage_runs"]]
    pit = pit.merge(brv, on=["playerID", "yearID"], how="left")

    xp = expected_pitching()
    if not xp.empty:
        xp_s = xp[["player_id", "year", "est_ba", "est_slg", "est_woba", "xera", "woba"]].rename(
            columns={"player_id": "mlbam_id", "year": "yearID",
                     "est_ba": "xBA", "est_slg": "xSLG", "est_woba": "xwOBA",
                     "xera": "xERA", "woba": "wOBA_allowed"})
        pit = pit.merge(xp_s, on=["mlbam_id", "yearID"], how="left")

    pf = pitcher_features()
    if not pf.empty:
        pf_s = pf[["playerID", "yearID", "velo_avg", "spin_avg", "whiff_pct"]].rename(
            columns={"velo_avg": "velo", "spin_avg": "spin", "whiff_pct": "whiff_rate"})
        pit = pit.merge(pf_s, on=["playerID", "yearID"], how="left")
        pit["whiff_rate"] = pit["whiff_rate"].astype(float) * 100

    return pit


# ----------------------------------------------------------------------
# Stat catalog (what can a user filter / rank by?)
# ----------------------------------------------------------------------

@dataclass
class Stat:
    key: str          # column in master table
    label: str        # display name
    kind: Literal["batter", "pitcher"]
    direction: Literal["high", "low"]  # higher = better (or lower)
    fmt: str = "{:.3f}"
    group: str = "Standard"


BATTER_STATS: list[Stat] = [
    # Counting
    Stat("G", "G", "batter", "high", "{:.0f}", "Standard"),
    Stat("PA", "PA", "batter", "high", "{:.0f}", "Standard"),
    Stat("AB", "AB", "batter", "high", "{:.0f}", "Standard"),
    Stat("H", "H", "batter", "high", "{:.0f}", "Standard"),
    Stat("HR", "HR", "batter", "high", "{:.0f}", "Standard"),
    Stat("RBI", "RBI", "batter", "high", "{:.0f}", "Standard"),
    Stat("R", "R", "batter", "high", "{:.0f}", "Standard"),
    Stat("SB", "SB", "batter", "high", "{:.0f}", "Standard"),
    Stat("BB", "BB", "batter", "high", "{:.0f}", "Standard"),
    Stat("SO", "SO", "batter", "low", "{:.0f}", "Standard"),
    # Rate
    Stat("AVG", "AVG", "batter", "high", "{:.3f}", "Rate"),
    Stat("OBP", "OBP", "batter", "high", "{:.3f}", "Rate"),
    Stat("SLG", "SLG", "batter", "high", "{:.3f}", "Rate"),
    Stat("OPS", "OPS", "batter", "high", "{:.3f}", "Rate"),
    Stat("ISO", "ISO", "batter", "high", "{:.3f}", "Rate"),
    Stat("K_pct", "K%", "batter", "low", "{:.1f}", "Rate"),
    Stat("BB_pct", "BB%", "batter", "high", "{:.1f}", "Rate"),
    # Statcast
    Stat("xBA", "xBA", "batter", "high", "{:.3f}", "Statcast"),
    Stat("xSLG", "xSLG", "batter", "high", "{:.3f}", "Statcast"),
    Stat("xwOBA", "xwOBA", "batter", "high", "{:.3f}", "Statcast"),
    Stat("wOBA", "wOBA", "batter", "high", "{:.3f}", "Statcast"),
    Stat("ev_avg", "Avg EV", "batter", "high", "{:.1f}", "Statcast"),
    Stat("ev_max", "Max EV", "batter", "high", "{:.1f}", "Statcast"),
    Stat("hardhit_pct", "Hard-Hit%", "batter", "high", "{:.1f}", "Statcast"),
    Stat("barrel_pct", "Barrel%", "batter", "high", "{:.1f}", "Statcast"),
    Stat("sweetspot_pct", "Sweet-Spot%", "batter", "high", "{:.1f}", "Statcast"),
    Stat("chase_rate", "Chase%", "batter", "low", "{:.1f}", "Statcast"),
    Stat("whiff_rate", "Whiff%", "batter", "low", "{:.1f}", "Statcast"),
    Stat("sprint_speed", "Sprint Speed", "batter", "high", "{:.1f}", "Statcast"),
    Stat("OAA", "OAA", "batter", "high", "{:.0f}", "Statcast"),
    # BRAVS
    Stat("bravs", "BRAVS", "batter", "high", "{:.2f}", "BRAVS"),
    Stat("bravs_war_eq", "WAR-eq", "batter", "high", "{:.2f}", "BRAVS"),
    Stat("hitting_runs", "Hit Runs", "batter", "high", "{:.1f}", "BRAVS"),
    Stat("baserunning_runs", "BsR Runs", "batter", "high", "{:.1f}", "BRAVS"),
    Stat("fielding_runs", "Fld Runs", "batter", "high", "{:.1f}", "BRAVS"),
]


PITCHER_STATS: list[Stat] = [
    Stat("G", "G", "pitcher", "high", "{:.0f}", "Standard"),
    Stat("GS", "GS", "pitcher", "high", "{:.0f}", "Standard"),
    Stat("IP", "IP", "pitcher", "high", "{:.1f}", "Standard"),
    Stat("W", "W", "pitcher", "high", "{:.0f}", "Standard"),
    Stat("L", "L", "pitcher", "low", "{:.0f}", "Standard"),
    Stat("SV", "SV", "pitcher", "high", "{:.0f}", "Standard"),
    Stat("SO", "K", "pitcher", "high", "{:.0f}", "Standard"),
    Stat("BB", "BB", "pitcher", "low", "{:.0f}", "Standard"),
    Stat("H", "H", "pitcher", "low", "{:.0f}", "Standard"),
    Stat("HR", "HR", "pitcher", "low", "{:.0f}", "Standard"),
    # Rate
    Stat("ERA", "ERA", "pitcher", "low", "{:.2f}", "Rate"),
    Stat("WHIP", "WHIP", "pitcher", "low", "{:.2f}", "Rate"),
    Stat("FIP", "FIP", "pitcher", "low", "{:.2f}", "Rate"),
    Stat("K9", "K/9", "pitcher", "high", "{:.1f}", "Rate"),
    Stat("BB9", "BB/9", "pitcher", "low", "{:.1f}", "Rate"),
    Stat("HR9", "HR/9", "pitcher", "low", "{:.1f}", "Rate"),
    Stat("K_pct", "K%", "pitcher", "high", "{:.1f}", "Rate"),
    Stat("BB_pct", "BB%", "pitcher", "low", "{:.1f}", "Rate"),
    Stat("K_BB_pct", "K-BB%", "pitcher", "high", "{:.1f}", "Rate"),
    # Statcast
    Stat("xERA", "xERA", "pitcher", "low", "{:.2f}", "Statcast"),
    Stat("xBA", "xBA", "pitcher", "low", "{:.3f}", "Statcast"),
    Stat("xSLG", "xSLG", "pitcher", "low", "{:.3f}", "Statcast"),
    Stat("xwOBA", "xwOBA", "pitcher", "low", "{:.3f}", "Statcast"),
    Stat("velo", "Velo", "pitcher", "high", "{:.1f}", "Statcast"),
    Stat("spin", "Spin", "pitcher", "high", "{:.0f}", "Statcast"),
    Stat("whiff_rate", "Whiff%", "pitcher", "high", "{:.1f}", "Statcast"),
    # BRAVS
    Stat("bravs", "BRAVS", "pitcher", "high", "{:.2f}", "BRAVS"),
    Stat("bravs_war_eq", "WAR-eq", "pitcher", "high", "{:.2f}", "BRAVS"),
    Stat("pitching_runs", "Pitch Runs", "pitcher", "high", "{:.1f}", "BRAVS"),
]


def stats_for(kind: str) -> list[Stat]:
    return BATTER_STATS if kind == "batter" else PITCHER_STATS


def stat_by_key(kind: str, key: str) -> Stat | None:
    for s in stats_for(kind):
        if s.key == key:
            return s
    return None


# ----------------------------------------------------------------------
# Convenience helpers
# ----------------------------------------------------------------------

def master(kind: str) -> pd.DataFrame:
    return batter_master() if kind == "batter" else pitcher_master()


def qualified(df: pd.DataFrame, kind: str, year: int | None = None,
              min_pa: int = 300, min_ip: float = 40) -> pd.DataFrame:
    out = df
    if year is not None:
        out = out[out["yearID"] == year]
    if kind == "batter":
        out = out[out["PA"] >= min_pa]
    else:
        out = out[out["IP"] >= min_ip]
    return out


def fmt_value(stat: Stat, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        return stat.fmt.format(float(v))
    except Exception:
        return str(v)


def format_team(row) -> str:
    t = row.get("teamID")
    return str(t) if isinstance(t, str) else "—"


def portrait_url(mlbam_id) -> str | None:
    """MLB's headshot CDN. Returns None for missing/invalid ids."""
    try:
        mid = int(mlbam_id)
    except (TypeError, ValueError):
        return None
    return (
        "https://img.mlbstatic.com/mlb-photos/image/upload/"
        "d_people:generic:headshot:67:current.png/w_213,q_auto:best/"
        f"v1/people/{mid}/headshot/67/current"
    )


# Official MLB team colors (primary, secondary) keyed by Lahman/Retrosheet code
TEAM_COLORS: dict[str, dict[str, str]] = {
    "BAL": {"primary": "#DF4601", "secondary": "#000000", "name": "Baltimore Orioles"},
    "BOS": {"primary": "#BD3039", "secondary": "#0C2340", "name": "Boston Red Sox"},
    "NYA": {"primary": "#132448", "secondary": "#C4CED3", "name": "New York Yankees"},
    "TBA": {"primary": "#092C5C", "secondary": "#8FBCE6", "name": "Tampa Bay Rays"},
    "TOR": {"primary": "#134A8E", "secondary": "#E8291C", "name": "Toronto Blue Jays"},
    "CHA": {"primary": "#27251F", "secondary": "#C4CED4", "name": "Chicago White Sox"},
    "CLE": {"primary": "#0C2340", "secondary": "#E31937", "name": "Cleveland Guardians"},
    "DET": {"primary": "#0C2340", "secondary": "#FA4616", "name": "Detroit Tigers"},
    "KCA": {"primary": "#004687", "secondary": "#BD9B60", "name": "Kansas City Royals"},
    "MIN": {"primary": "#002B5C", "secondary": "#D31145", "name": "Minnesota Twins"},
    "HOU": {"primary": "#002D62", "secondary": "#EB6E1F", "name": "Houston Astros"},
    "LAA": {"primary": "#BA0021", "secondary": "#003263", "name": "Los Angeles Angels"},
    "OAK": {"primary": "#003831", "secondary": "#EFB21E", "name": "Oakland Athletics"},
    "ATH": {"primary": "#003831", "secondary": "#EFB21E", "name": "Athletics"},
    "SEA": {"primary": "#0C2C56", "secondary": "#005C5C", "name": "Seattle Mariners"},
    "TEX": {"primary": "#003278", "secondary": "#C0111F", "name": "Texas Rangers"},
    "ATL": {"primary": "#CE1141", "secondary": "#13274F", "name": "Atlanta Braves"},
    "MIA": {"primary": "#00A3E0", "secondary": "#EF3340", "name": "Miami Marlins"},
    "FLO": {"primary": "#00A3E0", "secondary": "#EF3340", "name": "Miami Marlins"},
    "NYN": {"primary": "#002D72", "secondary": "#FF5910", "name": "New York Mets"},
    "PHI": {"primary": "#E81828", "secondary": "#002D72", "name": "Philadelphia Phillies"},
    "WAS": {"primary": "#AB0003", "secondary": "#14225A", "name": "Washington Nationals"},
    "CHN": {"primary": "#0E3386", "secondary": "#CC3433", "name": "Chicago Cubs"},
    "CIN": {"primary": "#C6011F", "secondary": "#000000", "name": "Cincinnati Reds"},
    "MIL": {"primary": "#12284B", "secondary": "#FFC52F", "name": "Milwaukee Brewers"},
    "PIT": {"primary": "#FDB827", "secondary": "#27251F", "name": "Pittsburgh Pirates"},
    "SLN": {"primary": "#C41E3A", "secondary": "#0C2340", "name": "St. Louis Cardinals"},
    "ARI": {"primary": "#A71930", "secondary": "#E3D4AD", "name": "Arizona Diamondbacks"},
    "COL": {"primary": "#33006F", "secondary": "#C4CED4", "name": "Colorado Rockies"},
    "LAN": {"primary": "#005A9C", "secondary": "#EF3E42", "name": "Los Angeles Dodgers"},
    "SDN": {"primary": "#2F241D", "secondary": "#FFC425", "name": "San Diego Padres"},
    "SFN": {"primary": "#FD5A1E", "secondary": "#27251F", "name": "San Francisco Giants"},
}


def team_colors(team_id: str | None) -> dict[str, str]:
    if not team_id:
        return {"primary": "#18181b", "secondary": "#71717a", "name": "—"}
    return TEAM_COLORS.get(team_id, {"primary": "#18181b", "secondary": "#71717a", "name": team_id})


def player_age(playerID: str, year: int) -> int | None:
    p = people()
    sub = p[p["playerID"] == playerID]
    if sub.empty:
        return None
    by = sub.iloc[0]["birthYear"]
    if pd.isna(by):
        return None
    return int(year - int(by))


# ----------------------------------------------------------------------
# Career awards / honors
# ----------------------------------------------------------------------

AWARD_COUNTS = {
    "All-Star": "allstar",
    "Gold Glove": "gold_glove",
    "Silver Slugger": "silver_slugger",
    "MVP": "mvp",
    "Cy Young Award": "cy_young",
    "Rookie of the Year": "roy",
    "World Series MVP": "ws_mvp",
    "TSN Player of the Year": "tsn_player",
    "Hank Aaron Award": "hank_aaron",
    "Comeback Player of the Year": "comeback",
}


@lru_cache(maxsize=1)
def _awards() -> pd.DataFrame:
    return pd.read_csv(os.path.join(LAHMAN, "AwardsPlayers.csv"))


@lru_cache(maxsize=1)
def _allstar() -> pd.DataFrame:
    return pd.read_csv(os.path.join(LAHMAN, "AllstarFull.csv"))


def career_awards(playerID: str) -> dict[str, int]:
    """Count honors for a player across their career."""
    out: dict[str, int] = {}
    aw = _awards()
    sub = aw[aw["playerID"] == playerID]
    for label, _key in AWARD_COUNTS.items():
        if label == "All-Star":
            continue  # handled separately
        out[label] = int((sub["awardID"] == label).sum())
    # All-Star Game selections: from AllstarFull
    asf = _allstar()
    out["All-Star"] = int((asf["playerID"] == playerID).sum())
    # Keep only non-zero counts for a cleaner display
    return {k: v for k, v in out.items() if v > 0}


def career_totals_batter(playerID: str) -> dict:
    df = batter_master()
    sub = df[df["playerID"] == playerID]
    if sub.empty:
        return {}
    ab = sub["AB"].sum()
    pa = sub["PA"].sum()
    h = sub["H"].sum()
    tb = sub["TB"].sum() if "TB" in sub else (sub["1B"] + 2*sub["X2B"] + 3*sub["X3B"] + 4*sub["HR"]).sum()
    bb = sub["BB"].sum()
    hbp = sub["HBP"].fillna(0).sum()
    sf = sub["SF"].fillna(0).sum()
    obp_den = ab + bb + hbp + sf
    avg = h / ab if ab else None
    obp = (h + bb + hbp) / obp_den if obp_den else None
    slg = tb / ab if ab else None
    return {
        "G": int(sub["G"].sum()),
        "PA": int(pa),
        "AB": int(ab),
        "H": int(h),
        "HR": int(sub["HR"].sum()),
        "RBI": int(sub["RBI"].sum()) if "RBI" in sub else 0,
        "R": int(sub["R"].sum()),
        "SB": int(sub["SB"].sum()),
        "BB": int(bb),
        "SO": int(sub["SO"].sum()),
        "AVG": avg,
        "OBP": obp,
        "SLG": slg,
        "OPS": (obp or 0) + (slg or 0) if obp and slg else None,
        "bravs_war_eq": float(sub["bravs_war_eq"].sum()) if "bravs_war_eq" in sub else None,
        "bravs": float(sub["bravs"].sum()) if "bravs" in sub else None,
        "first_year": int(sub["yearID"].min()),
        "last_year": int(sub["yearID"].max()),
    }


def career_totals_pitcher(playerID: str) -> dict:
    df = pitcher_master()
    sub = df[df["playerID"] == playerID]
    if sub.empty:
        return {}
    ip = sub["IP"].sum()
    er = sub["ER"].sum()
    bb = sub["BB"].sum()
    h = sub["H"].sum()
    so = sub["SO"].sum()
    era = er * 9 / ip if ip else None
    whip = (bb + h) / ip if ip else None
    k9 = so * 9 / ip if ip else None
    return {
        "G": int(sub["G"].sum()),
        "GS": int(sub["GS"].sum()),
        "IP": float(ip),
        "W": int(sub["W"].sum()),
        "L": int(sub["L"].sum()),
        "SV": int(sub["SV"].sum()),
        "SO": int(so),
        "BB": int(bb),
        "H": int(h),
        "HR": int(sub["HR"].sum()),
        "ERA": era,
        "WHIP": whip,
        "K9": k9,
        "bravs_war_eq": float(sub["bravs_war_eq"].sum()) if "bravs_war_eq" in sub else None,
        "bravs": float(sub["bravs"].sum()) if "bravs" in sub else None,
        "first_year": int(sub["yearID"].min()),
        "last_year": int(sub["yearID"].max()),
    }
