"""Compute Savant-style percentile rankings from raw data.

Loads raw Statcast leaderboards and aggregate CSVs on startup, then for any
(mlbam_id, year) pair returns the full percentile card.

Percentile = player's rank within the season's qualifying pool, 0-100.
Computed ourselves — Savant's precomputed percentiles are only used as
fallback for metrics we cannot derive (e.g. some raw stat is missing the
player's row in our CSVs).
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

# Direction: "high" means higher value = better (goes red on Savant)
# "low" means lower = better (e.g. chase%, whiff%, K% for pitchers is good-low-for-hitters)
Direction = Literal["high", "low"]


@dataclass
class MetricResult:
    key: str
    label: str
    value: float | None
    percentile: int | None
    direction: Direction
    fmt: str = "{:.1f}"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "value": None if self.value is None or pd.isna(self.value) else float(self.value),
            "value_fmt": (self.fmt.format(self.value)
                          if self.value is not None and not pd.isna(self.value) else "—"),
            "percentile": self.percentile,
            "direction": self.direction,
        }


# -----------------------------------------------------------------------------
# Data loaders (lazy, cached)
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _crosswalk() -> dict[int, str]:
    df = pd.read_csv(os.path.join(ROOT, "data", "id_crosswalk.csv"))
    return dict(zip(df["mlbam_id"].astype(int), df["lahman_id"].astype(str)))


@lru_cache(maxsize=1)
def _people() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(LAHMAN, "People.csv"))
    df["full_name"] = df["nameFirst"].fillna("") + " " + df["nameLast"].fillna("")
    return df[["playerID", "full_name", "nameFirst", "nameLast"]]


@lru_cache(maxsize=1)
def _batting() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(LAHMAN, "Batting.csv"))
    df = df.groupby(["playerID", "yearID"], as_index=False).agg({
        "G": "sum", "AB": "sum", "H": "sum", "BB": "sum", "SO": "sum",
        "HBP": "sum", "SF": "sum", "X2B": "sum", "X3B": "sum", "HR": "sum",
        "SB": "sum", "CS": "sum", "R": "sum", "RBI": "sum",
    })
    df["PA"] = df["AB"] + df["BB"].fillna(0) + df["HBP"].fillna(0) + df["SF"].fillna(0)
    df["K_pct"] = df["SO"] / df["PA"].replace(0, np.nan) * 100
    df["BB_pct"] = df["BB"] / df["PA"].replace(0, np.nan) * 100
    return df


@lru_cache(maxsize=1)
def _pitching() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(LAHMAN, "Pitching.csv"))
    df = df.groupby(["playerID", "yearID"], as_index=False).agg({
        "G": "sum", "GS": "sum", "IPouts": "sum", "H": "sum", "ER": "sum",
        "BB": "sum", "SO": "sum", "HR": "sum", "BFP": "sum", "HBP": "sum",
        "SV": "sum", "W": "sum", "L": "sum",
    })
    df["IP"] = df["IPouts"] / 3.0
    df["ERA"] = df["ER"] * 9 / df["IP"].replace(0, np.nan)
    df["K_pct"] = df["SO"] / df["BFP"].replace(0, np.nan) * 100
    df["BB_pct"] = df["BB"] / df["BFP"].replace(0, np.nan) * 100
    df["WHIP"] = (df["BB"] + df["H"]) / df["IP"].replace(0, np.nan)
    return df


def _safe_read(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


@lru_cache(maxsize=1)
def _exit_velocity() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "exit_velocity_all.csv"))


@lru_cache(maxsize=1)
def _expected_batting() -> pd.DataFrame:
    # Concat all expected_batting_YYYY files
    frames = []
    for f in sorted(os.listdir(STATCAST)):
        if f.startswith("expected_batting_") and f.endswith(".csv") and "all" not in f:
            frames.append(_safe_read(os.path.join(STATCAST, f)))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@lru_cache(maxsize=1)
def _expected_pitching() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "expected_pitching_all.csv"))


@lru_cache(maxsize=1)
def _bat_tracking() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "bat_tracking_all.csv"))


@lru_cache(maxsize=1)
def _sprint_speed() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "sprint_speed_all.csv"))


@lru_cache(maxsize=1)
def _arm_strength() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "arm_strength_all.csv"))


@lru_cache(maxsize=1)
def _oaa() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "outs_above_average.csv"))


@lru_cache(maxsize=1)
def _framing() -> pd.DataFrame:
    frames = []
    for y in range(2015, 2026):
        p = os.path.join(STATCAST, f"catcher_framing_{y}.csv")
        if os.path.exists(p):
            df = _safe_read(p)
            if "year" not in df.columns:
                df["year"] = y
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@lru_cache(maxsize=1)
def _poptime() -> pd.DataFrame:
    frames = []
    for y in range(2015, 2026):
        p = os.path.join(STATCAST, f"poptime_{y}.csv")
        if os.path.exists(p):
            df = _safe_read(p)
            if "year" not in df.columns:
                df["year"] = y
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@lru_cache(maxsize=1)
def _catcher_blocking() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "catcher_blocking_all.csv"))


@lru_cache(maxsize=1)
def _catcher_throwing() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "catcher_throwing_all.csv"))


@lru_cache(maxsize=1)
def _pitcher_season_features() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "pitcher_season_features.csv"))


@lru_cache(maxsize=1)
def _batter_season_features() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "batter_season_features.csv"))


@lru_cache(maxsize=1)
def _bravs() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ROOT, "data", "bravs_all_seasons.csv"))


@lru_cache(maxsize=1)
def _savant_pct_batter() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "percentile_rankings_batter_all.csv"))


@lru_cache(maxsize=1)
def _savant_pct_pitcher() -> pd.DataFrame:
    return _safe_read(os.path.join(STATCAST, "percentile_rankings_pitcher_all.csv"))


# -----------------------------------------------------------------------------
# Percentile helpers
# -----------------------------------------------------------------------------

def _pct_rank(series: pd.Series, value: float, direction: Direction) -> int | None:
    """Return percentile (0-100) of value within series, handling direction.

    'high' = bigger is better, 'low' = smaller is better.
    """
    s = series.dropna()
    if len(s) < 10 or value is None or pd.isna(value):
        return None
    if direction == "high":
        pct = (s < value).mean() * 100
    else:
        pct = (s > value).mean() * 100
    return int(round(pct))


def _find_player_row(df: pd.DataFrame, mlbam_id: int, year: int,
                     id_col: str = "player_id", year_col: str = "year") -> pd.Series | None:
    if df.empty or id_col not in df.columns:
        return None
    sub = df[(df[id_col] == mlbam_id) & (df[year_col] == year)]
    if sub.empty:
        return None
    return sub.iloc[0]


def _savant_pct(mlbam_id: int, year: int, kind: str, field: str) -> int | None:
    """Look up Savant's precomputed percentile (0-100) for a given metric.

    Used as a fallback for metrics where we can't compute our own (e.g.
    bat_speed for years pre-2025, because Savant's year-filtered endpoint
    is broken).
    """
    df = _savant_pct_batter() if kind == "batter" else _savant_pct_pitcher()
    if df.empty or field not in df.columns:
        return None
    sub = df[(df["player_id"] == mlbam_id) & (df["year"] == year)]
    if sub.empty:
        return None
    v = sub.iloc[0][field]
    if pd.isna(v):
        return None
    return int(round(float(v)))


# -----------------------------------------------------------------------------
# Build percentile card for a player
# -----------------------------------------------------------------------------

def classify_player(mlbam_id: int, year: int) -> Literal["batter", "pitcher", "unknown"]:
    """Decide if we render the batter or pitcher card.

    Heuristic: compare PA vs IP using Lahman aggregates via crosswalk.
    """
    cw = _crosswalk()
    lahman_id = cw.get(mlbam_id)
    if not lahman_id:
        return "unknown"

    bat = _batting()
    pit = _pitching()

    pa = 0
    bsub = bat[(bat["playerID"] == lahman_id) & (bat["yearID"] == year)]
    if not bsub.empty:
        pa = int(bsub.iloc[0]["PA"])

    ip = 0.0
    psub = pit[(pit["playerID"] == lahman_id) & (pit["yearID"] == year)]
    if not psub.empty:
        ip = float(psub.iloc[0]["IP"])

    if pa == 0 and ip == 0:
        return "unknown"
    # Two-way (Ohtani): prefer whichever has more volume relative to qualifier
    if ip > 20 and pa < 200:
        return "pitcher"
    return "pitcher" if ip > 20 and pa < 300 else "batter"


def batter_card(mlbam_id: int, year: int) -> dict:
    cw = _crosswalk()
    lahman_id = cw.get(mlbam_id)

    # Pool: all batters that year with >=300 PA (Savant-style qualifying)
    results: list[MetricResult] = []

    # --- Expected stats ---
    xb = _expected_batting()
    if not xb.empty:
        pool = xb[(xb["year"] == year) & (xb["pa"] >= 300)]
        row = _find_player_row(xb, mlbam_id, year)
        if row is not None:
            for key, label, fmt in [
                ("est_woba", "xwOBA", "{:.3f}"),
                ("est_ba", "xBA", "{:.3f}"),
                ("est_slg", "xSLG", "{:.3f}"),
            ]:
                v = row.get(key)
                pct = _pct_rank(pool[key], v, "high")
                results.append(MetricResult(key, label, v, pct, "high", fmt))

    # --- Exit velocity / barrels / hard-hit ---
    ev = _exit_velocity()
    if not ev.empty:
        pool = ev[(ev["year"] == year) & (ev["attempts"] >= 150)]
        row = _find_player_row(ev, mlbam_id, year)
        if row is not None:
            for key, label, fmt in [
                ("avg_hit_speed", "Avg Exit Velocity", "{:.1f}"),
                ("brl_percent", "Barrel %", "{:.1f}"),
                ("ev95percent", "Hard-Hit %", "{:.1f}"),
                ("anglesweetspotpercent", "Sweet-Spot %", "{:.1f}"),
                ("max_hit_speed", "Max Exit Velocity", "{:.1f}"),
            ]:
                v = row.get(key)
                pct = _pct_rank(pool[key], v, "high")
                results.append(MetricResult(key, label, v, pct, "high", fmt))

    # --- Bat tracking (2023+) ---
    # Savant's year-filtered bat-tracking endpoint always returns current-season data,
    # so raw values are only trustworthy for the current year. Use Savant's
    # precomputed percentiles from percentile_rankings_batter for year-specific pct.
    bt = _bat_tracking()
    bt_row = _find_player_row(bt, mlbam_id, year, id_col="id") if not bt.empty else None
    if bt_row is not None and year >= 2023:
        # Bat speed
        bs_pct = _savant_pct(mlbam_id, year, "batter", "bat_speed")
        bs_val = bt_row.get("avg_bat_speed") if year == bt["year"].max() else None
        results.append(MetricResult("bat_speed", "Bat Speed (mph)",
                                    bs_val, bs_pct, "high", "{:.1f}"))
        # Squared-up %
        su_pct = _savant_pct(mlbam_id, year, "batter", "squared_up_rate")
        su_val = (bt_row.get("squared_up_per_swing") * 100
                  if year == bt["year"].max() and not pd.isna(bt_row.get("squared_up_per_swing"))
                  else None)
        results.append(MetricResult("squared_up", "Squared-Up %",
                                    su_val, su_pct, "high"))

    # --- Chase % / Whiff % ---
    bsf = _batter_season_features()
    chase_raw = whiff_raw = None
    if not bsf.empty and lahman_id:
        sub = bsf[(bsf["playerID"] == lahman_id) & (bsf["yearID"] == year)]
        if not sub.empty:
            row = sub.iloc[0]
            # These are decimals (0.24), convert to percentage for display
            chase_raw = row.get("chase_pct") * 100 if not pd.isna(row.get("chase_pct")) else None
            whiff_raw = row.get("whiff_pct") * 100 if not pd.isna(row.get("whiff_pct")) else None
    # Use Savant's precomputed percentiles for chase/whiff (year-specific, reliable)
    chase_pct = _savant_pct(mlbam_id, year, "batter", "chase_percent")
    whiff_pct = _savant_pct(mlbam_id, year, "batter", "whiff_percent")
    results.append(MetricResult("whiff_pct", "Whiff %", whiff_raw, whiff_pct, "low"))
    results.append(MetricResult("chase_pct", "Chase %", chase_raw, chase_pct, "low"))

    # K% / BB% from Lahman
    bat = _batting()
    if lahman_id:
        pool = bat[(bat["yearID"] == year) & (bat["PA"] >= 300)]
        sub = bat[(bat["playerID"] == lahman_id) & (bat["yearID"] == year)]
        if not sub.empty:
            row = sub.iloc[0]
            v = row["K_pct"]
            pct = _pct_rank(pool["K_pct"], v, "low")
            results.append(MetricResult("k_pct", "K %", v, pct, "low"))
            v = row["BB_pct"]
            pct = _pct_rank(pool["BB_pct"], v, "high")
            results.append(MetricResult("bb_pct", "BB %", v, pct, "high"))

    # --- Sprint Speed ---
    ss = _sprint_speed()
    if not ss.empty:
        pool = ss[(ss["year"] == year) & (ss["competitive_runs"] >= 10)]
        row = _find_player_row(ss, mlbam_id, year)
        if row is not None:
            v = row.get("sprint_speed")
            pct = _pct_rank(pool["sprint_speed"], v, "high")
            results.append(MetricResult("sprint_speed", "Sprint Speed (ft/s)", v, pct, "high"))

    # --- OAA (Range) ---
    oaa = _oaa()
    if not oaa.empty:
        pool = oaa[oaa["year"] == year]
        row = _find_player_row(oaa, mlbam_id, year)
        if row is not None:
            v = row.get("outs_above_average")
            pct = _pct_rank(pool["outs_above_average"], v, "high")
            results.append(MetricResult("oaa", "Range (OAA)", v, pct, "high", "{:.0f}"))

    # --- Arm Strength (outfielder) ---
    arm = _arm_strength()
    if not arm.empty:
        pool = arm[(arm["year"] == year) & (arm["total_throws_of"] >= 20)]
        row = _find_player_row(arm, mlbam_id, year)
        if row is not None:
            v = row.get("arm_overall")
            if v is not None and not pd.isna(v):
                pct = _pct_rank(pool["arm_overall"], v, "high")
                results.append(MetricResult("arm_strength", "Arm Strength (mph)", v, pct, "high"))

    # --- Catcher metrics ---
    fr = _framing()
    if not fr.empty:
        pool = fr[(fr["year"] == year) & (fr["pitches"] >= 1500)]
        sub = fr[(fr["id"] == mlbam_id) & (fr["year"] == year)]
        if not sub.empty:
            v = sub.iloc[0].get("rv_tot")
            if v is not None and not pd.isna(v):
                pct = _pct_rank(pool["rv_tot"], v, "high")
                results.append(MetricResult("framing", "Framing Runs", v, pct, "high", "{:.0f}"))

    pop = _poptime()
    if not pop.empty:
        pool = pop[(pop["year"] == year)]
        sub = pop[(pop["entity_id"] == mlbam_id) & (pop["year"] == year)] \
            if "entity_id" in pop.columns else pd.DataFrame()
        if not sub.empty:
            v = sub.iloc[0].get("pop_2b_sba")
            if v is not None and not pd.isna(v):
                pct = _pct_rank(pool["pop_2b_sba"], v, "low")  # faster pop = lower seconds = better
                results.append(MetricResult("pop_time", "Pop Time (2B)", v, pct, "low", "{:.2f}"))

    cb = _catcher_blocking()
    if not cb.empty:
        year_col = "start_year" if "start_year" in cb.columns else "year"
        pool = cb[cb[year_col] == year]
        sub = cb[(cb["player_id"] == mlbam_id) & (cb[year_col] == year)]
        if not sub.empty:
            v = sub.iloc[0].get("blocks_above_average")
            if v is not None and not pd.isna(v):
                pct = _pct_rank(pool["blocks_above_average"], v, "high")
                results.append(MetricResult("blocks_aa", "Blocks Above Avg", v, pct, "high", "{:.0f}"))

    ct = _catcher_throwing()
    if not ct.empty:
        year_col = "start_year" if "start_year" in ct.columns else "year"
        pool = ct[ct[year_col] == year]
        sub = ct[(ct["player_id"] == mlbam_id) & (ct[year_col] == year)]
        if not sub.empty:
            v = sub.iloc[0].get("caught_stealing_above_average")
            if v is not None and not pd.isna(v):
                pct = _pct_rank(pool["caught_stealing_above_average"], v, "high")
                results.append(MetricResult("cs_aa", "CS Above Avg", v, pct, "high", "{:.0f}"))

    # --- Value runs ---
    value_rows: list[MetricResult] = []

    # BRAVS-based run values (always available, year-specific)
    br = _bravs()
    if lahman_id:
        pool = br[(br["yearID"] == year) & (br["PA"] >= 300)]
        sub = br[(br["playerID"] == lahman_id) & (br["yearID"] == year)]
        if not sub.empty:
            row = sub.iloc[0]
            v = row.get("hitting_runs")
            pct = _pct_rank(pool["hitting_runs"], v, "high")
            value_rows.append(MetricResult("bat_rv", "Batting Run Value", v, pct, "high", "{:.0f}"))
            v = row.get("baserunning_runs")
            pct = _pct_rank(pool["baserunning_runs"], v, "high")
            value_rows.append(MetricResult("bsr_rv", "Baserunning Run Value", v, pct, "high", "{:.0f}"))
            # Prefer Savant-style OAA-based fielding RV when available
            oaa = _oaa()
            fld_from_oaa = None
            if not oaa.empty:
                oaa_pool = oaa[oaa["year"] == year]
                oaa_row = _find_player_row(oaa, mlbam_id, year)
                if oaa_row is not None and "fielding_runs_prevented" in oaa_row:
                    v = oaa_row.get("fielding_runs_prevented")
                    if v is not None and not pd.isna(v):
                        pct = _pct_rank(oaa_pool["fielding_runs_prevented"], v, "high")
                        value_rows.append(MetricResult("fld_rv", "Fielding Run Value",
                                                       v, pct, "high", "{:.0f}"))
                        fld_from_oaa = True
            if not fld_from_oaa:
                v = row.get("fielding_runs")
                pct = _pct_rank(pool["fielding_runs"], v, "high")
                value_rows.append(MetricResult("fld_rv", "Fielding Run Value",
                                               v, pct, "high", "{:.0f}"))

    # Group results by section
    return _organize_batter(results, value_rows)


def _organize_batter(results: list[MetricResult], value_rows: list[MetricResult]) -> dict:
    # Order matters — list keys in the order they should appear
    order = ["est_woba", "est_ba", "est_slg", "avg_hit_speed", "brl_percent",
             "ev95percent", "anglesweetspotpercent", "max_hit_speed",
             "bat_speed", "squared_up", "chase_pct", "whiff_pct",
             "k_pct", "bb_pct"]
    by_key = {r.key: r for r in results}
    batting = [by_key[k].to_dict() for k in order if k in by_key]
    keys_batting = set(order)
    keys_fielding = {"oaa", "arm_strength"}
    keys_running = {"sprint_speed"}
    keys_catching = {"framing", "pop_time", "blocks_aa", "cs_aa"}

    out = {
        "value": [r.to_dict() for r in value_rows],
        "batting": batting,
        "fielding": [r.to_dict() for r in results if r.key in keys_fielding],
        "running": [r.to_dict() for r in results if r.key in keys_running],
        "catching": [r.to_dict() for r in results if r.key in keys_catching],
    }
    return out


def pitcher_card(mlbam_id: int, year: int) -> dict:
    cw = _crosswalk()
    lahman_id = cw.get(mlbam_id)

    results: list[MetricResult] = []

    # --- Expected pitching (xERA, xBA, xSLG, xwOBA) ---
    xp = _expected_pitching()
    if not xp.empty:
        pool = xp[(xp["year"] == year) & (xp["pa"] >= 150)]
        row = _find_player_row(xp, mlbam_id, year)
        if row is not None:
            for key, label, direction, fmt in [
                ("xera", "xERA", "low", "{:.2f}"),
                ("est_woba", "xwOBA", "low", "{:.3f}"),
                ("est_ba", "xBA", "low", "{:.3f}"),
                ("est_slg", "xSLG", "low", "{:.3f}"),
            ]:
                v = row.get(key)
                pct = _pct_rank(pool[key], v, direction)
                results.append(MetricResult(key, label, v, pct, direction, fmt))

    # --- Pitcher season features (velo, whiff, xwoba allowed) ---
    psf = _pitcher_season_features()
    if not psf.empty and lahman_id:
        pool = psf[(psf["yearID"] == year) & (psf["bf"] >= 150)]
        sub = psf[(psf["playerID"] == lahman_id) & (psf["yearID"] == year)]
        if not sub.empty:
            row = sub.iloc[0]
            v = row.get("velo_avg")
            if v is not None and not pd.isna(v):
                pct = _pct_rank(pool["velo_avg"], v, "high")
                results.append(MetricResult("velo", "Fastball Velo (mph)", v, pct, "high"))
            v = row.get("whiff_pct")
            if v is not None and not pd.isna(v):
                v_disp = v * 100 if v < 1.5 else v
                pool_v = pool["whiff_pct"] * 100 if pool["whiff_pct"].median() < 1.5 else pool["whiff_pct"]
                pct = _pct_rank(pool_v, v_disp, "high")
                results.append(MetricResult("whiff", "Whiff %", v_disp, pct, "high"))

    # --- K% / BB% from Lahman pitching ---
    pit = _pitching()
    if lahman_id:
        pool = pit[(pit["yearID"] == year) & (pit["IP"] >= 40)]
        sub = pit[(pit["playerID"] == lahman_id) & (pit["yearID"] == year)]
        if not sub.empty:
            row = sub.iloc[0]
            v = row["K_pct"]
            pct = _pct_rank(pool["K_pct"], v, "high")
            results.append(MetricResult("k_pct", "K %", v, pct, "high"))
            v = row["BB_pct"]
            pct = _pct_rank(pool["BB_pct"], v, "low")
            results.append(MetricResult("bb_pct", "BB %", v, pct, "low"))

    # --- Exit velocity allowed: use the pitcher-side exit_velocity file if we have one ---
    # The exit_velocity_*.csv has player_id — but they're batter IDs. For pitcher,
    # we'd need a different aggregate. Skip for now; BRAVS doesn't separate.

    # --- Value runs ---
    value_rows: list[MetricResult] = []
    br = _bravs()
    if lahman_id:
        pool = br[(br["yearID"] == year) & (br["IP"] >= 40) & (br["position"] == "P")]
        sub = br[(br["playerID"] == lahman_id) & (br["yearID"] == year)]
        if not sub.empty:
            row = sub.iloc[0]
            v = row.get("pitching_runs")
            pct = _pct_rank(pool["pitching_runs"], v, "high")
            value_rows.append(MetricResult("pitch_rv", "Pitching Run Value", v, pct, "high", "{:.0f}"))

    keys_pitching = {"xera", "est_woba", "est_ba", "est_slg", "velo", "whiff",
                     "k_pct", "bb_pct"}

    return {
        "value": [r.to_dict() for r in value_rows],
        "pitching": [r.to_dict() for r in results if r.key in keys_pitching],
    }


# -----------------------------------------------------------------------------
# Player search
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _search_index() -> pd.DataFrame:
    """Build name-to-ID index from the Savant percentile tables (has MLBAM IDs + names)."""
    frames = []
    for fn in [_savant_pct_batter, _savant_pct_pitcher]:
        df = fn()
        if df.empty:
            continue
        sub = df[["player_id", "player_name", "year"]].copy()
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["player_id", "player_name", "year"])
    idx = pd.concat(frames, ignore_index=True).drop_duplicates(["player_id", "year"])
    idx["player_id"] = idx["player_id"].astype(int)
    idx["year"] = idx["year"].astype(int)
    # Flip "Last, First" → "First Last" for display
    def flip(n: str) -> str:
        if "," in str(n):
            last, first = n.split(",", 1)
            return f"{first.strip()} {last.strip()}"
        return str(n)
    idx["display_name"] = idx["player_name"].apply(flip)
    return idx


def search_players(query: str, limit: int = 20) -> list[dict]:
    idx = _search_index()
    if idx.empty or not query:
        return []
    q = query.lower().strip()
    # filter
    mask = idx["display_name"].str.lower().str.contains(q, na=False, regex=False)
    hits = idx[mask]
    if hits.empty:
        return []
    # Aggregate by player: collect years
    by_id = hits.groupby(["player_id", "display_name"])["year"].apply(
        lambda s: sorted(set(int(x) for x in s), reverse=True)
    ).reset_index()
    by_id = by_id.head(limit)
    return [
        {"id": int(r.player_id), "name": r.display_name, "years": r.year}
        for r in by_id.itertuples()
    ]


def get_years_for_player(mlbam_id: int) -> list[int]:
    idx = _search_index()
    sub = idx[idx["player_id"] == mlbam_id]
    if sub.empty:
        return []
    return sorted(set(int(y) for y in sub["year"]), reverse=True)


def player_card(mlbam_id: int, year: int) -> dict:
    idx = _search_index()
    name_sub = idx[idx["player_id"] == mlbam_id]
    display_name = name_sub.iloc[0]["display_name"] if not name_sub.empty else f"Player {mlbam_id}"

    kind = classify_player(mlbam_id, year)
    if kind == "pitcher":
        data = pitcher_card(mlbam_id, year)
        data["kind"] = "pitcher"
    else:
        data = batter_card(mlbam_id, year)
        data["kind"] = "batter"
    data["name"] = display_name
    data["mlbam_id"] = mlbam_id
    data["year"] = year
    return data
