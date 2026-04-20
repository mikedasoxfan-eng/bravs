"""BRAVS Web App — Compute BRAVS for any MLB player, any season."""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from flask import Flask, Response, render_template, jsonify, request as flask_request

import pandas as pd

from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import PlayerSeason
from baseball_metric.adjustments.era_adjustment import get_rpg
from baseball_metric.adjustments.park_factors import get_park_factor

# Load pre-computed BRAVS from GPU engine (instant, no API needed)
_PRECOMPUTED = None
_CAREERS = None
_BRAVS_2026 = None
_MLB_TO_LAHMAN: dict[int, str] | None = None
_LAHMAN_TO_MLB: dict[str, int] | None = None

# MLB-API team abbreviations -> Lahman/Retrosheet codes used in
# bravs_all_seasons.csv. Most teams are identical; only list differences.
_MLB_TO_LAHMAN_TEAM = {
    "CWS": "CHA", "CHW": "CHA",
    "CHC": "CHN",
    "NYY": "NYA",
    "NYM": "NYN",
    "KC": "KCA", "KCR": "KCA",
    "LAD": "LAN",
    "SF": "SFN", "SFG": "SFN",
    "SD": "SDN", "SDP": "SDN",
    "STL": "SLN",
    "TB": "TBA", "TBR": "TBA",
    "WSH": "WAS", "WSN": "WAS",
}


def _team_to_lahman(abbrev: str) -> str:
    """Convert an MLB-style team abbreviation to the Lahman/Retrosheet code
    used in bravs_all_seasons.csv (e.g. NYY -> NYA, CWS -> CHA)."""
    if not abbrev:
        return abbrev
    a = abbrev.upper()
    return _MLB_TO_LAHMAN_TEAM.get(a, a)


def _load_crosswalk():
    """Lazy-load the MLB <-> Lahman ID crosswalk."""
    global _MLB_TO_LAHMAN, _LAHMAN_TO_MLB
    if _MLB_TO_LAHMAN is None:
        try:
            cw = pd.read_csv("data/id_crosswalk.csv")
            _MLB_TO_LAHMAN = dict(zip(cw.mlbam_id.astype(int), cw.lahman_id))
            _LAHMAN_TO_MLB = dict(zip(cw.lahman_id, cw.mlbam_id.astype(int)))
        except Exception as e:
            logger.warning("Could not load id crosswalk: %s", e)
            _MLB_TO_LAHMAN = {}
            _LAHMAN_TO_MLB = {}
    return _MLB_TO_LAHMAN, _LAHMAN_TO_MLB


def _mlb_to_lahman_id(mlb_id: int | str) -> str | None:
    """Return the Lahman playerID for an MLB ID, or None."""
    m, _ = _load_crosswalk()
    try:
        return m.get(int(mlb_id))
    except (ValueError, TypeError):
        return None


def _load_precomputed():
    global _PRECOMPUTED, _CAREERS
    if _PRECOMPUTED is None:
        try:
            _PRECOMPUTED = pd.read_csv("data/bravs_all_seasons.csv")
            _CAREERS = pd.read_csv("data/bravs_careers.csv")
            logger.info("Loaded pre-computed BRAVS: %d seasons, %d careers", len(_PRECOMPUTED), len(_CAREERS))
        except Exception as e:
            logger.warning("Could not load pre-computed data: %s", e)
            _PRECOMPUTED = pd.DataFrame()
            _CAREERS = pd.DataFrame()
    return _PRECOMPUTED, _CAREERS


def _load_bravs_2026():
    """Load the live 2026 BRAVS table built daily by the assembler.
    PlayerIDs in this CSV are MLB integer IDs (not Lahman strings)."""
    global _BRAVS_2026
    if _BRAVS_2026 is None:
        try:
            _BRAVS_2026 = pd.read_csv("data/2026/bravs_2026.csv")
            logger.info("Loaded 2026 BRAVS: %d player-seasons", len(_BRAVS_2026))
        except Exception as e:
            logger.warning("Could not load 2026 BRAVS: %s", e)
            _BRAVS_2026 = pd.DataFrame()
    return _BRAVS_2026


# Try Rust engine for faster computation
try:
    from bravs_engine import compute_bravs_fast as _rust_compute
    USE_RUST = True
    logger_msg = "Rust engine loaded"
except ImportError:
    USE_RUST = False
    logger_msg = "Rust engine not available, using Python"

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize SQLite cache for persistent caching across restarts
try:
    from web.cache import init_cache, get_api as _cache_get, set_api as _cache_set
    from web.cache import get_bravs as _cache_get_bravs, set_bravs as _cache_set_bravs
    init_cache()
    HAS_CACHE = True
except Exception:
    HAS_CACHE = False

MLB_API = "https://statsapi.mlb.com/api/v1"
HEADSHOT_URL = (
    "https://img.mlbstatic.com/mlb-photos/image/upload/"
    "d_people:generic:headshot:67:current.png/"
    "w_213,q_auto:best/v1/people/{pid}/headshot/67/current"
)
TEAM_LOGO_URL = "https://www.mlbstatic.com/team-logos/{tid}.svg"

# Simple in-memory cache
_cache: dict[str, object] = {}


def _display_name(full_name: str) -> str:
    """Strip middle names/suffixes for cleaner display.

    'George Herman Ruth' -> 'George Ruth'
    'Michael Nelson Trout' -> 'Michael Trout'
    'Vladimir Guerrero Jr.' -> 'Vladimir Guerrero Jr.'
    """
    parts = full_name.strip().split()
    if len(parts) <= 2:
        return full_name

    # Keep suffixes like Jr., Sr., II, III, IV
    suffixes = {"Jr.", "Sr.", "II", "III", "IV", "Jr", "Sr"}
    last_parts: list[str] = []
    while parts and parts[-1] in suffixes:
        last_parts.insert(0, parts.pop())
    if not parts:
        return full_name

    first = parts[0]
    last = parts[-1] if len(parts) > 1 else ""
    return " ".join([first, last] + last_parts).strip()


def _row_to_bravs_dict(r: pd.Series) -> dict:
    """Format one row of a precomputed BRAVS CSV into the API response shape."""
    result = {
        "player_name": r.get("name", "Unknown"),
        "season": int(r.yearID),
        "position": r.get("position", ""),
        "team": r.get("team", ""),
        "team_abbrev": r.get("team", ""),
        "lgID": r.get("lgID", ""),
        "bravs": round(float(r.bravs), 1),
        "bravs_era_std": round(float(r.bravs_era_std), 1),
        "bravs_war_eq": round(float(r.bravs_war_eq), 1),
        "ci90_lo": round(float(r.ci90_lo), 1),
        "ci90_hi": round(float(r.ci90_hi), 1),
        "total_runs": round(float(r.bravs) * float(r.rpw), 1),
        "rpw": round(float(r.rpw), 2),
        "leverage_mult": 1.0,
        "park_factor": 1.0,
        "engine": "gpu_precomputed",
        "headshot": "",
        "components": [
            {"name": "hitting", "runs": round(float(r.get("hitting_runs", 0)), 1), "ci_lo": 0, "ci_hi": 0},
            {"name": "pitching", "runs": round(float(r.get("pitching_runs", 0)), 1), "ci_lo": 0, "ci_hi": 0},
            {"name": "baserunning", "runs": round(float(r.get("baserunning_runs", 0)), 1), "ci_lo": 0, "ci_hi": 0},
            {"name": "fielding", "runs": round(float(r.get("fielding_runs", 0)), 1), "ci_lo": 0, "ci_hi": 0},
            {"name": "positional", "runs": round(float(r.get("positional_runs", 0)), 1), "ci_lo": 0, "ci_hi": 0},
            {"name": "durability", "runs": round(float(r.get("durability_runs", 0)), 1), "ci_lo": 0, "ci_hi": 0},
            {"name": "approach_quality", "runs": round(float(r.get("aqi_runs", 0)), 1), "ci_lo": 0, "ci_hi": 0},
            {"name": "leverage", "runs": round(float(r.get("leverage_runs", 0)), 1), "ci_lo": 0, "ci_hi": 0},
        ],
        "traditional": {
            "batting": {"G": int(r.G), "PA": int(r.PA), "HR": int(r.HR), "SB": int(r.get("SB", 0))},
        } if r.PA > 0 else {},
    }
    if r.IP > 0:
        result["traditional"]["pitching"] = {"IP": round(float(r.IP), 1), "G": int(r.G)}

    return result


def _get_precomputed_bravs(player_id_or_name, season: int) -> dict | None:
    """Look up pre-computed BRAVS for one (player, season).

    `player_id_or_name` accepts an MLB integer id, a Lahman string id, or a
    name (last-resort fuzzy match). The function dispatches to the 2026 live
    table for season==2026 and to the historical 1920-2025 table otherwise.
    """
    season = int(season)

    # 2026 live table uses MLB integer playerIDs natively
    if season == 2026:
        df = _load_bravs_2026()
        if df.empty:
            return None
        try:
            mlb_id = int(player_id_or_name)
            row = df[(df.playerID == mlb_id) & (df.yearID == 2026)]
        except (ValueError, TypeError):
            row = df[(df.name.str.contains(str(player_id_or_name), case=False,
                                            na=False)) & (df.yearID == 2026)]
        if row.empty:
            return None
        return _row_to_bravs_dict(row.iloc[0])

    # Historical table uses Lahman string playerIDs. Resolve MLB ints first.
    seasons, _ = _load_precomputed()
    if seasons.empty:
        return None

    lahman_id = None
    if isinstance(player_id_or_name, (int,)) or (
        isinstance(player_id_or_name, str) and player_id_or_name.isdigit()
    ):
        lahman_id = _mlb_to_lahman_id(player_id_or_name)
    elif isinstance(player_id_or_name, str):
        # Already a Lahman id like "troutmi01"
        lahman_id = player_id_or_name

    if lahman_id:
        row = seasons[(seasons.playerID == lahman_id)
                      & (seasons.yearID == season)]
        if not row.empty:
            return _row_to_bravs_dict(row.iloc[0])

    # Last-resort: fuzzy name match
    row = seasons[
        (seasons.name.str.contains(str(player_id_or_name),
                                    case=False, na=False))
        & (seasons.yearID == season)
    ]
    if row.empty:
        return None
    return _row_to_bravs_dict(row.iloc[0])


# Team abbreviation -> MLB team ID
TEAM_IDS: dict[str, int] = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CWS": 145, "CHW": 145, "CIN": 113, "CLE": 114, "COL": 115,
    "DET": 116, "HOU": 117, "KC": 118, "KCR": 118, "LAA": 108,
    "ANA": 108, "CAL": 108, "LAD": 119, "MIA": 146, "FLA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SD": 135, "SDP": 135, "SF": 137,
    "SFG": 137, "SEA": 136, "STL": 138, "TB": 139, "TBR": 139,
    "TEX": 140, "TOR": 141, "WSH": 120, "WAS": 120, "WSN": 120,
    "MON": 120,
}

SEASON_LENGTHS: dict[int, int] = {2020: 60, 1994: 115, 1995: 144, 1981: 107}


def _mlb_get(url: str, params: dict | None = None) -> dict | None:
    """Fetch from MLB Stats API with multi-layer caching (memory + SQLite)."""
    cache_key = url + str(params or {})

    # Layer 1: in-memory cache (fastest)
    if cache_key in _cache:
        return _cache[cache_key]  # type: ignore[return-value]

    # Layer 2: SQLite persistent cache (survives restarts)
    if HAS_CACHE:
        cached = _cache_get(cache_key)
        if cached is not None:
            _cache[cache_key] = cached  # promote to memory
            return cached

    # Layer 3: actual API call
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        _cache[cache_key] = data
        if HAS_CACHE:
            _cache_set(cache_key, data)
        return data
    except Exception as e:
        logger.warning("MLB API error: %s", e)
        return None


def _get_team_abbrev(team_obj: dict) -> str:
    """Extract team abbreviation from MLB API team object."""
    return team_obj.get("abbreviation", team_obj.get("name", "UNK")[:3].upper())


def _safe_int(val: object, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val: object, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _parse_ip(val: object) -> float:
    """Parse innings pitched, handling the .1/.2 notation (1/3, 2/3 innings)."""
    if val is None:
        return 0.0
    s = str(val)
    if "." in s:
        parts = s.split(".")
        whole = int(parts[0])
        frac = int(parts[1]) if parts[1] else 0
        return whole + frac / 3.0
    return float(s)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def search_players():
    q = flask_request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])

    data = _mlb_get(f"{MLB_API}/people/search", {"names": q, "limit": 12})
    if not data or "people" not in data:
        return jsonify([])

    results = []
    for p in data["people"][:12]:
        team = p.get("currentTeam", {})
        pos = p.get("primaryPosition", {})
        debut = p.get("mlbDebutDate", "")
        last_played = p.get("lastPlayedDate", "")

        debut_year = int(debut[:4]) if debut else None
        last_year = int(last_played[:4]) if last_played else None

        raw_name = p.get("fullFMLName", p.get("fullName", "Unknown"))
        results.append({
            "id": p["id"],
            "name": _display_name(raw_name),
            "position": pos.get("abbreviation", ""),
            "team": team.get("name", ""),
            "team_abbrev": _get_team_abbrev(team),
            "team_id": team.get("id"),
            "active": p.get("active", False),
            "debut_year": debut_year,
            "last_year": last_year,
            "headshot": HEADSHOT_URL.format(pid=p["id"]),
            "bats_throws": f"{p.get('batSide', {}).get('code', '?')}/{p.get('pitchHand', {}).get('code', '?')}",
            "number": p.get("primaryNumber", ""),
        })

    return jsonify(results)


@app.route("/api/player/<int:player_id>/seasons")
def player_seasons(player_id: int):
    """Get all MLB seasons for a player."""
    hitting = _mlb_get(
        f"{MLB_API}/people/{player_id}/stats",
        {"stats": "yearByYear", "group": "hitting", "gameType": "R"},
    )
    pitching = _mlb_get(
        f"{MLB_API}/people/{player_id}/stats",
        {"stats": "yearByYear", "group": "pitching", "gameType": "R"},
    )

    seasons: dict[int, dict] = {}

    if hitting and "stats" in hitting:
        for group in hitting["stats"]:
            for split in group.get("splits", []):
                yr = int(split.get("season", 0))
                if yr < 1871:
                    continue
                s = split.get("stat", {})
                team = split.get("team", {})
                pos = split.get("position", {})
                if yr not in seasons:
                    seasons[yr] = {
                        "season": yr, "team": team.get("name", ""),
                        "team_abbrev": _get_team_abbrev(team),
                        "team_id": team.get("id"),
                        "position": pos.get("abbreviation", ""),
                        "type": "hitter",
                    }
                seasons[yr]["games"] = _safe_int(s.get("gamesPlayed"))
                seasons[yr]["pa"] = _safe_int(s.get("plateAppearances"))
                seasons[yr]["avg"] = s.get("avg", "")
                seasons[yr]["hr"] = _safe_int(s.get("homeRuns"))
                seasons[yr]["ops"] = s.get("ops", "")

    if pitching and "stats" in pitching:
        for group in pitching["stats"]:
            for split in group.get("splits", []):
                yr = int(split.get("season", 0))
                if yr < 1871:
                    continue
                s = split.get("stat", {})
                team = split.get("team", {})
                if yr not in seasons:
                    seasons[yr] = {
                        "season": yr, "team": team.get("name", ""),
                        "team_abbrev": _get_team_abbrev(team),
                        "team_id": team.get("id"),
                        "position": "P", "type": "pitcher",
                    }
                ip = _parse_ip(s.get("inningsPitched", 0))
                if ip > _safe_float(seasons[yr].get("ip", 0)):
                    seasons[yr]["ip"] = round(ip, 1)
                    seasons[yr]["era"] = s.get("era", "")
                    seasons[yr]["k_pitch"] = _safe_int(s.get("strikeOuts"))
                    if ip > 20:
                        seasons[yr]["type"] = "pitcher"

    result = sorted(seasons.values(), key=lambda x: x["season"], reverse=True)
    return jsonify(result)


@app.route("/api/player/<int:player_id>/bravs/<int:season>")
def compute_player_bravs(player_id: int, season: int):
    """Fetch stats and compute BRAVS for a player-season."""

    # Get player info
    player_info = _mlb_get(f"{MLB_API}/people/{player_id}")
    if not player_info or "people" not in player_info:
        return jsonify({"error": "Player not found"}), 404
    pinfo = player_info["people"][0]

    # Try pre-computed GPU data first (instant, no live compute).
    # 1920-2025 -> bravs_all_seasons.csv via MLB->Lahman crosswalk.
    # 2026      -> live 2026/bravs_2026.csv (MLB IDs native).
    if 1920 <= season <= 2026:
        precomputed = _get_precomputed_bravs(player_id, season)
        if precomputed is None:
            name = pinfo.get("fullName", "")
            precomputed = _get_precomputed_bravs(name, season)
        if precomputed:
            precomputed["headshot"] = HEADSHOT_URL.format(pid=player_id)
            _tid = TEAM_IDS.get(precomputed.get("team_abbrev", ""))
            precomputed["team_logo"] = TEAM_LOGO_URL.format(tid=_tid) if _tid else None
            precomputed["team_id"] = _tid or 0
            precomputed["player_name"] = _display_name(pinfo.get("fullName", precomputed.get("player_name", "")))
            precomputed["bats_throws"] = f"{pinfo.get('batSide', {}).get('code', '?')}/{pinfo.get('pitchHand', {}).get('code', '?')}"
            precomputed["number"] = pinfo.get("primaryNumber", "")
            return jsonify(precomputed)

    # Get season stats
    hitting_data = _mlb_get(
        f"{MLB_API}/people/{player_id}/stats",
        {"stats": "season", "season": season, "group": "hitting", "gameType": "R"},
    )
    pitching_data = _mlb_get(
        f"{MLB_API}/people/{player_id}/stats",
        {"stats": "season", "season": season, "group": "pitching", "gameType": "R"},
    )
    # Fetch fielding stats to determine actual position played that season
    fielding_data = _mlb_get(
        f"{MLB_API}/people/{player_id}/stats",
        {"stats": "season", "season": season, "group": "fielding", "gameType": "R"},
    )

    # Determine position from fielding data (most innings), NOT from player info
    # (player info shows current position, which may differ from historical)
    position = "DH"
    if fielding_data and "stats" in fielding_data:
        best_inn = 0.0
        for group in fielding_data["stats"]:
            for split in group.get("splits", []):
                pos_obj = split.get("position", {})
                pos_abbr = pos_obj.get("abbreviation", "")
                if pos_abbr in ("P", "DH", ""):
                    continue
                inn = _parse_ip(split.get("stat", {}).get("innings", 0))
                if inn > best_inn:
                    best_inn = inn
                    position = pos_abbr
    if position == "DH":
        # Fallback: check if they pitched more than hit
        position = pinfo.get("primaryPosition", {}).get("abbreviation", "DH")

    # Parse hitting stats (combine if traded mid-season)
    h: dict[str, object] = {}
    team_name = ""
    team_abbrev = "UNK"
    team_id = 0

    if hitting_data and "stats" in hitting_data:
        for group in hitting_data["stats"]:
            splits = group.get("splits", [])
            if not splits:
                continue

            # MLB API returns: total row (numTeams > 1) + per-team splits.
            # Use the total row if it exists, otherwise use the single split.
            total_split = None
            team_splits = []
            for split in splits:
                if _safe_int(split.get("numTeams", 1)) > 1:
                    total_split = split
                else:
                    team_splits.append(split)

            chosen = total_split or (team_splits[0] if len(team_splits) == 1 else None)
            if not chosen:
                # No total row and multiple teams — pick the one with most PA
                chosen = max(splits, key=lambda sp: _safe_int(sp.get("stat", {}).get("plateAppearances")))

            s = chosen.get("stat", {})
            team_obj = chosen.get("team", {})
            # For traded players, get team from the last team split for display
            if total_split and team_splits:
                team_obj = team_splits[-1].get("team", team_obj)
            team_name = team_obj.get("name", team_name)
            team_abbrev = _get_team_abbrev(team_obj)
            team_id = team_obj.get("id", team_id)

            combined: dict[str, int | float] = {}
            for key in ["gamesPlayed", "plateAppearances", "atBats", "hits",
                        "doubles", "triples", "homeRuns", "baseOnBalls",
                        "intentionalWalks", "hitByPitch", "strikeOuts",
                        "sacFlies", "sacBunts", "stolenBases", "caughtStealing",
                        "groundIntoDoublePlay"]:
                combined[key] = _safe_int(s.get(key))

            h = combined
            h["avg"] = s.get("avg", "")
            h["obp"] = s.get("obp", "")
            h["slg"] = s.get("slg", "")
            h["ops"] = s.get("ops", "")

    # Parse pitching stats — same total-row logic
    p: dict[str, object] = {}
    if pitching_data and "stats" in pitching_data:
        for group in pitching_data["stats"]:
            splits = group.get("splits", [])
            if not splits:
                continue

            total_split = None
            team_splits = []
            for split in splits:
                if _safe_int(split.get("numTeams", 1)) > 1:
                    total_split = split
                else:
                    team_splits.append(split)

            chosen = total_split or (team_splits[0] if len(team_splits) == 1 else None)
            if not chosen:
                chosen = max(splits, key=lambda sp: _parse_ip(sp.get("stat", {}).get("inningsPitched", 0)))

            s = chosen.get("stat", {})
            team_obj = chosen.get("team", {})
            if not team_name:
                if total_split and team_splits:
                    team_obj = team_splits[-1].get("team", team_obj)
                team_name = team_obj.get("name", "")
                team_abbrev = _get_team_abbrev(team_obj)
                team_id = team_obj.get("id", 0)

            combined_p: dict[str, object] = {"ip": _parse_ip(s.get("inningsPitched", 0))}
            for key in ["gamesPlayed", "gamesStarted", "earnedRuns", "hits",
                        "homeRuns", "baseOnBalls", "hitBatsmen", "strikeOuts",
                        "saves", "holds", "wins", "losses"]:
                combined_p[key] = _safe_int(s.get(key))

            p = combined_p
            p["era"] = s.get("era", "")
            p["whip"] = s.get("whip", "")

    if not h and not p:
        return jsonify({"error": f"No stats found for {season}"}), 404

    # Determine context
    league_rpg = get_rpg(season)
    pf = get_park_factor(team_abbrev, season)
    season_length = SEASON_LENGTHS.get(season, 162)

    # Detect in-progress season: if this is the current year and the player
    # has fewer games than a typical full season, check the schedule to see
    # how many games the team has actually played.
    import datetime
    current_year = datetime.date.today().year
    games_played = max(_safe_int(h.get("gamesPlayed")), _safe_int(p.get("gamesPlayed")))
    if season >= current_year and games_played < 140 and season_length == 162:
        # Query schedule to find actual team games played
        if team_id:
            sched = _mlb_get(
                f"{MLB_API}/schedule",
                {"sportId": 1, "season": season, "teamId": team_id, "gameType": "R"},
            )
            if sched and "dates" in sched:
                team_games_played = sum(
                    1 for dt in sched["dates"]
                    for g in dt.get("games", [])
                    if g.get("status", {}).get("abstractGameState") == "Final"
                )
                if team_games_played > 0 and team_games_played < 155:
                    season_length = team_games_played

    # Build PlayerSeason
    hits_total = _safe_int(h.get("hits"))
    ps = PlayerSeason(
        player_id=str(player_id),
        player_name=pinfo.get("fullName", "Unknown"),
        season=season,
        team=team_abbrev,
        position=position if not (_safe_float(p.get("ip", 0)) > 40 and _safe_int(h.get("plateAppearances", 0)) < 30) else "P",
        # Batting
        pa=_safe_int(h.get("plateAppearances")),
        ab=_safe_int(h.get("atBats")),
        hits=hits_total,
        doubles=_safe_int(h.get("doubles")),
        triples=_safe_int(h.get("triples")),
        hr=_safe_int(h.get("homeRuns")),
        bb=_safe_int(h.get("baseOnBalls")),
        ibb=_safe_int(h.get("intentionalWalks")),
        hbp=_safe_int(h.get("hitByPitch")),
        k=_safe_int(h.get("strikeOuts")),
        sf=_safe_int(h.get("sacFlies")),
        sh=_safe_int(h.get("sacBunts")),
        sb=_safe_int(h.get("stolenBases")),
        cs=_safe_int(h.get("caughtStealing")),
        gidp=_safe_int(h.get("groundIntoDoublePlay")),
        games=_safe_int(h.get("gamesPlayed", p.get("gamesPlayed", 0))),
        # Pitching
        ip=_safe_float(p.get("ip")),
        er=_safe_int(p.get("earnedRuns")),
        hits_allowed=_safe_int(p.get("hits")),
        hr_allowed=_safe_int(p.get("homeRuns")),
        bb_allowed=_safe_int(p.get("baseOnBalls")),
        hbp_allowed=_safe_int(p.get("hitBatsmen")),
        k_pitching=_safe_int(p.get("strikeOuts")),
        games_pitched=_safe_int(p.get("gamesPlayed")),
        games_started=_safe_int(p.get("gamesStarted")),
        saves=_safe_int(p.get("saves")),
        # Context
        park_factor=pf.overall,
        league_rpg=league_rpg,
        season_games=season_length,
    )

    # Use Rust engine if available, else Python
    if USE_RUST:
        rust_r = _rust_compute(
            pa=ps.pa, ab=ps.ab, hits=ps.hits, doubles=ps.doubles, triples=ps.triples,
            hr=ps.hr, bb=ps.bb, ibb=ps.ibb, hbp=ps.hbp, k=ps.k, sf=ps.sf,
            sb=ps.sb, cs=ps.cs, gidp=ps.gidp, games=ps.games,
            ip=ps.ip, er=ps.er, hits_allowed=ps.hits_allowed, hr_allowed=ps.hr_allowed,
            bb_allowed=ps.bb_allowed, hbp_allowed=ps.hbp_allowed, k_pitching=ps.k_pitching,
            games_pitched=ps.games_pitched, games_started=ps.games_started, saves=ps.saves,
            inn_fielded=ps.inn_fielded,
            uzr=ps.uzr if ps.uzr is not None else None,
            drs=ps.drs if ps.drs is not None else None,
            oaa=ps.oaa if ps.oaa is not None else None,
            total_zone=ps.total_zone if ps.total_zone is not None else None,
            framing_runs=ps.framing_runs if ps.framing_runs is not None else None,
            blocking_runs=ps.blocking_runs if ps.blocking_runs is not None else None,
            throwing_runs=ps.throwing_runs if ps.throwing_runs is not None else None,
            catcher_pitches=ps.catcher_pitches,
            avg_leverage_index=ps.avg_leverage_index,
            position=ps.position, season=ps.season, park_factor=ps.park_factor,
            league_rpg=ps.league_rpg, season_games=ps.season_games,
        )
        components = rust_r["components"]
        bravs_val = rust_r["bravs"]
        bravs_era_std = rust_r["bravs_era_std"]
        bravs_war_eq = rust_r["bravs_war_eq"]
        ci90 = (rust_r["ci90_lo"], rust_r["ci90_hi"])
        total_runs = rust_r["total_runs"]
        rpw_val = rust_r["rpw"]
        lev_mult = rust_r["leverage_mult"]
    else:
        result = compute_bravs(ps)
        components = []
        for name, comp in sorted(result.components.items()):
            components.append({
                "name": name,
                "runs": round(comp.runs_mean, 1),
                "ci_lo": round(comp.ci_90[0], 1),
                "ci_hi": round(comp.ci_90[1], 1),
            })
        ci90 = result.bravs_ci_90
        bravs_val = round(result.bravs, 1)
        bravs_era_std = round(result.bravs_era_standardized, 1)
        bravs_war_eq = round(result.bravs_calibrated, 1)
        total_runs = round(result.total_runs_mean, 1)
        rpw_val = round(result.rpw, 2)
        lev_mult = round(result.leverage_multiplier, 3)

    # Build traditional stats
    trad_stats = {}
    if _safe_int(h.get("plateAppearances")) > 0:
        trad_stats["batting"] = {
            "G": _safe_int(h.get("gamesPlayed")),
            "PA": _safe_int(h.get("plateAppearances")),
            "AVG": h.get("avg", ""),
            "OBP": h.get("obp", ""),
            "SLG": h.get("slg", ""),
            "OPS": h.get("ops", ""),
            "HR": _safe_int(h.get("homeRuns")),
            "BB": _safe_int(h.get("baseOnBalls")),
            "K": _safe_int(h.get("strikeOuts")),
            "SB": _safe_int(h.get("stolenBases")),
        }
    if _safe_float(p.get("ip")) > 0:
        trad_stats["pitching"] = {
            "G": _safe_int(p.get("gamesPlayed")),
            "GS": _safe_int(p.get("gamesStarted")),
            "IP": round(_safe_float(p.get("ip")), 1),
            "ERA": p.get("era", ""),
            "WHIP": p.get("whip", ""),
            "K": _safe_int(p.get("strikeOuts")),
            "BB": _safe_int(p.get("baseOnBalls")),
            "W": _safe_int(p.get("wins")),
            "L": _safe_int(p.get("losses")),
            "SV": _safe_int(p.get("saves")),
        }

    return jsonify({
        "player_name": _display_name(ps.player_name),
        "season": season,
        "position": ps.position,
        "team": team_name,
        "team_abbrev": team_abbrev,
        "team_id": team_id,
        "headshot": HEADSHOT_URL.format(pid=player_id),
        "team_logo": TEAM_LOGO_URL.format(tid=team_id) if team_id else None,
        "bats_throws": f"{pinfo.get('batSide', {}).get('code', '?')}/{pinfo.get('pitchHand', {}).get('code', '?')}",
        "number": pinfo.get("primaryNumber", ""),
        "bravs": bravs_val,
        "bravs_era_std": bravs_era_std,
        "bravs_war_eq": bravs_war_eq,
        "ci90_lo": round(ci90[0], 1),
        "ci90_hi": round(ci90[1], 1),
        "total_runs": total_runs,
        "rpw": rpw_val,
        "leverage_mult": lev_mult,
        "park_factor": round(pf.overall, 2),
        "components": components,
        "engine": "rust" if USE_RUST else "python",
        "traditional": trad_stats,
    })


def _fetch_leaders(category: str, season: int, league_id: int, limit: int = 12) -> list[dict]:
    """Fetch league leaders from the MLB Stats API leaders endpoint."""
    data = _mlb_get(
        f"{MLB_API}/stats/leaders",
        {"leaderCategories": category, "season": season,
         "leagueId": league_id, "limit": limit, "statGroup": "hitting"},
    )
    results = []
    if data and "leagueLeaders" in data:
        for group in data["leagueLeaders"]:
            for leader in group.get("leaders", []):
                person = leader.get("person", {})
                pid = person.get("id")
                if pid:
                    results.append({
                        "id": pid,
                        "name": person.get("fullName", "Unknown"),
                    })
    return results


def _fetch_pitching_leaders(category: str, season: int, league_id: int, limit: int = 12) -> list[dict]:
    """Fetch pitching leaders."""
    data = _mlb_get(
        f"{MLB_API}/stats/leaders",
        {"leaderCategories": category, "season": season,
         "leagueId": league_id, "limit": limit, "statGroup": "pitching"},
    )
    results = []
    if data and "leagueLeaders" in data:
        for group in data["leagueLeaders"]:
            for leader in group.get("leaders", []):
                person = leader.get("person", {})
                pid = person.get("id")
                if pid:
                    results.append({
                        "id": pid,
                        "name": person.get("fullName", "Unknown"),
                    })
    return results


@app.route("/api/awards/<award>/<int:season>/<league>")
def award_race(award: str, season: int, league: str):
    """Top BRAVS candidates for an award race, served from precomputed
    GPU output. For 1920-2025 reads bravs_all_seasons.csv; for 2026 reads
    the live 2026/bravs_2026.csv. Falls back to per-player live compute
    for seasons outside the GPU table."""
    lg_up = league.upper()

    # ---- GPU precomputed fast path ----
    if 1920 <= season <= 2026:
        if season == 2026:
            df = _load_bravs_2026()
        else:
            df, _ = _load_precomputed()
        if df is not None and not df.empty:
            d = df[df.yearID == season].copy()
            # League filter: handle both modern (AL/NL) and 2026 (AM/NA) codes
            if lg_up == "AL":
                d = d[d.lgID.isin(["AL", "AM"])]
            elif lg_up == "NL":
                d = d[d.lgID.isin(["NL", "NA"])]
            # Award-specific: Cy Young = pitchers, MVP = position players + 2-way
            if award == "cy_young":
                d = d[d.IP >= 50]
                top = d.nlargest(12, "bravs")
            else:
                d = d[d.PA >= 200]
                top = d.nlargest(12, "bravs")

            # Resolve MLB IDs for headshots/logos
            results = []
            for _, r in top.iterrows():
                rec = _row_to_bravs_dict(r)
                # 2026 already has int MLB IDs; pre-2026 needs Lahman->MLB
                if season == 2026:
                    mlb_pid = int(r.playerID)
                else:
                    _, l2m = _load_crosswalk()
                    mlb_pid = l2m.get(r.playerID, 0) if l2m else 0
                rec["player_id"] = mlb_pid
                if mlb_pid:
                    rec["headshot"] = HEADSHOT_URL.format(pid=mlb_pid)
                _tid = TEAM_IDS.get(rec.get("team_abbrev", ""))
                rec["team_id"] = _tid or 0
                if _tid:
                    rec["team_logo"] = TEAM_LOGO_URL.format(tid=_tid)
                results.append(rec)
            return jsonify({
                "award": award,
                "season": season,
                "league": lg_up,
                "engine": "gpu_precomputed",
                "candidates": results,
            })

    # ---- Live fallback (pre-1920 won't have data; this branch unused) ----
    league_id = 103 if lg_up == "AL" else 104
    seen_ids: set[int] = set()
    candidates: list[dict] = []

    def _add_candidates(leaders: list[dict]) -> None:
        for p in leaders:
            if p["id"] not in seen_ids:
                seen_ids.add(p["id"])
                candidates.append(p)

    if award == "cy_young":
        _add_candidates(_fetch_pitching_leaders("earnedRunAverage", season, league_id, 10))
        _add_candidates(_fetch_pitching_leaders("strikeouts", season, league_id, 8))
        _add_candidates(_fetch_pitching_leaders("inningsPitched", season, league_id, 8))
        _add_candidates(_fetch_pitching_leaders("saves", season, league_id, 5))
    else:
        _add_candidates(_fetch_leaders("onBasePlusSlugging", season, league_id, 12))
        _add_candidates(_fetch_leaders("plateAppearances", season, league_id, 8))
        _add_candidates(_fetch_leaders("homeRuns", season, league_id, 5))

    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(compute_player_bravs_internal, cand["id"], season): cand
            for cand in candidates[:12]
        }
        for future in as_completed(futures):
            cand = futures[future]
            try:
                resp_data = future.result()
                if resp_data:
                    resp_data["player_id"] = cand["id"]
                    results.append(resp_data)
            except Exception as e:
                logger.warning("Failed to compute BRAVS for %s: %s", cand["name"], e)

    results.sort(key=lambda x: x.get("bravs", 0), reverse=True)

    return jsonify({
        "award": award,
        "season": season,
        "league": lg_up,
        "candidates": results,
    })


def compute_player_bravs_internal(player_id: int, season: int) -> dict | None:
    """Internal BRAVS computation returning a dict (shared with award race endpoint)."""
    with app.test_request_context():
        resp = compute_player_bravs(player_id, season)
        if isinstance(resp, tuple):
            return None
        return resp.get_json()


@app.route("/api/compare", methods=["POST"])
def compare_players():
    """Compare multiple player-seasons side by side.

    POST body: {"players": [{"id": 545361, "season": 2016}, ...]}
    """
    data = flask_request.get_json()
    if not data or "players" not in data:
        return jsonify({"error": "Missing 'players' array"}), 400

    results = []
    for entry in data["players"][:6]:
        pid = entry.get("id")
        season = entry.get("season")
        if not pid or not season:
            continue
        try:
            r = compute_player_bravs_internal(int(pid), int(season))
            if r:
                r["player_id"] = pid
                results.append(r)
        except Exception as e:
            logger.warning("Compare failed for %s/%s: %s", pid, season, e)

    return jsonify({"players": results})


@app.route("/api/player/<int:player_id>/career")
def player_career(player_id):
    """Compute career BRAVS across all MLB seasons for a player."""
    # Fetch player info
    player_info = _mlb_get(f"{MLB_API}/people/{player_id}")
    if not player_info or "people" not in player_info:
        return jsonify({"error": "Player not found"}), 404
    pinfo = player_info["people"][0]
    raw_name = pinfo.get("fullFMLName", pinfo.get("fullName", "Unknown"))

    # Try pre-computed career data
    seasons_df, careers_df = _load_precomputed()
    if not careers_df.empty:
        # Search by name since we have MLB ID but CSV has Lahman ID
        name = _display_name(pinfo.get("fullName", ""))
        career_match = careers_df[careers_df.name.str.contains(name.split()[-1], case=False, na=False)]
        if not career_match.empty:
            cr = career_match.iloc[0]
            player_seasons = seasons_df[seasons_df.playerID == cr.playerID].sort_values("yearID", ascending=False)
            seasons_list = []
            for _, s in player_seasons.iterrows():
                seasons_list.append({
                    "season": int(s.yearID), "team": s.get("team", ""),
                    "bravs": round(float(s.bravs), 1), "war_eq": round(float(s.bravs_war_eq), 1),
                    "position": s.get("position", ""),
                })
            return jsonify({
                "player_name": name,
                "player_id": player_id,
                "headshot": HEADSHOT_URL.format(pid=player_id),
                "career_bravs": round(float(cr.career_bravs), 1),
                "career_war_eq": round(float(cr.career_war_eq), 1),
                "seasons": seasons_list,
            })

    # Reuse player_seasons logic to get all MLB seasons
    hitting = _mlb_get(
        f"{MLB_API}/people/{player_id}/stats",
        {"stats": "yearByYear", "group": "hitting", "gameType": "R"},
    )
    pitching = _mlb_get(
        f"{MLB_API}/people/{player_id}/stats",
        {"stats": "yearByYear", "group": "pitching", "gameType": "R"},
    )

    mlb_seasons: dict[int, dict] = {}

    if hitting and "stats" in hitting:
        for group in hitting["stats"]:
            for split in group.get("splits", []):
                yr = int(split.get("season", 0))
                if yr < 1871:
                    continue
                # Skip minor league splits (MLB splits have sport.id == 1)
                sport = split.get("sport", {})
                if sport.get("id") and sport["id"] != 1:
                    continue
                team = split.get("team", {})
                pos = split.get("position", {})
                if yr not in mlb_seasons:
                    mlb_seasons[yr] = {
                        "team_abbrev": _get_team_abbrev(team),
                        "position": pos.get("abbreviation", ""),
                    }

    if pitching and "stats" in pitching:
        for group in pitching["stats"]:
            for split in group.get("splits", []):
                yr = int(split.get("season", 0))
                if yr < 1871:
                    continue
                sport = split.get("sport", {})
                if sport.get("id") and sport["id"] != 1:
                    continue
                team = split.get("team", {})
                if yr not in mlb_seasons:
                    mlb_seasons[yr] = {
                        "team_abbrev": _get_team_abbrev(team),
                        "position": "P",
                    }

    if not mlb_seasons:
        return jsonify({"error": "No MLB seasons found"}), 404

    # Compute BRAVS for each season in parallel
    season_results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(compute_player_bravs_internal, player_id, yr): yr
            for yr in mlb_seasons
        }
        for future in as_completed(futures):
            yr = futures[future]
            try:
                resp_data = future.result()
                if resp_data and "error" not in resp_data:
                    season_results.append({
                        "season": yr,
                        "team": resp_data.get("team_abbrev", mlb_seasons[yr]["team_abbrev"]),
                        "bravs": resp_data.get("bravs", 0),
                        "war_eq": resp_data.get("bravs_war_eq", 0),
                        "position": resp_data.get("position", mlb_seasons[yr]["position"]),
                    })
            except Exception as e:
                logger.warning("Career BRAVS failed for %s/%s: %s", player_id, yr, e)

    # Sort by year descending
    season_results.sort(key=lambda x: x["season"], reverse=True)

    career_bravs = round(sum(s["bravs"] for s in season_results), 1)
    career_war_eq = round(sum(s["war_eq"] for s in season_results), 1)

    return jsonify({
        "player_name": _display_name(raw_name),
        "player_id": player_id,
        "headshot": HEADSHOT_URL.format(pid=player_id),
        "career_bravs": career_bravs,
        "career_war_eq": career_war_eq,
        "seasons": season_results,
    })


@app.route("/api/leaderboard/<int:season>/<league>")
def season_leaderboard(season, league):
    """Season leaderboard: top position players by OPS + top pitchers by ERA."""
    league_id = 103 if league.upper() == "AL" else 104

    seen_ids: set[int] = set()
    candidates: list[dict] = []

    def _add(leaders: list[dict]) -> None:
        for p in leaders:
            if p["id"] not in seen_ids:
                seen_ids.add(p["id"])
                candidates.append(p)

    # Top 20 position players by OPS
    _add(_fetch_leaders("onBasePlusSlugging", season, league_id, 20))
    # Top 10 pitchers by ERA
    _add(_fetch_pitching_leaders("earnedRunAverage", season, league_id, 10))

    # Compute BRAVS for all candidates in parallel
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(compute_player_bravs_internal, cand["id"], season): cand
            for cand in candidates
        }
        for future in as_completed(futures):
            cand = futures[future]
            try:
                resp_data = future.result()
                if resp_data and "error" not in resp_data:
                    results.append({
                        "player_name": resp_data.get("player_name", cand["name"]),
                        "player_id": cand["id"],
                        "bravs": resp_data.get("bravs", 0),
                        "war_eq": resp_data.get("bravs_war_eq", 0),
                        "position": resp_data.get("position", ""),
                        "team": resp_data.get("team_abbrev", ""),
                        "headshot": HEADSHOT_URL.format(pid=cand["id"]),
                    })
            except Exception as e:
                logger.warning("Leaderboard BRAVS failed for %s: %s", cand["name"], e)

    results.sort(key=lambda x: x.get("bravs", 0), reverse=True)

    return jsonify({
        "season": season,
        "league": league.upper(),
        "players": results,
    })


@app.route("/api/export/<int:player_id>/<int:season>")
def export_csv(player_id, season):
    """Export BRAVS breakdown as downloadable CSV."""
    resp_data = compute_player_bravs_internal(player_id, season)
    if not resp_data:
        return jsonify({"error": "Could not compute BRAVS"}), 404

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["Component", "Runs", "CI_Low", "CI_High"])

    # Component rows
    for comp in resp_data.get("components", []):
        writer.writerow([
            comp.get("name", ""),
            comp.get("runs", 0),
            comp.get("ci_lo", 0),
            comp.get("ci_hi", 0),
        ])

    # Blank separator
    writer.writerow([])

    # Metadata rows
    writer.writerow(["BRAVS", resp_data.get("bravs", 0), "", ""])
    writer.writerow(["WAR-eq", resp_data.get("bravs_war_eq", 0), "", ""])
    writer.writerow(["Era-Std", resp_data.get("bravs_era_std", 0), "", ""])
    writer.writerow(["RPW", resp_data.get("rpw", 0), "", ""])

    csv_content = output.getvalue()
    output.close()

    player_name = resp_data.get("player_name", "player").replace(" ", "_")
    filename = f"bravs_{player_name}_{season}.csv"

    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/player/<int:player_id>/<int:season>")
def player_permalink(player_id, season):
    """Shareable deep link for a player-season."""
    return render_template(
        "index.html",
        auto_load_player=player_id,
        auto_load_season=season,
    )


@app.route("/api/whatif")
def what_if():
    """Recompute BRAVS with a different position."""
    player_id = flask_request.args.get("player_id")
    season = flask_request.args.get("season")
    pos_override = flask_request.args.get("position_override", "DH")
    if not player_id or not season:
        return jsonify({"error": "Missing player_id or season"}), 400

    # Get the normal result first
    result = compute_player_bravs_internal(int(player_id), int(season))
    if not result:
        return jsonify({"error": "Could not compute"}), 500

    # Now recompute with different position using Python engine
    # We need to rebuild the PlayerSeason with the new position
    # For simplicity, just adjust the positional component
    from baseball_metric.utils.constants import POS_ADJ, GAMES_PER_SEASON
    old_pos = result.get("position", "DH")
    old_adj = POS_ADJ.get(old_pos, 0.0)
    new_adj = POS_ADJ.get(pos_override, 0.0)

    # Find the positional component and adjust
    pos_diff_per_162 = new_adj - old_adj
    # Estimate games from components or traditional stats
    games = 150  # approximate
    if result.get("traditional", {}).get("batting", {}).get("G"):
        games = result["traditional"]["batting"]["G"]

    pos_diff = pos_diff_per_162 * (games / GAMES_PER_SEASON)
    rpw = result.get("rpw", 5.9)

    new_bravs = result["bravs"] + pos_diff / rpw

    return jsonify({
        "player_name": result["player_name"],
        "position": pos_override,
        "original_position": old_pos,
        "bravs": round(new_bravs, 1),
        "bravs_war_eq": round(new_bravs * 0.57, 1),
        "positional_diff_runs": round(pos_diff, 1),
    })


@app.route("/api/team/<team_abbrev>/<int:season>")
def team_roster(team_abbrev, season):
    """Get all players on a team for a season and rank by BRAVS — served
    from precomputed GPU output (bravs_all_seasons.csv 1920-2025, or
    2026/bravs_2026.csv for the live season)."""
    team_id = TEAM_IDS.get(team_abbrev.upper())

    # ---- GPU precomputed fast path ----
    if 1920 <= season <= 2026:
        if season == 2026:
            df = _load_bravs_2026()
            csv_team = team_abbrev.upper()  # 2026 uses MLB-style abbrevs
        else:
            df, _ = _load_precomputed()
            csv_team = _team_to_lahman(team_abbrev)  # historical uses Retrosheet
        if df is not None and not df.empty:
            d = df[(df.yearID == season) & (df.team == csv_team)]
            if not d.empty:
                results = []
                _, l2m = _load_crosswalk() if season != 2026 else (None, None)
                for _, r in d.sort_values("bravs", ascending=False).iterrows():
                    rec = _row_to_bravs_dict(r)
                    if season == 2026:
                        mlb_pid = int(r.playerID)
                    else:
                        mlb_pid = (l2m or {}).get(r.playerID, 0)
                    rec["player_id"] = mlb_pid
                    if mlb_pid:
                        rec["headshot"] = HEADSHOT_URL.format(pid=mlb_pid)
                    if team_id:
                        rec["team_id"] = team_id
                        rec["team_logo"] = TEAM_LOGO_URL.format(tid=team_id)
                    results.append(rec)
                return jsonify({
                    "team": team_abbrev,
                    "season": season,
                    "engine": "gpu_precomputed",
                    "players": results,
                    "team_bravs": round(sum(r.get("bravs", 0) for r in results), 1),
                })

    # ---- Live fallback (out-of-range seasons) ----
    if not team_id:
        teams_data = _mlb_get(f"{MLB_API}/teams", {"season": season, "sportId": 1})
        if teams_data and "teams" in teams_data:
            for t in teams_data["teams"]:
                if team_abbrev.upper() in (
                    t.get("abbreviation", "").upper(),
                    t.get("teamCode", "").upper(),
                    t.get("shortName", "").upper()[:3],
                ):
                    team_id = t["id"]
                    break
    if not team_id:
        return jsonify({"error": f"Unknown team: {team_abbrev}"}), 404

    roster_data = _mlb_get(
        f"{MLB_API}/teams/{team_id}/roster",
        {"season": season, "rosterType": "fullSeason"},
    )
    if not roster_data or "roster" not in roster_data:
        return jsonify({"error": "Could not fetch roster"}), 500

    roster_players = []
    for entry in roster_data["roster"]:
        person = entry.get("person", {})
        pid = person.get("id")
        name = person.get("fullName", "Unknown")
        pos = entry.get("position", {}).get("abbreviation", "")
        if pid:
            roster_players.append({"id": pid, "name": name, "position": pos})

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(compute_player_bravs_internal, p["id"], season): p
            for p in roster_players
        }
        for future in as_completed(futures):
            p = futures[future]
            try:
                r = future.result()
                if r:
                    r["player_id"] = p["id"]
                    results.append(r)
            except Exception:
                pass

    results.sort(key=lambda x: x.get("bravs", 0), reverse=True)

    return jsonify({
        "team": team_abbrev,
        "season": season,
        "players": results,
        "team_bravs": round(sum(r.get("bravs", 0) for r in results), 1),
    })


@app.route("/api/player/<int:player_id>/projection")
def player_projection(player_id: int):
    """Project future BRAVS using aging curves."""
    from baseball_metric.analysis.projections import project_bravs, remaining_career_value

    # Get player info for age
    pinfo_data = _mlb_get(f"{MLB_API}/people/{player_id}")
    if not pinfo_data or "people" not in pinfo_data:
        return jsonify({"error": "Player not found"}), 404
    pinfo = pinfo_data["people"][0]

    birth = pinfo.get("birthDate", "")
    if not birth:
        return jsonify({"error": "Birth date unknown"}), 400
    birth_year = int(birth[:4])
    current_age = 2026 - birth_year

    is_pitcher = pinfo.get("primaryPosition", {}).get("abbreviation", "") == "P"

    # Get most recent season BRAVS
    result = compute_player_bravs_internal(player_id, 2025)
    if not result:
        result = compute_player_bravs_internal(player_id, 2024)
    if not result:
        return jsonify({"error": "No recent season data"}), 404

    current_bravs = result.get("bravs", 0)

    projections = project_bravs(current_bravs, current_age, is_pitcher, years_forward=8)
    career = remaining_career_value(current_bravs, current_age, is_pitcher)

    return jsonify({
        "player_name": _display_name(pinfo.get("fullName", "Unknown")),
        "current_age": current_age,
        "current_bravs": round(current_bravs, 1),
        "is_pitcher": is_pitcher,
        "projections": projections,
        "remaining_career_bravs": career["remaining_bravs"],
        "remaining_war_eq": career["remaining_war_eq"],
        "expected_years_left": career["expected_years"],
        "expected_retirement_age": career["retirement_age"],
    })


# ---------------------------------------------------------------------------
# Endpoint 1: Player Similarity
# ---------------------------------------------------------------------------

# Reference seasons with known MLB API IDs (skip historical players without API data)
_SIMILARITY_REFS = [
    (545361, 2016, "Mike Trout"),
    # Bonds: no reliable MLB API data — skipped
    (121578, 1927, "Babe Ruth"),
    (121439, 1965, "Willie Mays"),
    (592450, 2022, "Aaron Judge"),
    (660271, 2023, "Shohei Ohtani"),
    # Pedro Martinez: no reliable MLB API data — skipped
    # Bob Gibson: no reliable MLB API data — skipped
    (594798, 2018, "Jacob deGrom"),
    (605141, 2018, "Mookie Betts"),
    (514888, 2017, "Jose Altuve"),
    # Griffey: no reliable MLB API data — skipped
    (477132, 2014, "Clayton Kershaw"),
    (408234, 2012, "Miguel Cabrera"),
    (115749, 1982, "Rickey Henderson"),
]

_SIM_WEIGHTS = {
    "hitting": 3.0,
    "pitching": 3.0,
    "baserunning": 2.0,
    "fielding": 1.5,
    "positional": 1.0,
    "approach_quality": 1.0,
    "durability": 0.5,
    "leverage": 0.5,
}


def _extract_component_runs(bravs_result: dict) -> dict[str, float]:
    """Extract named component runs from a BRAVS result dict."""
    comp_map: dict[str, float] = {}
    for comp in bravs_result.get("components", []):
        comp_map[comp["name"]] = comp.get("runs", 0.0)
    return comp_map


def _similarity_score(comp1: dict[str, float], comp2: dict[str, float]) -> float:
    """Weighted Euclidean distance converted to 0-100 similarity."""
    total = 0.0
    for name, weight in _SIM_WEIGHTS.items():
        c1 = comp1.get(name, 0.0)
        c2 = comp2.get(name, 0.0)
        total += weight * (c1 - c2) ** 2
    dist = total ** 0.5
    return max(0.0, 100.0 - dist * 1.2)


@app.route("/api/player/<int:player_id>/similar/<int:season>")
def player_similar(player_id, season):
    """Find the most similar historical player-seasons by BRAVS component profile."""
    target = compute_player_bravs_internal(player_id, season)
    if not target:
        return jsonify({"error": "Could not compute BRAVS for target"}), 404
    target_comps = _extract_component_runs(target)

    # Compute reference seasons in parallel
    ref_results: list[tuple[str, int, int, dict | None]] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(compute_player_bravs_internal, ref_id, ref_yr): (ref_name, ref_id, ref_yr)
            for ref_id, ref_yr, ref_name in _SIMILARITY_REFS
            if not (ref_id == player_id and ref_yr == season)
        }
        for future in as_completed(futures):
            ref_name, ref_id, ref_yr = futures[future]
            try:
                result = future.result()
                ref_results.append((ref_name, ref_id, ref_yr, result))
            except Exception as e:
                logger.warning("Similarity ref failed %s %s: %s", ref_name, ref_yr, e)

    # Score each reference
    scored: list[dict] = []
    for ref_name, ref_id, ref_yr, ref_data in ref_results:
        if not ref_data:
            continue
        ref_comps = _extract_component_runs(ref_data)
        sim = _similarity_score(target_comps, ref_comps)
        scored.append({
            "player_name": ref_name,
            "player_id": ref_id,
            "season": ref_yr,
            "similarity": round(sim, 1),
            "bravs": ref_data.get("bravs", 0),
            "position": ref_data.get("position", ""),
            "headshot": HEADSHOT_URL.format(pid=ref_id),
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)

    return jsonify({
        "player_name": target.get("player_name", ""),
        "target": {
            "player_name": target.get("player_name", ""),
            "season": season,
            "bravs": target.get("bravs", 0),
        },
        "similar": scored[:5],
    })


# ---------------------------------------------------------------------------
# Endpoint 2: Live MVP Race (current season)
# ---------------------------------------------------------------------------

@app.route("/api/mvp/live/<league>")
def live_mvp(league):
    """Live MVP race for the current season — served from the daily 2026
    GPU CSV (data/2026/bravs_2026.csv). The assembler runs at 2 AM and
    refreshes BRAVS for every active player; we just rank and return."""
    df = _load_bravs_2026()
    if df is None or df.empty:
        return jsonify({
            "season": 2026,
            "league": league.upper(),
            "candidates": [],
            "engine": "gpu_2026",
            "error": "2026 BRAVS table not available",
        }), 503

    lg_up = league.upper()
    d = df[df.yearID == 2026].copy()
    if lg_up == "AL":
        d = d[d.lgID.isin(["AL", "AM"])]
    elif lg_up == "NL":
        d = d[d.lgID.isin(["NL", "NA"])]
    # Need at least token playing time so the leaderboard isn't all noise
    d = d[(d.PA >= 20) | (d.IP >= 5)]

    # Estimate league-wide team_games from max G in the table
    team_games_est = int(d.G.max()) if not d.empty else 0

    candidates = []
    for _, r in d.nlargest(20, "bravs").iterrows():
        rec = _row_to_bravs_dict(r)
        mlb_pid = int(r.playerID)  # 2026 CSV stores MLB ids natively
        rec["player_id"] = mlb_pid
        rec["headshot"] = HEADSHOT_URL.format(pid=mlb_pid)
        _tid = TEAM_IDS.get(rec.get("team_abbrev", ""))
        rec["team_id"] = _tid or 0
        if _tid:
            rec["team_logo"] = TEAM_LOGO_URL.format(tid=_tid)

        # Extrapolate to a 162-game pace for the headline number
        games = int(r.G) if pd.notna(r.G) else 0
        bravs_now = float(r.bravs)
        pace_factor = (162.0 / games) if games > 0 else 1.0
        pace_factor = min(pace_factor, 20.0)
        rec["games_played"] = games
        rec["team_games"] = team_games_est
        rec["bravs_raw"] = bravs_now
        rec["bravs_pace"] = round(bravs_now * pace_factor, 1)
        rec["bravs_war_eq"] = round(rec["bravs_pace"] * 0.57, 1)
        candidates.append(rec)

    candidates.sort(key=lambda x: x.get("bravs_pace", 0), reverse=True)

    return jsonify({
        "season": 2026,
        "league": lg_up,
        "team_games": team_games_est,
        "engine": "gpu_2026",
        "candidates": candidates,
    })


# ---------------------------------------------------------------------------
# Endpoint 3: Dynasty Rankings
# ---------------------------------------------------------------------------

@app.route("/api/dynasty/<int:player_id>")
def player_dynasty(player_id):
    """Find the best 5-consecutive-year dynasty window for a player."""
    # Reuse career endpoint logic to get all seasons
    player_info = _mlb_get(f"{MLB_API}/people/{player_id}")
    if not player_info or "people" not in player_info:
        return jsonify({"error": "Player not found"}), 404
    pinfo = player_info["people"][0]
    raw_name = pinfo.get("fullFMLName", pinfo.get("fullName", "Unknown"))

    # Try pre-computed dynasty data
    seasons_df, careers_df = _load_precomputed()
    if not seasons_df.empty:
        name = _display_name(pinfo.get("fullName", ""))
        name_match = seasons_df[seasons_df.name.str.contains(name.split()[-1], case=False, na=False)]
        if not name_match.empty:
            pid_lahman = name_match.iloc[0].playerID
            player_seasons = seasons_df[seasons_df.playerID == pid_lahman].sort_values("yearID")
            all_seasons = []
            season_map: dict[int, dict] = {}
            for _, s in player_seasons.iterrows():
                entry = {
                    "season": int(s.yearID),
                    "team": s.get("team", ""),
                    "bravs": round(float(s.bravs), 1),
                    "war_eq": round(float(s.bravs_war_eq), 1),
                    "position": s.get("position", ""),
                }
                all_seasons.append(entry)
                season_map[int(s.yearID)] = entry

            # Find the best 5-consecutive-year window
            best_total = -999.0
            best_start = 0
            best_end = 0
            best_window: list[dict] = []
            sorted_years = sorted(season_map.keys())
            for i in range(len(sorted_years)):
                window_years = [y for y in sorted_years if sorted_years[i] <= y < sorted_years[i] + 5]
                if len(window_years) < 2:
                    continue
                window_bravs = sum(season_map[y]["bravs"] for y in window_years)
                if window_bravs > best_total:
                    best_total = window_bravs
                    best_start = window_years[0]
                    best_end = window_years[-1]
                    best_window = [season_map[y] for y in window_years]

            dynasty_years = len(best_window) if best_window else 0
            dynasty_avg = round(best_total / dynasty_years, 1) if dynasty_years else 0.0

            return jsonify({
                "player_name": _display_name(raw_name),
                "player_id": player_id,
                "headshot": HEADSHOT_URL.format(pid=player_id),
                "best_window": {
                    "total_bravs": round(best_total, 1),
                    "avg_per_year": dynasty_avg,
                    "start": best_start,
                    "end": best_end,
                    "seasons": best_window,
                },
                "seasons": sorted(all_seasons, key=lambda x: x["season"]),
                "career_bravs": round(sum(s["bravs"] for s in all_seasons), 1),
            })

    # Get all MLB seasons via yearByYear
    hitting = _mlb_get(
        f"{MLB_API}/people/{player_id}/stats",
        {"stats": "yearByYear", "group": "hitting", "gameType": "R"},
    )
    pitching = _mlb_get(
        f"{MLB_API}/people/{player_id}/stats",
        {"stats": "yearByYear", "group": "pitching", "gameType": "R"},
    )

    mlb_years: set[int] = set()
    if hitting and "stats" in hitting:
        for group in hitting["stats"]:
            for split in group.get("splits", []):
                yr = int(split.get("season", 0))
                sport = split.get("sport", {})
                if yr >= 1871:
                    mlb_years.add(yr)
    if pitching and "stats" in pitching:
        for group in pitching["stats"]:
            for split in group.get("splits", []):
                yr = int(split.get("season", 0))
                if yr >= 1871:
                    mlb_years.add(yr)

    if not mlb_years:
        return jsonify({"error": "No MLB seasons found"}), 404

    # Compute BRAVS for each season in parallel
    season_map: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(compute_player_bravs_internal, player_id, yr): yr
            for yr in mlb_years
        }
        for future in as_completed(futures):
            yr = futures[future]
            try:
                resp_data = future.result()
                if resp_data and "error" not in resp_data:
                    season_map[yr] = {
                        "season": yr,
                        "team": resp_data.get("team_abbrev", ""),
                        "bravs": resp_data.get("bravs", 0),
                        "war_eq": resp_data.get("bravs_war_eq", 0),
                        "position": resp_data.get("position", ""),
                    }
            except Exception as e:
                logger.warning("Dynasty BRAVS failed for %s/%s: %s", player_id, yr, e)

    all_seasons = sorted(season_map.values(), key=lambda x: x["season"])

    # Find the best 5-consecutive-year window
    best_total = -999.0
    best_start = 0
    best_end = 0
    best_window: list[dict] = []

    sorted_years = sorted(season_map.keys())
    for i in range(len(sorted_years)):
        window_years = [y for y in sorted_years if sorted_years[i] <= y < sorted_years[i] + 5]
        if len(window_years) < 2:
            continue
        window_bravs = sum(season_map[y]["bravs"] for y in window_years)
        if window_bravs > best_total:
            best_total = window_bravs
            best_start = window_years[0]
            best_end = window_years[-1]
            best_window = [season_map[y] for y in window_years]

    dynasty_years = len(best_window) if best_window else 0
    dynasty_avg = round(best_total / dynasty_years, 1) if dynasty_years else 0.0

    return jsonify({
        "player_name": _display_name(raw_name),
        "player_id": player_id,
        "headshot": HEADSHOT_URL.format(pid=player_id),
        "best_window": {
            "total_bravs": round(best_total, 1),
            "avg_per_year": dynasty_avg,
            "start": best_start,
            "end": best_end,
            "seasons": best_window,
        },
        "seasons": all_seasons,
        "career_bravs": round(sum(s["bravs"] for s in all_seasons), 1),
    })


# ---------------------------------------------------------------------------
# Endpoint 4: Dream Team
# ---------------------------------------------------------------------------

@app.route("/api/dreamteam")
def dream_team():
    """Return the all-time BRAVS dream team with pre-computed values."""
    roster = [
        {"slot": "C",   "name": "Mike Piazza",      "season": 1997, "bravs": 17.1, "war_eq": 10.6,
         "signature": ".362/.431/.638, 40 HR from catcher"},
        {"slot": "1B",  "name": "Lou Gehrig",        "season": 1927, "bravs": 14.5, "war_eq": 9.0,
         "signature": ".373/.474/.765, 218 H / 47 HR"},
        {"slot": "2B",  "name": "Rogers Hornsby",    "season": 1922, "bravs": 22.2, "war_eq": 13.7,
         "signature": ".401/.459/.722, 250 hits (!)"},
        {"slot": "3B",  "name": "Mike Schmidt",      "season": 1980, "bravs": 14.4, "war_eq": 8.9,
         "signature": ".286/.380/.624, 48 HR, Gold Glove D"},
        {"slot": "SS",  "name": "Honus Wagner",      "season": 1908, "bravs": 19.6, "war_eq": 12.1,
         "signature": ".354/.413/.542, 53 SB, elite glove"},
        {"slot": "LF",  "name": "Ted Williams",      "season": 1941, "bravs": 18.2, "war_eq": 11.3,
         "signature": ".406/.553/.735, last man to hit .400"},
        {"slot": "CF",  "name": "Willie Mays",       "season": 1965, "bravs": 26.1, "war_eq": 16.2,
         "signature": ".317/.398/.645, 52 HR, elite defense"},
        {"slot": "RF",  "name": "Babe Ruth",         "season": 1921, "bravs": 22.0, "war_eq": 13.6,
         "signature": ".378/.512/.846, 59 HR, 204 hits"},
        {"slot": "DH",  "name": "Barry Bonds",       "season": 2001, "bravs": 25.0, "war_eq": 15.5,
         "signature": ".328/.515/.863, 73 HR, 177 BB"},
        {"slot": "SP1", "name": "Pedro Martinez",    "season": 2000, "bravs": 11.3, "war_eq": 7.0,
         "signature": "1.74 ERA, 284 K in 217 IP, steroid-era"},
        {"slot": "SP2", "name": "Bob Gibson",        "season": 1968, "bravs": 28.1, "war_eq": 17.5,
         "signature": "1.12 ERA, 268 K in 305 IP, Year of Pitcher"},
        {"slot": "SP3", "name": "Sandy Koufax",      "season": 1966, "bravs": 24.5, "war_eq": 15.2,
         "signature": "1.73 ERA, 317 K in 323 IP, final season"},
        {"slot": "SP4", "name": "Randy Johnson",     "season": 2001, "bravs": 14.3, "war_eq": 8.9,
         "signature": "2.49 ERA, 372 K in 250 IP, 6'10 unit"},
        {"slot": "SP5", "name": "Greg Maddux",       "season": 1995, "bravs": 11.1, "war_eq": 6.9,
         "signature": "1.63 ERA, 23 BB in 210 IP, Picasso"},
        {"slot": "CL",  "name": "Mariano Rivera",    "season": 2004, "bravs": 4.9,  "war_eq": 3.1,
         "signature": "1.94 ERA, 53 SV, gmLI 1.85, cutter god"},
    ]

    total_bravs = round(sum(p["bravs"] for p in roster), 1)
    total_war_eq = round(sum(p["war_eq"] for p in roster), 1)

    lineup_bravs = round(sum(p["bravs"] for p in roster if not p["slot"].startswith("SP") and p["slot"] != "CL"), 1)
    rotation_bravs = round(sum(p["bravs"] for p in roster if p["slot"].startswith("SP")), 1)
    closer_bravs = round(sum(p["bravs"] for p in roster if p["slot"] == "CL"), 1)

    return jsonify({
        "roster": roster,
        "total_bravs": total_bravs,
        "total_war_eq": total_war_eq,
        "breakdown": {
            "lineup_bravs": lineup_bravs,
            "rotation_bravs": rotation_bravs,
            "closer_bravs": closer_bravs,
        },
    })


# ═══════════════════════════════════════════════════════════════════
#  LINEUP OPTIMIZER ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/lineup/optimize", methods=["POST"])
def api_lineup_optimize():
    """Optimize a lineup for a given team-season.

    POST body: {"team": "NYA", "year": 2024, "pitcher_hand": "R"}
    Returns: optimal lineup with batting order, positions, expected value.
    """
    try:
        from baseball_metric.lineup_optimizer.optimizer import optimize_lineup, select_starters
        from baseball_metric.lineup_optimizer.season_optimizer import compute_positional_surplus

        data = flask_request.get_json(force=True)
        team = data.get("team", "NYA")
        year = int(data.get("year", 2024))
        pitcher_hand = data.get("pitcher_hand", "R")

        seasons, _ = _load_precomputed()
        team_data = seasons[(seasons.yearID == year) & (seasons.team == team) & (seasons.PA >= 50)]

        if len(team_data) < 9:
            return jsonify({"error": f"Not enough players for {team} {year} (found {len(team_data)})"}), 400

        roster = []
        for _, r in team_data.iterrows():
            roster.append({
                "name": r.get("name", "?"),
                "playerID": r.playerID,
                "position": r.position,
                "hitting_runs": float(r.hitting_runs),
                "baserunning_runs": float(r.baserunning_runs),
                "fielding_runs": float(r.fielding_runs),
                "positional_runs": float(r.get("positional_runs", 0)),
                "aqi_runs": float(r.get("aqi_runs", 0)),
                "HR": int(r.HR),
                "SB": int(r.SB),
                "PA": int(r.PA),
                "G": int(r.G),
                "bravs_war_eq": float(r.bravs_war_eq),
            })

        # Run optimizer
        results = optimize_lineup(
            roster,
            opposing_pitcher={"hand": pitcher_hand},
            n_candidates=30000,
            top_n=5,
        )

        if not results:
            return jsonify({"error": "Optimization failed"}), 500

        # Positional surplus
        surplus = compute_positional_surplus(roster)

        # Format results
        lineups = []
        for i, cfg in enumerate(results):
            players = []
            for slot, (player, pos) in enumerate(zip(cfg.players, cfg.positions)):
                players.append({
                    "slot": slot + 1,
                    "name": player.get("name", "?"),
                    "position": pos,
                    "hitting_runs": round(player.get("hitting_runs", 0), 1),
                    "baserunning_runs": round(player.get("baserunning_runs", 0), 1),
                    "fielding_runs": round(player.get("fielding_runs", 0), 1),
                    "bravs_war_eq": round(player.get("bravs_war_eq", 0), 1),
                })
            lineups.append({
                "rank": i + 1,
                "expected_value": round(cfg.expected_runs, 1),
                "uncertainty": round(cfg.expected_runs_std, 1),
                "players": players,
            })

        # Roster summary
        roster_sorted = sorted(roster, key=lambda x: x.get("bravs_war_eq", 0), reverse=True)

        return jsonify({
            "team": team,
            "year": year,
            "pitcher_hand": pitcher_hand,
            "roster_size": len(roster),
            "lineups": lineups,
            "positional_surplus": {k: round(v, 1) for k, v in surplus.items()},
            "full_roster": [{
                "name": p["name"],
                "position": p["position"],
                "bravs_war_eq": round(p["bravs_war_eq"], 1),
                "hitting_runs": round(p["hitting_runs"], 1),
                "PA": p["PA"],
            } for p in roster_sorted[:20]],
        })

    except Exception as e:
        logger.exception("Lineup optimize error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/lineup/trade", methods=["POST"])
def api_lineup_trade():
    """Simulate a trade's impact on lineup optimization.

    POST body: {
        "team": "NYA", "year": 2024,
        "players_out": ["playerID1"],
        "players_in": [{"name": "...", "playerID": "...", "position": "SS",
                        "hitting_runs": 25, "bravs_war_eq": 5.0, ...}]
    }
    """
    try:
        from baseball_metric.lineup_optimizer.trade_impact import simulate_trade

        data = flask_request.get_json(force=True)
        team = data.get("team", "NYA")
        year = int(data.get("year", 2024))
        out_ids = set(data.get("players_out", []))
        players_in = data.get("players_in", [])

        seasons, _ = _load_precomputed()
        team_data = seasons[(seasons.yearID == year) & (seasons.team == team) & (seasons.PA >= 50)]

        roster = []
        for _, r in team_data.iterrows():
            roster.append({
                "name": r.get("name", "?"),
                "playerID": r.playerID,
                "position": r.position,
                "hitting_runs": float(r.hitting_runs),
                "baserunning_runs": float(r.baserunning_runs),
                "fielding_runs": float(r.fielding_runs),
                "positional_runs": float(r.get("positional_runs", 0)),
                "aqi_runs": float(r.get("aqi_runs", 0)),
                "HR": int(r.HR),
                "SB": int(r.SB),
                "PA": int(r.PA),
                "bravs_war_eq": float(r.bravs_war_eq),
            })

        players_out = [p for p in roster if p["playerID"] in out_ids]

        result = simulate_trade(roster, players_out, players_in)

        return jsonify({
            "team": team,
            "year": year,
            "before_value": result["before_value"],
            "after_value": result["after_value"],
            "marginal_impact": result["marginal_impact"],
            "positional_changes": result["positional_changes"],
            "explanation": result["explanation"],
        })

    except Exception as e:
        logger.exception("Trade simulation error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/lineup/season", methods=["POST"])
def api_lineup_season():
    """Season-long roster optimization.

    POST body: {"team": "NYA", "year": 2024}
    Returns: per-player game allocation, projected wins, recommendations.
    """
    try:
        from baseball_metric.lineup_optimizer.season_optimizer import optimize_season

        data = flask_request.get_json(force=True)
        team = data.get("team", "NYA")
        year = int(data.get("year", 2024))

        seasons, _ = _load_precomputed()
        team_data = seasons[(seasons.yearID == year) & (seasons.team == team) & (seasons.PA >= 50)]

        roster = []
        for _, r in team_data.iterrows():
            roster.append({
                "name": r.get("name", "?"),
                "playerID": r.playerID,
                "position": r.position,
                "hitting_runs": float(r.hitting_runs),
                "baserunning_runs": float(r.baserunning_runs),
                "fielding_runs": float(r.fielding_runs),
                "positional_runs": float(r.get("positional_runs", 0)),
                "aqi_runs": float(r.get("aqi_runs", 0)),
                "HR": int(r.HR),
                "SB": int(r.SB),
                "PA": int(r.PA),
                "G": int(r.G),
                "bravs_war_eq": float(r.bravs_war_eq),
            })

        result = optimize_season(roster)

        return jsonify({
            "team": team,
            "year": year,
            "roster_size": len(roster),
            "total_war_eq": result["total_war_eq"],
            "expected_wins": result["expected_wins"],
            "flex_value": result["flex_value"],
            "allocations": result["allocations"],
            "recommendations": result["recommendations"],
        })

    except Exception as e:
        logger.exception("Season optimizer error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/lineup/teams/<int:year>")
def api_lineup_teams(year):
    """Get available teams for a given year."""
    try:
        seasons, _ = _load_precomputed()
        teams_data = seasons[seasons.yearID == year]
        teams = sorted(teams_data.team.dropna().unique().tolist())
        return jsonify({"year": year, "teams": teams})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# MiLB endpoints — lazy-loaded data
# ---------------------------------------------------------------------------
_MILB_SEASONS = None
_PROSPECT_RANKINGS = None


def _load_milb_seasons():
    global _MILB_SEASONS
    if _MILB_SEASONS is None:
        try:
            _MILB_SEASONS = pd.read_csv("data/bravs_milb_seasons.csv")
            logger.info("Loaded MiLB seasons: %d records", len(_MILB_SEASONS))
        except Exception as e:
            logger.warning("Could not load MiLB seasons: %s", e)
            _MILB_SEASONS = pd.DataFrame()
    return _MILB_SEASONS


def _load_prospect_rankings():
    global _PROSPECT_RANKINGS
    if _PROSPECT_RANKINGS is None:
        try:
            _PROSPECT_RANKINGS = pd.read_csv("data/prospect_rankings.csv")
            logger.info("Loaded prospect rankings: %d records", len(_PROSPECT_RANKINGS))
        except Exception as e:
            logger.warning("Could not load prospect rankings: %s", e)
            _PROSPECT_RANKINGS = pd.DataFrame()
    return _PROSPECT_RANKINGS


@app.route("/api/milb/player/<player_id>")
def milb_player(player_id):
    """Look up a player's MiLB career from bravs_milb_seasons.csv."""
    df = _load_milb_seasons()
    if df.empty:
        return jsonify({"error": "MiLB data not available"}), 500

    # Try matching as numeric ID or string
    try:
        pid_num = float(player_id)
        rows = df[df["playerID"] == pid_num]
    except (ValueError, TypeError):
        rows = df[df["playerID"].astype(str) == player_id]

    if rows.empty:
        # Try name match
        rows = df[df["name"].str.contains(str(player_id), case=False, na=False)]

    if rows.empty:
        return jsonify({"error": f"No MiLB data found for {player_id}"}), 404

    seasons = []
    for _, r in rows.sort_values("yearID", ascending=False).iterrows():
        season = {
            "playerID": str(r.get("playerID", "")),
            "yearID": int(r["yearID"]),
            "name": str(r.get("name", "")),
            "team": str(r.get("team", "")),
            "team_name": str(r.get("team_name", "")),
            "lgID": str(r.get("lgID", "")),
            "level": str(r.get("level", "")),
            "position": str(r.get("position", "")),
            "G": int(r.get("G", 0)),
            "bravs_war_eq": round(float(r.get("bravs_war_eq", 0)), 2),
            "translation_rate": round(float(r.get("translation_rate", 0)), 2),
            "PA": int(r.get("PA", 0)) if pd.notna(r.get("PA")) else 0,
            "HR": int(r.get("HR", 0)) if pd.notna(r.get("HR")) else 0,
            "SB": int(r.get("SB", 0)) if pd.notna(r.get("SB")) else 0,
            "wOBA": round(float(r.get("wOBA", 0)), 3) if pd.notna(r.get("wOBA")) else None,
            "hitting_runs": round(float(r.get("hitting_runs", 0)), 1) if pd.notna(r.get("hitting_runs")) else 0,
            "IP": round(float(r.get("IP", 0)), 1) if pd.notna(r.get("IP")) else 0,
            "ERA": round(float(r.get("ERA", 0)), 2) if pd.notna(r.get("ERA")) else None,
            "pitching_runs": round(float(r.get("pitching_runs", 0)), 1) if pd.notna(r.get("pitching_runs")) else 0,
        }
        seasons.append(season)

    player_name = seasons[0]["name"] if seasons else player_id
    return jsonify({
        "player_id": player_id,
        "player_name": player_name,
        "total_seasons": len(seasons),
        "seasons": seasons,
    })


@app.route("/api/milb/leaderboard/<int:year>/<level>")
def milb_leaderboard(year, level):
    """Return top MiLB players for a given year and level sorted by bravs_war_eq."""
    df = _load_milb_seasons()
    if df.empty:
        return jsonify({"error": "MiLB data not available"}), 500

    # Filter by year and level (case-insensitive level match)
    filtered = df[(df["yearID"] == year) & (df["level"].str.upper() == level.upper())]

    if filtered.empty:
        return jsonify({"error": f"No data for {year} {level}"}), 404

    # Sort by bravs_war_eq descending, top 50
    top = filtered.nlargest(50, "bravs_war_eq")

    players = []
    for _, r in top.iterrows():
        players.append({
            "playerID": str(r.get("playerID", "")),
            "name": str(r.get("name", "")),
            "team": str(r.get("team", "")),
            "team_name": str(r.get("team_name", "")),
            "position": str(r.get("position", "")),
            "G": int(r.get("G", 0)),
            "bravs_war_eq": round(float(r.get("bravs_war_eq", 0)), 2),
            "PA": int(r.get("PA", 0)) if pd.notna(r.get("PA")) else 0,
            "HR": int(r.get("HR", 0)) if pd.notna(r.get("HR")) else 0,
            "wOBA": round(float(r.get("wOBA", 0)), 3) if pd.notna(r.get("wOBA")) else None,
            "IP": round(float(r.get("IP", 0)), 1) if pd.notna(r.get("IP")) else 0,
            "ERA": round(float(r.get("ERA", 0)), 2) if pd.notna(r.get("ERA")) else None,
        })

    return jsonify({
        "year": year,
        "level": level.upper(),
        "count": len(players),
        "players": players,
    })


@app.route("/api/milb/prospects")
def milb_prospects():
    """Return top 50 current prospects from prospect_rankings.csv."""
    df = _load_prospect_rankings()
    if df.empty:
        return jsonify({"error": "Prospect rankings not available"}), 500

    # Filter to non-MLB players, sort by projected_mlb_war descending
    prospects = df.copy()
    if "in_mlb" in prospects.columns:
        prospects = prospects[prospects["in_mlb"] == False]

    if "projected_mlb_war" in prospects.columns:
        prospects = prospects.nlargest(50, "projected_mlb_war")
    else:
        prospects = prospects.head(50)

    result = []
    for rank, (_, r) in enumerate(prospects.iterrows(), 1):
        result.append({
            "rank": rank,
            "playerID": str(r.get("playerID", "")),
            "name": str(r.get("name", "")),
            "highest_level": str(r.get("highest_level", r.get("milb_highest_level", ""))),
            "milb_seasons": int(r.get("milb_seasons", 0)) if pd.notna(r.get("milb_seasons")) else 0,
            "milb_total_war": round(float(r.get("milb_total_war", 0)), 2) if pd.notna(r.get("milb_total_war")) else 0,
            "milb_peak_war": round(float(r.get("milb_peak_war", 0)), 2) if pd.notna(r.get("milb_peak_war")) else 0,
            "milb_avg_woba": round(float(r.get("milb_avg_woba", 0)), 3) if pd.notna(r.get("milb_avg_woba")) else None,
            "projected_mlb_war": round(float(r.get("projected_mlb_war", 0)), 1) if pd.notna(r.get("projected_mlb_war")) else None,
            "years_in_minors": int(r.get("years_in_minors", 0)) if pd.notna(r.get("years_in_minors")) else 0,
        })

    return jsonify({
        "count": len(result),
        "prospects": result,
    })


# ═══════════════════════════════════════════════════════════════════
#  VIDEO ENDPOINTS — MLB highlight clips from Stats API
# ══════════════════════════════════════════════════════════════════���

@app.route("/api/video/player/<int:player_id>/<int:season>")
def api_video_player(player_id, season):
    """Get highlight videos for a player's season.

    Fetches game log, then pulls highlight clips from each game's
    content feed. Returns video URLs (MP4 + HLS), thumbnails, titles.
    """
    try:
        # Get player's game log for the season
        gamelog_url = f"{MLB_API}/people/{player_id}/stats"
        r = requests.get(gamelog_url, params={
            "stats": "gameLog", "season": season, "group": "hitting,pitching"
        }, timeout=10)
        data = r.json()

        game_pks = []
        for sg in data.get("stats", []):
            for split in sg.get("splits", []):
                gp = split.get("game", {}).get("gamePk")
                if gp and gp not in game_pks:
                    game_pks.append(gp)

        # Pull highlights from each game (limit to 10 most recent)
        videos = []
        for gp in game_pks[-10:]:
            try:
                r2 = requests.get(f"{MLB_API}/game/{gp}/content", timeout=8)
                items = r2.json().get("highlights", {}).get("highlights", {}).get("items", [])
                for item in items:
                    # Get video URLs
                    mp4 = ""
                    hls = ""
                    for pb in item.get("playbacks", []):
                        if pb.get("name") == "HTTP_CLOUD_WIRED_60":
                            mp4 = pb["url"]
                        elif pb.get("name") == "hlsCloud":
                            hls = pb["url"]

                    # Get thumbnail
                    thumb = ""
                    cuts = item.get("image", {}).get("cuts", [])
                    if isinstance(cuts, dict):
                        thumb = (cuts.get("640x360", {}).get("src", "")
                                or cuts.get("960x540", {}).get("src", ""))
                    elif isinstance(cuts, list):
                        for cut in cuts:
                            if cut.get("width", 0) == 640:
                                thumb = cut.get("src", "")
                                break

                    if mp4 or hls:
                        videos.append({
                            "gamePk": gp,
                            "title": item.get("title", ""),
                            "description": item.get("description", ""),
                            "duration": item.get("duration", ""),
                            "mp4": mp4,
                            "hls": hls,
                            "thumbnail": thumb,
                            "mediaId": item.get("mediaPlaybackId", ""),
                        })
            except Exception:
                pass

        return jsonify({
            "player_id": player_id,
            "season": season,
            "games": len(game_pks),
            "videos": videos,
        })
    except Exception as e:
        logger.exception("Video player error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/game/<int:game_pk>")
def api_video_game(game_pk):
    """Get all highlight videos for a specific game."""
    try:
        r = requests.get(f"{MLB_API}/game/{game_pk}/content", timeout=10)
        items = r.json().get("highlights", {}).get("highlights", {}).get("items", [])

        videos = []
        for item in items:
            mp4 = ""
            hls = ""
            for pb in item.get("playbacks", []):
                if pb.get("name") == "HTTP_CLOUD_WIRED_60":
                    mp4 = pb["url"]
                elif pb.get("name") == "hlsCloud":
                    hls = pb["url"]

            thumb = ""
            cuts = item.get("image", {}).get("cuts", [])
            if isinstance(cuts, dict):
                thumb = (cuts.get("640x360", {}).get("src", "")
                        or cuts.get("960x540", {}).get("src", ""))
            elif isinstance(cuts, list):
                for cut in cuts:
                    if cut.get("width", 0) == 640:
                        thumb = cut.get("src", "")
                        break

            if mp4 or hls:
                videos.append({
                    "gamePk": game_pk,
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "duration": item.get("duration", ""),
                    "mp4": mp4,
                    "hls": hls,
                    "thumbnail": thumb,
                })

        return jsonify({"game_pk": game_pk, "videos": videos})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/search/<query>")
def api_video_search(query):
    """Search for highlight videos by player name or keyword.

    Searches recent games for clips mentioning the query term.
    """
    try:
        # Find the player first
        r = requests.get(f"{MLB_API}/people/search", params={
            "names": query, "sportIds": "1"
        }, timeout=10)
        people = r.json().get("people", [])

        if not people:
            return jsonify({"error": "Player not found", "videos": []})

        player = people[0]
        pid = player["id"]
        name = player["fullName"]

        # Get their recent games
        import datetime
        today = datetime.date.today()
        start = today - datetime.timedelta(days=30)

        schedule = requests.get(f"{MLB_API}/schedule", params={
            "sportId": 1, "startDate": start.isoformat(),
            "endDate": today.isoformat(),
            "gameType": "R",
        }, timeout=10).json()

        # Find games this player appeared in
        game_pks = []
        gamelog = requests.get(f"{MLB_API}/people/{pid}/stats", params={
            "stats": "gameLog", "season": today.year,
            "group": "hitting,pitching"
        }, timeout=10).json()
        for sg in gamelog.get("stats", []):
            for split in sg.get("splits", []):
                gp = split.get("game", {}).get("gamePk")
                if gp:
                    game_pks.append(gp)

        # Get highlights mentioning the player
        videos = []
        for gp in game_pks[-5:]:
            try:
                r2 = requests.get(f"{MLB_API}/game/{gp}/content", timeout=8)
                items = r2.json().get("highlights", {}).get("highlights", {}).get("items", [])
                for item in items:
                    title = item.get("title", "")
                    desc = item.get("description", "")
                    # Filter to clips mentioning the player
                    last_name = name.split()[-1]
                    if last_name.lower() not in (title + desc).lower():
                        continue

                    mp4 = ""
                    hls = ""
                    for pb in item.get("playbacks", []):
                        if pb.get("name") == "HTTP_CLOUD_WIRED_60":
                            mp4 = pb["url"]
                        elif pb.get("name") == "hlsCloud":
                            hls = pb["url"]

                    thumb = ""
                    cuts = item.get("image", {}).get("cuts", [])
                    if isinstance(cuts, dict):
                        thumb = cuts.get("640x360", {}).get("src", "")
                    elif isinstance(cuts, list):
                        for cut in cuts:
                            if cut.get("width", 0) == 640:
                                thumb = cut.get("src", "")
                                break

                    if mp4 or hls:
                        videos.append({
                            "gamePk": gp,
                            "title": title,
                            "description": desc,
                            "duration": item.get("duration", ""),
                            "mp4": mp4,
                            "hls": hls,
                            "thumbnail": thumb,
                        })
            except Exception:
                pass

        return jsonify({
            "query": query,
            "player": name,
            "player_id": pid,
            "videos": videos,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/pitches/<int:game_pk>/<int:pitcher_id>")
def api_video_pitches(game_pk, pitcher_id):
    """Get pitch-by-pitch data with video URLs for a pitcher's outing.

    Returns every pitch thrown by the pitcher in the game, including:
    - Pitch type, velocity, result
    - Statcast data (spin, break, zone)
    - Video URL for each individual pitch (from Baseball Savant)
    """
    try:
        import re as _re

        # Get live feed with pitch-by-pitch data
        r = requests.get(f"{MLB_API}.1/game/{game_pk}/feed/live", timeout=15)
        data = r.json()

        all_plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])

        at_bats = []
        pitch_count = 0

        for play in all_plays:
            matchup = play.get("matchup", {})
            pitcher = matchup.get("pitcher", {})

            if pitcher.get("id") != pitcher_id:
                continue

            batter = matchup.get("batter", {})
            result = play.get("result", {})

            pitches = []
            for ev in play.get("playEvents", []):
                if not ev.get("isPitch"):
                    continue

                pitch_data = ev.get("pitchData", {})
                details = ev.get("details", {})
                pitch_count += 1

                pitch = {
                    "pitchNumber": pitch_count,
                    "abPitchNumber": ev.get("pitchNumber", 0),
                    "type": details.get("type", {}).get("code", ""),
                    "typeName": details.get("type", {}).get("description", ""),
                    "speed": pitch_data.get("startSpeed"),
                    "result": details.get("description", ""),
                    "isStrike": details.get("isStrike", False),
                    "isBall": details.get("isBall", False),
                    "isInPlay": details.get("isInPlay", False),
                    "zone": pitch_data.get("zone"),
                    "spinRate": pitch_data.get("breaks", {}).get("spinRate"),
                    "breakLength": pitch_data.get("breaks", {}).get("breakLength"),
                    "playId": ev.get("playId", ""),
                    "videoUrl": "",  # populated below
                }

                # Build video URL from playId
                if pitch["playId"]:
                    pitch["savantUrl"] = (
                        f"https://baseballsavant.mlb.com/sporty-videos"
                        f"?playId={pitch['playId']}"
                    )

                pitches.append(pitch)

            if pitches:
                at_bats.append({
                    "batter": batter.get("fullName", "?"),
                    "batterId": batter.get("id"),
                    "result": result.get("event", ""),
                    "description": result.get("description", ""),
                    "rbi": result.get("rbi", 0),
                    "pitches": pitches,
                })

        # Resolve video URLs in batch (grab from Savant)
        # Only resolve the first N to avoid hammering Savant
        resolved = 0
        for ab in at_bats:
            for pitch in ab["pitches"]:
                if pitch.get("savantUrl") and resolved < 50:
                    try:
                        rv = requests.get(
                            pitch["savantUrl"], timeout=5,
                            headers={"User-Agent": "Mozilla/5.0"},
                        )
                        if rv.status_code == 200:
                            match = _re.search(
                                r'src="(https://sporty-clips\.mlb\.com/[^"]+\.mp4)"',
                                rv.text,
                            )
                            if match:
                                pitch["videoUrl"] = match.group(1)
                                resolved += 1
                    except Exception:
                        pass

        # Game info
        game_data = data.get("gameData", {})
        teams = game_data.get("teams", {})
        away_name = teams.get("away", {}).get("name", "?")
        home_name = teams.get("home", {}).get("name", "?")
        date = game_data.get("datetime", {}).get("originalDate", "?")

        pitcher_name = ""
        if at_bats:
            # Get pitcher name from the play data
            for play in all_plays:
                if play.get("matchup", {}).get("pitcher", {}).get("id") == pitcher_id:
                    pitcher_name = play["matchup"]["pitcher"].get("fullName", "?")
                    break

        return jsonify({
            "game_pk": game_pk,
            "pitcher_id": pitcher_id,
            "pitcher_name": pitcher_name,
            "date": date,
            "matchup": f"{away_name} @ {home_name}",
            "total_pitches": pitch_count,
            "total_at_bats": len(at_bats),
            "videos_resolved": resolved,
            "at_bats": at_bats,
        })

    except Exception as e:
        logger.exception("Pitch video error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/pitch/<play_id>")
def api_video_single_pitch(play_id):
    """Get the video URL for a single pitch by its playId."""
    try:
        import re as _re
        url = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            match = _re.search(
                r'src="(https://sporty-clips\.mlb\.com/[^"]+\.mp4)"', r.text
            )
            if match:
                return jsonify({"playId": play_id, "videoUrl": match.group(1)})
        return jsonify({"playId": play_id, "videoUrl": "", "error": "Video not found"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════
#  ANALYTICS ENDPOINTS — projections, aging curves, HOF, embeddings
# ═══════════════════════════════════════════════════════════════════

_PROJECTIONS = None
_HOF_PROBS = None
_AGING = None


@app.route("/api/projections/<int:player_id>")
def api_projection(player_id):
    """Get 2026 projection for a player."""
    global _PROJECTIONS
    try:
        if _PROJECTIONS is None:
            _PROJECTIONS = pd.read_csv("data/projections_2026.csv")
        # Find by MLB playerID (lookup via search)
        r = requests.get(f"{MLB_API}/people/{player_id}", timeout=5)
        name = r.json().get("people", [{}])[0].get("fullName", "")
        if not name:
            return jsonify({"error": "Player not found"})

        last = name.split()[-1]
        match = _PROJECTIONS[_PROJECTIONS.name.str.contains(last, na=False)]
        if len(match) == 0:
            return jsonify({"error": "No projection available"})

        row = match.sort_values("projected_war", ascending=False).iloc[0]
        return jsonify({
            "name": row["name"],
            "team": row.get("team", "?"),
            "position": row.get("position", "?"),
            "age_2026": int(row.get("age_2026", 0)),
            "projected_war": round(float(row.projected_war), 1),
            "confidence": row.get("confidence", "Medium"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/hof/<player_id>")
def api_hof_probability(player_id):
    """Get Hall of Fame probability for a player."""
    global _HOF_PROBS
    try:
        if _HOF_PROBS is None:
            _HOF_PROBS = pd.read_csv("data/hof_probabilities.csv")
        match = _HOF_PROBS[_HOF_PROBS.playerID == player_id]
        if len(match) == 0:
            return jsonify({"error": "Player not found"})
        row = match.iloc[0]
        return jsonify({
            "playerID": player_id,
            "name": row["name"],
            "hof_probability": round(float(row.hof_prob), 4),
            "career_war_eq": round(float(row.career_war_eq), 1),
            "is_inducted": bool(row.get("is_hof", False)),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/aging-curve/<position>")
def api_aging_curve(position):
    """Get empirical aging curve for a position."""
    global _AGING
    try:
        if _AGING is None:
            _AGING = pd.read_csv("data/aging_curves_by_position.csv")
        pos_data = _AGING[_AGING.position == position.upper()]
        if len(pos_data) == 0:
            # Fall back to overall hitter curve
            hitter_aging = pd.read_csv("data/aging_curves_hitters.csv")
            return jsonify({
                "position": position,
                "curve": [{"age": int(r.age), "avg_war": round(float(r.avg_war), 2),
                           "n": int(r.n)} for _, r in hitter_aging.iterrows()],
            })
        return jsonify({
            "position": position,
            "curve": [{"age": int(r.age), "avg_war": round(float(r.avg_war), 2),
                       "n": int(r.n)} for _, r in pos_data.iterrows()],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/team-rankings/<int:year>")
def api_team_rankings(year):
    """Get team power rankings for a given year."""
    try:
        rankings = pd.read_csv("data/team_power_rankings.csv")
        yr = rankings[rankings.year == year].sort_values("total_war", ascending=False)
        return jsonify({
            "year": year,
            "teams": [{"team": r.team, "total_war": r.total_war, "bat_war": r.bat_war,
                       "pit_war": r.pit_war, "actual_w": int(r.actual_w)}
                      for _, r in yr.iterrows()],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/roster-optimizer")
def api_roster_optimizer():
    """Optimize a 26-man roster under a payroll budget.

    Query params:
        budget: Payroll budget in millions (default: 200)
    """
    try:
        from baseball_metric.analysis.roster_optimizer import (
            optimize_roster, load_projections, load_salaries,
        )
        budget_m = float(flask_request.args.get("budget", 200))
        budget_m = max(30, min(500, budget_m))  # clamp to reasonable range
        budget = budget_m * 1_000_000

        projections = load_projections()
        salaries = load_salaries()
        roster = optimize_roster(budget, projections=projections, salaries=salaries)
        return jsonify(roster.to_dict())
    except Exception as e:
        logger.exception("Roster optimizer error")
        return jsonify({"error": str(e)}), 500


# ── Live 2026 Dashboard ───────────────────────────────────────────────
# Self-contained endpoint using MLB Stats API, no database lookups.
# Results cached in-memory for 5 minutes to avoid hammering the API.

import time as _time

_dashboard_cache: dict[str, tuple[float, object]] = {}
_DASHBOARD_TTL = 300  # 5 minutes


def _dashboard_cached_get(key: str, fetcher):
    """Return cached value if fresh, otherwise call fetcher and cache it."""
    now = _time.time()
    if key in _dashboard_cache:
        ts, val = _dashboard_cache[key]
        if now - ts < _DASHBOARD_TTL:
            return val
    try:
        val = fetcher()
    except Exception as exc:
        logger.warning("Dashboard fetch failed for %s: %s", key, exc)
        # Return stale cache if available
        if key in _dashboard_cache:
            return _dashboard_cache[key][1]
        return None
    _dashboard_cache[key] = (now, val)
    return val


def _fetch_standings_2026():
    """Fetch current 2026 standings from MLB Stats API."""
    url = f"{MLB_API}/standings?leagueId=103,104&season=2026&standingsTypes=regularSeason"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    standings = []
    for record in data.get("records", []):
        league = record.get("league", {}).get("abbreviation", "")
        division = record.get("division", {}).get("abbreviation", "")
        for team_rec in record.get("teamRecords", []):
            team_info = team_rec.get("team", {})
            standings.append({
                "team_id": team_info.get("id"),
                "team_name": team_info.get("name", ""),
                "league": league,
                "division": division,
                "wins": team_rec.get("wins", 0),
                "losses": team_rec.get("losses", 0),
                "pct": team_rec.get("winningPercentage", ".000"),
                "games_back": team_rec.get("gamesBack", "-"),
                "streak": team_rec.get("streak", {}).get("streakCode", ""),
                "last_10": f"{team_rec.get('records', {}).get('splitRecords', [{}])[0].get('wins', 0)}-{team_rec.get('records', {}).get('splitRecords', [{}])[0].get('losses', 0)}",
                "run_diff": team_rec.get("runDifferential", 0),
            })
    return standings


def _fetch_stat_leaders_2026(stat_group: str, top_n: int = 15):
    """Fetch 2026 stat leaders from MLB Stats API.

    stat_group: 'hitting' or 'pitching'
    Returns top players by WAR-proxy metrics.
    """
    # Use the stats leaders endpoint for the current season
    if stat_group == "hitting":
        # Fetch OPS leaders as a proxy for offensive value
        url = (f"{MLB_API}/stats/leaders?leaderCategories=onBasePlusSlugging"
               f"&season=2026&sportId=1&limit={top_n}&statGroup=hitting")
    else:
        url = (f"{MLB_API}/stats/leaders?leaderCategories=earnedRunAverage"
               f"&season=2026&sportId=1&limit={top_n}&statGroup=pitching")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    leaders = []
    for cat in data.get("leagueLeaders", []):
        for entry in cat.get("leaders", []):
            person = entry.get("person", {})
            team = entry.get("team", {})
            stat_val = entry.get("value", "0")

            # Also fetch their full stats for WAR estimation
            leaders.append({
                "rank": entry.get("rank", 0),
                "player_id": person.get("id"),
                "name": person.get("fullName", ""),
                "team": team.get("name", ""),
                "team_id": team.get("id"),
                "stat_value": stat_val,
            })
    return leaders


def _fetch_player_stats_2026(player_id: int):
    """Fetch a single player's 2026 season stats."""
    url = (f"{MLB_API}/people/{player_id}/stats"
           f"?stats=season&season=2026&group=hitting,pitching")
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    result = {}
    for split_group in data.get("stats", []):
        group_name = split_group.get("group", {}).get("displayName", "")
        splits = split_group.get("splits", [])
        if splits:
            result[group_name] = splits[0].get("stat", {})
    return result


def _estimate_war_from_stats(stats: dict, is_pitcher: bool, games_played: int) -> float:
    """Estimate WAR from raw stats using wOBA-based approach.

    For hitters: WAR ~ (wOBA - .310) / 1.15 * PA / 600 * 6
    For pitchers: WAR ~ (4.50 - ERA) / 4.50 * IP / 200 * 6
    These are rough approximations for dashboard display.
    """
    if is_pitcher:
        era = float(stats.get("era", "4.50") or "4.50")
        ip = float(stats.get("inningsPitched", "0") or "0")
        if ip == 0:
            return 0.0
        # Pitcher WAR estimate: league-average ERA is ~4.30
        war = (4.30 - era) / 4.30 * (ip / 200) * 5.5
        return round(max(war, -2.0), 1)
    else:
        obp = float(stats.get("obp", ".000") or ".000")
        slg = float(stats.get("slg", ".000") or ".000")
        pa = int(stats.get("plateAppearances", 0) or 0)
        if pa == 0:
            return 0.0
        # wOBA estimate from OBP/SLG
        woba_est = obp * 0.69 + slg * 0.31  # rough wOBA proxy
        war = (woba_est - 0.310) / 1.15 * (pa / 600) * 6.0
        return round(max(war, -2.0), 1)


def _extrapolate_to_162(war: float, games: int) -> float:
    """Extrapolate current WAR pace to a full 162-game season."""
    if games <= 0:
        return 0.0
    return round(war * (162 / games), 1)


def _fetch_schedule_today():
    """Fetch today's MLB schedule."""
    from datetime import date
    today = date.today().isoformat()
    url = f"{MLB_API}/schedule?sportId=1&date={today}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            away = game.get("teams", {}).get("away", {})
            home = game.get("teams", {}).get("home", {})
            games.append({
                "game_pk": game.get("gamePk"),
                "status": game.get("status", {}).get("detailedState", ""),
                "away_team": away.get("team", {}).get("name", ""),
                "away_id": away.get("team", {}).get("id"),
                "away_score": away.get("score"),
                "away_wins": away.get("leagueRecord", {}).get("wins", 0),
                "away_losses": away.get("leagueRecord", {}).get("losses", 0),
                "home_team": home.get("team", {}).get("name", ""),
                "home_id": home.get("team", {}).get("id"),
                "home_score": home.get("score"),
                "home_wins": home.get("leagueRecord", {}).get("wins", 0),
                "home_losses": home.get("leagueRecord", {}).get("losses", 0),
                "venue": game.get("venue", {}).get("name", ""),
                "game_time": game.get("gameDate", ""),
            })
    return games


def _build_war_leaders():
    """Build WAR pace leader boards for hitters and pitchers."""
    hitter_leaders_raw = _dashboard_cached_get(
        "leaders_hitting", lambda: _fetch_stat_leaders_2026("hitting", 20)
    )
    pitcher_leaders_raw = _dashboard_cached_get(
        "leaders_pitching", lambda: _fetch_stat_leaders_2026("pitching", 20)
    )

    def _enrich(raw_leaders, is_pitcher):
        enriched = []
        if not raw_leaders:
            return enriched
        for ldr in raw_leaders[:15]:
            pid = ldr.get("player_id")
            if not pid:
                continue
            cache_key = f"pstats_{pid}"
            stats = _dashboard_cached_get(
                cache_key, lambda _pid=pid: _fetch_player_stats_2026(_pid)
            )
            if not stats:
                continue
            group_key = "pitching" if is_pitcher else "hitting"
            s = stats.get(group_key, stats.get("Pitching", stats.get("Hitting", {})))
            if not s:
                continue
            gp = int(s.get("gamesPlayed", 0) or 0)
            war_est = _estimate_war_from_stats(s, is_pitcher, gp)
            war_pace = _extrapolate_to_162(war_est, gp) if gp > 0 else 0.0
            entry = {
                "rank": ldr["rank"],
                "player_id": pid,
                "name": ldr["name"],
                "team": ldr["team"],
                "games": gp,
                "war_current": war_est,
                "war_pace_162": war_pace,
            }
            if is_pitcher:
                entry["era"] = s.get("era", "-")
                entry["ip"] = s.get("inningsPitched", "0")
                entry["strikeouts"] = s.get("strikeOuts", 0)
                entry["whip"] = s.get("whip", "-")
            else:
                entry["avg"] = s.get("avg", "-")
                entry["ops"] = s.get("ops", "-")
                entry["hr"] = s.get("homeRuns", 0)
                entry["rbi"] = s.get("rbi", 0)
                entry["pa"] = s.get("plateAppearances", 0)
            enriched.append(entry)
        # Sort by WAR pace descending
        enriched.sort(key=lambda x: x.get("war_pace_162", 0), reverse=True)
        return enriched

    return {
        "hitters": _enrich(hitter_leaders_raw, False),
        "pitchers": _enrich(pitcher_leaders_raw, True),
    }


def _build_award_races(war_leaders: dict, standings: list):
    """Build AL/NL MVP and Cy Young snapshots from WAR pace leaders."""
    # Map team names to league using standings data
    team_league = {}
    for s in (standings or []):
        team_league[s["team_name"]] = s.get("league", "")

    def _split_by_league(players):
        al, nl = [], []
        for p in players:
            league = team_league.get(p["team"], "")
            if league == "AL":
                al.append(p)
            elif league == "NL":
                nl.append(p)
            else:
                # Guess from team name if standings unavailable
                al.append(p)  # default fallback
        return al[:5], nl[:5]

    hitters = war_leaders.get("hitters", [])
    pitchers = war_leaders.get("pitchers", [])

    al_mvp, nl_mvp = _split_by_league(hitters)
    al_cy, nl_cy = _split_by_league(pitchers)

    return {
        "al_mvp": al_mvp,
        "nl_mvp": nl_mvp,
        "al_cy_young": al_cy,
        "nl_cy_young": nl_cy,
    }


@app.route("/api/dashboard/2026")
def api_dashboard_2026():
    """Live 2026 season dashboard.

    Returns WAR pace leaders, standings, award race snapshots, and today's
    schedule — all fetched from the MLB Stats API with 5-minute caching.
    """
    try:
        # Fetch all data (each piece is individually cached for 5 min)
        standings = _dashboard_cached_get("standings_2026", _fetch_standings_2026)
        war_leaders = _build_war_leaders()
        award_races = _build_award_races(war_leaders, standings or [])
        todays_games = _dashboard_cached_get("schedule_today", _fetch_schedule_today)

        return jsonify({
            "season": 2026,
            "war_pace_leaders": war_leaders,
            "standings": standings or [],
            "award_races": award_races,
            "todays_games": todays_games or [],
            "cached_at": _time.strftime("%Y-%m-%d %H:%M:%S UTC", _time.gmtime()),
        })
    except Exception as e:
        logger.exception("Dashboard 2026 error")
        return jsonify({"error": str(e)}), 500


# ── Trade Value API ────────────────────────────────────────────────────

@app.route("/api/trade-value/<player_name>")
def api_trade_value(player_name):
    """Compute trade value (surplus value) for a player.

    Optional query params:
        package: if 'true', also compute a fair-value trade package
        budget: max salary budget in millions for trade package (default: unlimited)
    """
    try:
        from baseball_metric.analysis.trade_calculator import (
            compute_value_by_name, compute_trade_package,
            load_projections, load_salaries, load_prospects,
        )
        projections = load_projections()
        salaries = load_salaries()
        prospects = load_prospects()

        # URL-decode spaces (Flask handles %20 but underscores are common too)
        name = player_name.replace("_", " ")

        val = compute_value_by_name(name, projections, salaries, prospects)
        if val is None:
            return jsonify({"error": f"Player '{name}' not found"}), 404

        result = {"player": val.to_dict()}

        # Optionally build a trade package
        if flask_request.args.get("package", "").lower() in ("true", "1", "yes"):
            budget_m = flask_request.args.get("budget", None)
            budget = float(budget_m) * 1e6 if budget_m else float("inf")
            pkg = compute_trade_package(
                name, budget, projections, salaries, prospects
            )
            if pkg:
                result["trade_package"] = pkg.to_dict()

        return jsonify(result)
    except Exception as e:
        logger.exception("Trade value error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    """Return comprehensive system statistics."""
    import glob, os
    csv_count = len(glob.glob("data/**/*.csv", recursive=True)) + len(glob.glob("data/*.csv"))
    model_count = len(glob.glob("models/*.pt"))
    py_count = len(glob.glob("**/*.py", recursive=True))
    return jsonify({
        "mlb_player_seasons": 75265,
        "milb_player_seasons": 223855,
        "total_player_seasons": 299120,
        "unique_careers": 14092,
        "year_range": "1920-2026",
        "csv_files": csv_count,
        "pytorch_models": model_count,
        "api_endpoints": 37,
        "web_tabs": 9,
        "data_sources": [
            "Lahman Baseball Database (1871-2025)",
            "MiLB Data Repository (2005-2026)",
            "Baseball Savant / Statcast (2015-2025)",
            "MLB Stats API (2026 live)",
        ],
        "models": [
            {"name": "Universal Player Transformer v2", "params": 3457093, "metric": "r=0.994 career WAR"},
            {"name": "Win Prediction", "params": 5000, "metric": "r=0.903 team wins"},
            {"name": "HOF Classifier", "params": 5000, "metric": "97.4% accuracy"},
            {"name": "Prospect Neural Net", "params": 20000, "metric": "r=0.405 MiLB->MLB"},
            {"name": "Player Autoencoder", "params": 10000, "metric": "16-dim embeddings"},
            {"name": "Lineup Optimizer", "params": 15000, "metric": "r=0.905 vs runs"},
            {"name": "Breakout Predictor", "params": "GBM", "metric": "AUC=0.774"},
        ],
    })


@app.route("/api/leaderboard-all/<int:year>")
def api_leaderboard_all(year):
    """Get full positional leaderboard for a year."""
    try:
        seasons, _ = _load_precomputed()
        yr = seasons[(seasons.yearID == year) & (seasons.PA >= 200)]
        result = {}
        for pos in ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]:
            pos_data = yr[yr.position == pos].sort_values("bravs_war_eq", ascending=False).head(10)
            result[pos] = [{
                "name": r["name"], "team": r.team,
                "war": round(r.bravs_war_eq, 1),
                "hitting": round(r.hitting_runs, 1),
                "pa": int(r.PA),
            } for _, r in pos_data.iterrows()]
        return jsonify({"year": year, "positions": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Live Game Feed ──────────────────────────────────────────────────
from datetime import date as _date


_WP_BUNDLE = None


def _load_wp_bundle():
    """Lazy-load the v7 model bundle (includes v5 table for state lookups)."""
    global _WP_BUNDLE
    if _WP_BUNDLE is not None:
        return _WP_BUNDLE
    import pickle
    path = os.path.join(os.path.dirname(__file__), "..",
                        "data", "win_prob", "wp_model_v7.pkl")
    if not os.path.exists(path):
        # Fall back to v6, then nothing
        alt = os.path.join(os.path.dirname(__file__), "..",
                           "data", "win_prob", "wp_model_v6.pkl")
        path = alt if os.path.exists(alt) else None
    if path is None:
        _WP_BUNDLE = {}
        return _WP_BUNDLE
    with open(path, "rb") as f:
        _WP_BUNDLE = pickle.load(f)
    return _WP_BUNDLE


def _wp_lookup_state(inning, is_bot, outs, diff, bases, prior_weight=15.0):
    """State-only WP from the empirical v5 table. Batter's POV."""
    bundle = _load_wp_bundle()
    table = bundle.get("v5_table")
    prior_map = bundle.get("v5_prior_map")
    if not table:
        return 0.5
    inning_b = min(max(int(inning), 1), 12)
    diff_c = max(-10, min(10, int(diff)))
    is_bot = 1 if int(is_bot) else 0
    outs = max(0, min(2, int(outs)))
    bases = max(0, min(7, int(bases)))

    p = prior_map["ihd"].get((inning_b, is_bot, diff_c))
    if p is None:
        p = prior_map["d"].get(diff_c, 0.5)

    key = (inning_b, is_bot, outs, diff_c, bases)
    if key in table:
        w, t = table[key]
        return (w + p * prior_weight) / (t + prior_weight)
    # Fall back: aggregate across outs+bases at same (inning, half, diff)
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


def _win_probability(score_ahead, score_behind, inning, top_bottom="top"):
    """Empirical v5-table win probability from home team's POV.

    Uses ~1.7M plays of real Retrosheet outcomes. When live-game state is
    partial (no outs/bases info from the MLB StatsAPI schedule), we assume
    0 outs / empty bases — the inning and score diff still dominate.
    """
    # Normalise: our table keys on the BATTING team's score diff.
    is_bot = 1 if str(top_bottom).lower().startswith("b") else 0
    diff = score_ahead - score_behind  # from the "ahead" (home) perspective
    # Convert to batter's POV and look up
    bat_diff = diff if is_bot else -diff
    wp_bat = _wp_lookup_state(int(inning), is_bot, 0, bat_diff, 0)
    # Return from home's POV
    wp_home = wp_bat if is_bot else 1.0 - wp_bat
    return max(0.01, min(0.99, float(wp_home)))


def _team_record_str(rec: dict | None) -> str:
    if not rec:
        return ""
    w = rec.get("wins")
    l = rec.get("losses")
    if w is None or l is None:
        return ""
    return f"{w}-{l}"


def _format_game_time(date_str: str | None) -> str:
    """Convert MLB schedule date string to a friendly local time."""
    if not date_str:
        return ""
    try:
        from datetime import datetime, timezone
        # MLB returns ISO 8601 like 2026-04-14T18:35:00Z
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        local = dt.astimezone()
        return local.strftime("%-I:%M %p") if os.name != "nt" else local.strftime("%#I:%M %p")
    except Exception:
        return date_str


def _bases_state(linescore: dict) -> dict:
    """Return which bases are occupied from linescore.offense block."""
    off = linescore.get("offense", {}) or {}
    return {
        "first": bool(off.get("first")),
        "second": bool(off.get("second")),
        "third": bool(off.get("third")),
    }


def _bases_idx(bases: dict) -> int:
    return (1 if bases.get("first") else 0) + (2 if bases.get("second") else 0) + (4 if bases.get("third") else 0)


@app.route("/api/live/scoreboard")
def api_live_scoreboard():
    """Today's MLB games with rich data: logos, records, probable pitcher,
    bases, count, broadcast, weather, venue, and v5-empirical win probability.
    """
    today = flask_request.args.get("date", _date.today().isoformat())
    try:
        hydrate = (
            "linescore(matchup,runners),team,probablePitcher,broadcasts(all),"
            "weather,venue,decisions,flags,seriesStatus,game(content(summary,media(epg)))"
        )
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate={hydrate}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        games = []
        for d in data.get("dates", []):
            for g in d.get("games", []):
                away = g.get("teams", {}).get("away", {})
                home = g.get("teams", {}).get("home", {})
                status = g.get("status", {})
                linescore = g.get("linescore", {}) or {}
                inning = linescore.get("currentInning", 0) or 0
                half = linescore.get("inningHalf", "Top") or "Top"
                inning_state = linescore.get("inningState", "") or ""
                away_score = away.get("score", 0) or 0
                home_score = home.get("score", 0) or 0

                abstract = status.get("abstractGameState", "Preview")
                detail = status.get("detailedState", "Scheduled")

                # Bases / count / outs (only meaningful when game is live)
                bases = _bases_state(linescore)
                bases_idx = _bases_idx(bases)
                balls = linescore.get("balls", 0) or 0
                strikes = linescore.get("strikes", 0) or 0
                outs = linescore.get("outs", 0) or 0
                away_h = (linescore.get("teams", {}).get("away", {}) or {}).get("hits", 0)
                home_h = (linescore.get("teams", {}).get("home", {}) or {}).get("hits", 0)
                away_e = (linescore.get("teams", {}).get("away", {}) or {}).get("errors", 0)
                home_e = (linescore.get("teams", {}).get("home", {}) or {}).get("errors", 0)
                away_lob = (linescore.get("teams", {}).get("away", {}) or {}).get("leftOnBase", 0)
                home_lob = (linescore.get("teams", {}).get("home", {}) or {}).get("leftOnBase", 0)

                # Win probability via v5 empirical table (state-aware when live)
                if abstract == "Live":
                    is_bot = 1 if str(half).lower().startswith("b") else 0
                    bat_diff = (home_score - away_score) if is_bot else (away_score - home_score)
                    wp_bat = _wp_lookup_state(inning, is_bot, outs, bat_diff, bases_idx)
                    home_wp = wp_bat if is_bot else 1.0 - wp_bat
                elif abstract == "Final":
                    if home_score > away_score:
                        home_wp = 1.0
                    elif home_score < away_score:
                        home_wp = 0.0
                    else:
                        home_wp = 0.5
                else:
                    home_wp = 0.5
                away_wp = 1.0 - home_wp

                away_team = away.get("team", {}) or {}
                home_team = home.get("team", {}) or {}

                # Probable / current pitcher
                def _pitcher_summary(p: dict) -> dict | None:
                    if not p:
                        return None
                    pid = p.get("id")
                    return {
                        "id": pid,
                        "name": p.get("fullName") or p.get("name", "?"),
                        "headshot": HEADSHOT_URL.format(pid=pid) if pid else "",
                    }

                away_prob = _pitcher_summary(away.get("probablePitcher"))
                home_prob = _pitcher_summary(home.get("probablePitcher"))

                # Decisions (winning/losing pitcher — only post-game)
                decisions = g.get("decisions", {}) or {}
                winner_p = decisions.get("winner")
                loser_p = decisions.get("loser")
                save_p = decisions.get("save")

                # Broadcasts (filter to TV)
                bcasts = g.get("broadcasts", []) or []
                tv = [b.get("name") for b in bcasts
                      if b.get("type") == "TV" and b.get("name")]

                weather = g.get("weather", {}) or {}
                venue = g.get("venue", {}) or {}

                games.append({
                    "gamePk": g.get("gamePk"),
                    "status": abstract,
                    "detailedState": detail,
                    "inning": inning,
                    "inningHalf": half,
                    "inningState": inning_state,
                    "gameTime": _format_game_time(g.get("gameDate")),
                    "venue": venue.get("name", ""),
                    "weather": {
                        "condition": weather.get("condition", ""),
                        "temp": weather.get("temp", ""),
                        "wind": weather.get("wind", ""),
                    },
                    "broadcasts": tv,
                    "bases": bases,
                    "basesIdx": bases_idx,
                    "balls": balls,
                    "strikes": strikes,
                    "outs": outs,
                    "decisions": {
                        "win": _pitcher_summary(winner_p),
                        "loss": _pitcher_summary(loser_p),
                        "save": _pitcher_summary(save_p),
                    } if (winner_p or loser_p or save_p) else None,
                    "away": {
                        "name": away_team.get("name", "?"),
                        "abbreviation": away_team.get("abbreviation", "?"),
                        "shortName": away_team.get("teamName", away_team.get("name", "?")),
                        "id": away_team.get("id"),
                        "logo": TEAM_LOGO_URL.format(tid=away_team.get("id")) if away_team.get("id") else "",
                        "score": away_score,
                        "hits": away_h,
                        "errors": away_e,
                        "lob": away_lob,
                        "wp": round(away_wp, 3),
                        "record": _team_record_str(away.get("leagueRecord")),
                        "probablePitcher": away_prob,
                    },
                    "home": {
                        "name": home_team.get("name", "?"),
                        "abbreviation": home_team.get("abbreviation", "?"),
                        "shortName": home_team.get("teamName", home_team.get("name", "?")),
                        "id": home_team.get("id"),
                        "logo": TEAM_LOGO_URL.format(tid=home_team.get("id")) if home_team.get("id") else "",
                        "score": home_score,
                        "hits": home_h,
                        "errors": home_e,
                        "lob": home_lob,
                        "wp": round(home_wp, 3),
                        "record": _team_record_str(home.get("leagueRecord")),
                        "probablePitcher": home_prob,
                    },
                })
        has_live = any(g["status"] == "Live" for g in games)
        return jsonify({
            "date": today,
            "games": games,
            "hasLive": has_live,
            "fetchedAt": _date.today().isoformat(),
        })
    except Exception as e:
        logger.error("Live scoreboard error: %s", e)
        return jsonify({"error": str(e), "games": []}), 500


@app.route("/api/live/game/<int:game_pk>")
def api_live_game(game_pk):
    """Rich game data for the modal view: full linescore w/ R/H/E,
    current at-bat with last few pitches, recent plays, scoring summary,
    box-score totals, decisions, weather, venue."""
    try:
        feed_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        feed_resp = requests.get(feed_url, timeout=12)
        feed_resp.raise_for_status()
        feed = feed_resp.json()

        gamedata = feed.get("gameData", {}) or {}
        livedata = feed.get("liveData", {}) or {}
        ls = livedata.get("linescore", {}) or {}
        plays_block = livedata.get("plays", {}) or {}
        all_plays = plays_block.get("allPlays", []) or []
        cur_play = plays_block.get("currentPlay") or {}

        teams = gamedata.get("teams", {}) or {}
        away_team = teams.get("away", {}) or {}
        home_team = teams.get("home", {}) or {}
        status = gamedata.get("status", {}) or {}
        venue = gamedata.get("venue", {}) or {}
        weather = gamedata.get("weather", {}) or {}

        away_id = away_team.get("id")
        home_id = home_team.get("id")
        away_logo = TEAM_LOGO_URL.format(tid=away_id) if away_id else ""
        home_logo = TEAM_LOGO_URL.format(tid=home_id) if home_id else ""

        # Inning-by-inning linescore with R/H/E
        innings = []
        for inn in ls.get("innings", []):
            innings.append({
                "num": inn.get("num"),
                "away": {
                    "runs": (inn.get("away", {}) or {}).get("runs"),
                    "hits": (inn.get("away", {}) or {}).get("hits"),
                    "errors": (inn.get("away", {}) or {}).get("errors"),
                },
                "home": {
                    "runs": (inn.get("home", {}) or {}).get("runs"),
                    "hits": (inn.get("home", {}) or {}).get("hits"),
                    "errors": (inn.get("home", {}) or {}).get("errors"),
                },
            })
        ls_teams = ls.get("teams", {}) or {}
        away_totals = {
            "R": (ls_teams.get("away", {}) or {}).get("runs", 0),
            "H": (ls_teams.get("away", {}) or {}).get("hits", 0),
            "E": (ls_teams.get("away", {}) or {}).get("errors", 0),
            "LOB": (ls_teams.get("away", {}) or {}).get("leftOnBase", 0),
        }
        home_totals = {
            "R": (ls_teams.get("home", {}) or {}).get("runs", 0),
            "H": (ls_teams.get("home", {}) or {}).get("hits", 0),
            "E": (ls_teams.get("home", {}) or {}).get("errors", 0),
            "LOB": (ls_teams.get("home", {}) or {}).get("leftOnBase", 0),
        }

        # Current at-bat
        offense = ls.get("offense", {}) or {}
        defense = ls.get("defense", {}) or {}
        cur_batter_obj = offense.get("batter") or {}
        cur_pitcher_obj = defense.get("pitcher") or {}

        def _person_card(p: dict) -> dict | None:
            if not p:
                return None
            pid = p.get("id")
            return {
                "id": pid,
                "name": p.get("fullName", p.get("name", "?")),
                "headshot": HEADSHOT_URL.format(pid=pid) if pid else "",
            }

        # Pull last few pitches of the current at-bat
        recent_pitches = []
        if cur_play and isinstance(cur_play, dict):
            for ev in (cur_play.get("playEvents", []) or []):
                if not ev.get("isPitch"):
                    continue
                det = ev.get("details", {}) or {}
                pd_ = ev.get("pitchData", {}) or {}
                recent_pitches.append({
                    "type": (det.get("type", {}) or {}).get("description", ""),
                    "code": (det.get("type", {}) or {}).get("code", ""),
                    "result": det.get("description", ""),
                    "ball": det.get("isBall", False),
                    "strike": det.get("isStrike", False),
                    "inPlay": det.get("isInPlay", False),
                    "velocity": round(pd_.get("startSpeed", 0), 1) if pd_.get("startSpeed") else None,
                    "nastyFactor": pd_.get("breaks", {}).get("breakLength") if isinstance(pd_.get("breaks"), dict) else None,
                })
            recent_pitches = recent_pitches[-6:]  # last 6

        # Recent plays (last 8, completed only)
        recent_plays = []
        for p in reversed(all_plays):
            if not p.get("about", {}).get("isComplete"):
                continue
            about = p.get("about", {}) or {}
            res = p.get("result", {}) or {}
            recent_plays.append({
                "halfInning": about.get("halfInning", ""),
                "inning": about.get("inning"),
                "event": res.get("event", ""),
                "description": res.get("description", ""),
                "rbi": res.get("rbi", 0),
                "awayScore": res.get("awayScore", 0),
                "homeScore": res.get("homeScore", 0),
                "isScoringPlay": about.get("isScoringPlay", False),
            })
            if len(recent_plays) >= 8:
                break

        # Scoring plays (canonical list)
        scoring = []
        for idx in plays_block.get("scoringPlays", []) or []:
            if isinstance(idx, int) and 0 <= idx < len(all_plays):
                p = all_plays[idx]
                about = p.get("about", {}) or {}
                res = p.get("result", {}) or {}
                scoring.append({
                    "halfInning": about.get("halfInning", ""),
                    "inning": about.get("inning"),
                    "description": res.get("description", ""),
                    "awayScore": res.get("awayScore", 0),
                    "homeScore": res.get("homeScore", 0),
                })

        # Decisions (post-game)
        decisions_raw = livedata.get("decisions", {}) or {}
        decisions = None
        if decisions_raw:
            decisions = {
                "win": _person_card(decisions_raw.get("winner")),
                "loss": _person_card(decisions_raw.get("loser")),
                "save": _person_card(decisions_raw.get("save")),
            }

        # Probable pitchers (from gameData.probablePitchers)
        prob = gamedata.get("probablePitchers", {}) or {}
        prob_away = _person_card(prob.get("away"))
        prob_home = _person_card(prob.get("home"))

        # Bases
        bases = {
            "first": bool(offense.get("first")),
            "second": bool(offense.get("second")),
            "third": bool(offense.get("third")),
        }
        bases_idx = _bases_idx(bases)

        # Win probability snapshot
        inning = ls.get("currentInning", 0) or 0
        half = ls.get("inningHalf", "Top") or "Top"
        outs = ls.get("outs", 0) or 0
        balls = ls.get("balls", 0) or 0
        strikes = ls.get("strikes", 0) or 0
        abstract = status.get("abstractGameState", "Preview")
        if abstract == "Live" and inning:
            is_bot = 1 if str(half).lower().startswith("b") else 0
            bat_diff = (home_totals["R"] - away_totals["R"]) if is_bot \
                else (away_totals["R"] - home_totals["R"])
            wp_bat = _wp_lookup_state(inning, is_bot, outs, bat_diff, bases_idx)
            home_wp = wp_bat if is_bot else 1.0 - wp_bat
        elif abstract == "Final":
            home_wp = 1.0 if home_totals["R"] > away_totals["R"] else (
                0.5 if home_totals["R"] == away_totals["R"] else 0.0)
        else:
            home_wp = 0.5
        away_wp = 1.0 - home_wp

        return jsonify({
            "gamePk": game_pk,
            "status": abstract,
            "detailedState": status.get("detailedState", ""),
            "venue": venue.get("name", ""),
            "weather": {
                "condition": weather.get("condition", ""),
                "temp": weather.get("temp", ""),
                "wind": weather.get("wind", ""),
            },
            "away": {
                "id": away_id,
                "name": away_team.get("name", "?"),
                "abbreviation": away_team.get("abbreviation", "?"),
                "shortName": away_team.get("teamName", away_team.get("name", "?")),
                "logo": away_logo,
                "totals": away_totals,
                "wp": round(away_wp, 3),
                "probablePitcher": prob_away,
            },
            "home": {
                "id": home_id,
                "name": home_team.get("name", "?"),
                "abbreviation": home_team.get("abbreviation", "?"),
                "shortName": home_team.get("teamName", home_team.get("name", "?")),
                "logo": home_logo,
                "totals": home_totals,
                "wp": round(home_wp, 3),
                "probablePitcher": prob_home,
            },
            "innings": innings,
            "currentInning": inning,
            "inningHalf": half,
            "inningState": ls.get("inningState", ""),
            "balls": balls,
            "strikes": strikes,
            "outs": outs,
            "bases": bases,
            "currentBatter": _person_card(cur_batter_obj),
            "currentPitcher": _person_card(cur_pitcher_obj),
            "recentPitches": recent_pitches,
            "recentPlays": recent_plays,
            "scoringPlays": scoring,
            "decisions": decisions,
        })
    except Exception as e:
        logger.error("Live game %d error: %s", game_pk, e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/simulate", methods=["POST"])
def api_simulate_game():
    """Simulate a game between two teams.

    POST body: {"home_team": "NYA", "away_team": "LAN", "year": 2024, "n_sims": 5000}
    """
    try:
        from baseball_metric.analysis.game_simulator import simulate_matchup

        data = flask_request.get_json(force=True)
        home_t = data.get("home_team", "NYA")
        away_t = data.get("away_team", "LAN")
        year = int(data.get("year", 2024))
        n_sims = min(int(data.get("n_sims", 5000)), 20000)

        seasons, _ = _load_precomputed()
        home_data = seasons[(seasons.yearID == year) & (seasons.team == home_t) & (seasons.PA >= 100)]
        away_data = seasons[(seasons.yearID == year) & (seasons.team == away_t) & (seasons.PA >= 100)]

        if len(home_data) < 5 or len(away_data) < 5:
            return jsonify({"error": f"Not enough data for {home_t} or {away_t} in {year}"}), 400

        result = simulate_matchup(
            home_data.to_dict("records"),
            away_data.to_dict("records"),
            n_sims=n_sims,
        )

        return jsonify({
            "home_team": home_t,
            "away_team": away_t,
            "year": year,
            "n_sims": n_sims,
            "home_win_pct": round(result.home_win_pct, 3),
            "avg_home_runs": round(result.avg_home_runs, 1),
            "avg_away_runs": round(result.avg_away_runs, 1),
            "summary": result.summary,
        })
    except Exception as e:
        logger.exception("Simulation error")
        return jsonify({"error": str(e)}), 500


# ── Win Probability / WPA API ───────────────────────────────────────


@app.route("/api/wp/state")
def api_wp_state():
    """Return v5-empirical win probability for a game state.

    Query params:
      inning (1-12), half (T/B), outs (0-2), diff (-10..10, batter POV),
      bases (0-7: bit0=1B, bit1=2B, bit2=3B)

    Returns WP from both the batting team's and home team's POV.
    """
    try:
        inning = int(flask_request.args.get("inning", 1))
        half = str(flask_request.args.get("half", "T")).upper()
        outs = int(flask_request.args.get("outs", 0))
        diff = int(flask_request.args.get("diff", 0))
        bases = int(flask_request.args.get("bases", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "bad params"}), 400
    is_bot = 1 if half.startswith("B") else 0
    wp_bat = _wp_lookup_state(inning, is_bot, outs, diff, bases)
    wp_home = wp_bat if is_bot else 1.0 - wp_bat
    return jsonify({
        "inning": inning, "half": "B" if is_bot else "T",
        "outs": outs, "score_diff_bat": diff, "bases_idx": bases,
        "wp_bat": round(float(wp_bat), 4),
        "wp_home": round(float(wp_home), 4),
        "source": "v5_empirical_table",
    })


@app.route("/api/wpa/top_plays")
def api_wpa_top_plays():
    """Top single-play WP swings 2015-2024 from the pre-computed table."""
    limit = int(flask_request.args.get("limit", 50))
    year = flask_request.args.get("year")
    path = os.path.join(os.path.dirname(__file__), "..",
                        "data", "win_prob", "top_plays_2015_2024.csv")
    if not os.path.exists(path):
        return jsonify({"error": "top plays not yet generated"}), 404
    df = pd.read_csv(path)
    if year:
        df = df[df.year == int(year)]
    df = df.reindex(df.dWP.abs().sort_values(ascending=False).index).head(limit)
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/wpa/leaderboard/<role>")
def api_wpa_leaderboard(role):
    """WPA leaderboard for batters or pitchers, by season or career.

    role: 'batter' or 'pitcher'
    query: year=2024 (season) or career=1 (career 2015-2024)
            clutch=1 to sort by clutch_WPA instead of WPA
            limit=N (default 25)
    """
    if role not in ("batter", "pitcher"):
        return jsonify({"error": "role must be batter or pitcher"}), 400
    career = flask_request.args.get("career", "").lower() in ("1", "true", "yes")
    clutch = flask_request.args.get("clutch", "").lower() in ("1", "true", "yes")
    limit = int(flask_request.args.get("limit", 25))
    base = os.path.join(os.path.dirname(__file__), "..", "data", "win_prob")
    if career:
        path = os.path.join(base, f"wpa_career_{role}s.csv")
    else:
        path = os.path.join(base, f"wpa_season_{role}s.csv")
    if not os.path.exists(path):
        return jsonify({"error": f"{path} not generated"}), 404
    df = pd.read_csv(path)
    year = flask_request.args.get("year")
    if year and not career:
        df = df[df.year == int(year)]
    sort_col = "clutch_WPA" if clutch else "WPA"
    df = df.sort_values(sort_col, ascending=False).head(limit)
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/wpa/managers/<int:year>")
def api_wpa_managers(year):
    """Manager decision stats (pitching changes, defensive WPA) for a year."""
    path = os.path.join(os.path.dirname(__file__), "..",
                        "data", "win_prob",
                        f"manager_decisions_{year}.csv")
    if not os.path.exists(path):
        return jsonify({"error": f"{path} not generated"}), 404
    df = pd.read_csv(path).sort_values("def_WPA", ascending=False)
    return jsonify(df.to_dict(orient="records"))


if __name__ == "__main__":
    print("\n  BRAVS Web App")
    print(f"  Engine: {'Rust (bravs_engine)' if USE_RUST else 'Python (fallback)'}")
    print("  http://localhost:5000\n")
    app.run(debug=True, port=5000)
