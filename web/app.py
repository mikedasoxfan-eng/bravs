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

from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import PlayerSeason
from baseball_metric.adjustments.era_adjustment import get_rpg
from baseball_metric.adjustments.park_factors import get_park_factor

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
    """Compute BRAVS for all top candidates in an award race.

    Uses multiple leader categories to build a robust candidate pool:
    - MVP: OPS leaders + PA leaders + HR leaders (deduped)
    - Cy Young: ERA leaders + K leaders + IP leaders + SV leaders (deduped)
    """
    league_id = 103 if league.upper() == "AL" else 104

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
        # MVP: combine OPS leaders (best performers) + PA leaders (everyday players)
        _add_candidates(_fetch_leaders("onBasePlusSlugging", season, league_id, 12))
        _add_candidates(_fetch_leaders("plateAppearances", season, league_id, 8))
        _add_candidates(_fetch_leaders("homeRuns", season, league_id, 5))

    # Compute BRAVS for all candidates in parallel
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
        "league": league.upper(),
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
        "bravs_war_eq": round(new_bravs * 0.62, 1),
        "positional_diff_runs": round(pos_diff, 1),
    })


@app.route("/api/team/<team_abbrev>/<int:season>")
def team_roster(team_abbrev, season):
    """Get all players on a team for a season and rank by BRAVS."""
    # Resolve team abbreviation to MLB team ID
    team_id = TEAM_IDS.get(team_abbrev.upper())
    if not team_id:
        # Try looking up by searching all teams
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

    # Fetch actual roster from MLB API
    roster_data = _mlb_get(
        f"{MLB_API}/teams/{team_id}/roster",
        {"season": season, "rosterType": "fullSeason"},
    )
    if not roster_data or "roster" not in roster_data:
        return jsonify({"error": "Could not fetch roster"}), 500

    # Extract player IDs and names
    roster_players = []
    for entry in roster_data["roster"]:
        person = entry.get("person", {})
        pid = person.get("id")
        name = person.get("fullName", "Unknown")
        pos = entry.get("position", {}).get("abbreviation", "")
        if pid:
            roster_players.append({"id": pid, "name": name, "position": pos})

    # Compute BRAVS for all roster players in parallel
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
    """Live MVP race for the current season, adjusted for pace."""
    import datetime
    current_year = datetime.date.today().year
    league_id = 103 if league.upper() == "AL" else 104

    # Gather candidates from multiple leader categories
    seen_ids: set[int] = set()
    candidates: list[dict] = []

    def _add(leaders: list[dict]) -> None:
        for p in leaders:
            if p["id"] not in seen_ids:
                seen_ids.add(p["id"])
                candidates.append(p)

    _add(_fetch_leaders("onBasePlusSlugging", current_year, league_id, 12))
    _add(_fetch_leaders("plateAppearances", current_year, league_id, 8))
    _add(_fetch_leaders("homeRuns", current_year, league_id, 5))
    _add(_fetch_pitching_leaders("earnedRunAverage", current_year, league_id, 8))
    _add(_fetch_pitching_leaders("strikeouts", current_year, league_id, 5))
    _add(_fetch_pitching_leaders("inningsPitched", current_year, league_id, 5))

    # Compute BRAVS for all candidates in parallel
    raw_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(compute_player_bravs_internal, cand["id"], current_year): cand
            for cand in candidates[:20]
        }
        for future in as_completed(futures):
            cand = futures[future]
            try:
                resp_data = future.result()
                if resp_data and "error" not in resp_data:
                    resp_data["player_id"] = cand["id"]
                    raw_results.append(resp_data)
            except Exception as e:
                logger.warning("Live MVP failed for %s: %s", cand["name"], e)

    # Adjust for in-progress season: if durability is significantly negative,
    # add it back to get a "pace" BRAVS that doesn't penalize unplayed games.
    adjusted: list[dict] = []
    for r in raw_results:
        comps = _extract_component_runs(r)
        durability = comps.get("durability", 0.0)
        raw_bravs = r.get("bravs", 0.0)
        if durability < -5.0:
            pace_bravs = round(raw_bravs - durability / r.get("rpw", 5.9), 1)
        else:
            pace_bravs = raw_bravs

        games = 0
        trad = r.get("traditional", {})
        if trad.get("batting", {}).get("G"):
            games = trad["batting"]["G"]
        elif trad.get("pitching", {}).get("G"):
            games = trad["pitching"]["G"]
        est_team_games = games + 5

        adjusted.append({
            "player_name": r.get("player_name", ""),
            "player_id": r.get("player_id"),
            "position": r.get("position", ""),
            "team": r.get("team", ""),
            "team_abbrev": r.get("team_abbrev", ""),
            "team_id": r.get("team_id"),
            "headshot": r.get("headshot", ""),
            "bravs": pace_bravs,
            "bravs_raw": raw_bravs,
            "bravs_pace": pace_bravs,
            "bravs_era_std": r.get("bravs_era_std", 0),
            "bravs_war_eq": round(pace_bravs * 0.62, 1),
            "games_played": games,
            "est_team_games": est_team_games,
            "components": r.get("components", []),
        })

    adjusted.sort(key=lambda x: x["bravs_pace"], reverse=True)

    return jsonify({
        "season": current_year,
        "league": league.upper(),
        "candidates": adjusted,
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


if __name__ == "__main__":
    print("\n  BRAVS Web App")
    print(f"  Engine: {'Rust (bravs_engine)' if USE_RUST else 'Python (fallback)'}")
    print("  http://localhost:5000\n")
    app.run(debug=True, port=5000)
