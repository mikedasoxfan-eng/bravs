"""BRAVS Web App — Compute BRAVS for any MLB player, any season."""

from __future__ import annotations

import logging
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from flask import Flask, render_template, jsonify, request as flask_request

from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import PlayerSeason
from baseball_metric.adjustments.era_adjustment import get_rpg
from baseball_metric.adjustments.park_factors import get_park_factor

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Fetch from MLB Stats API with caching."""
    cache_key = url + str(params or {})
    if cache_key in _cache:
        return _cache[cache_key]  # type: ignore[return-value]
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        _cache[cache_key] = data
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

    result = compute_bravs(ps)

    # Build response
    components = []
    for name, comp in sorted(result.components.items()):
        components.append({
            "name": name,
            "runs": round(comp.runs_mean, 1),
            "ci_lo": round(comp.ci_90[0], 1),
            "ci_hi": round(comp.ci_90[1], 1),
        })

    ci90 = result.bravs_ci_90

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
        "bravs": round(result.bravs, 1),
        "bravs_era_std": round(result.bravs_era_standardized, 1),
        "bravs_war_eq": round(result.bravs_calibrated, 1),
        "ci90_lo": round(ci90[0], 1),
        "ci90_hi": round(ci90[1], 1),
        "total_runs": round(result.total_runs_mean, 1),
        "rpw": round(result.rpw, 2),
        "leverage_mult": round(result.leverage_multiplier, 3),
        "park_factor": round(pf.overall, 2),
        "components": components,
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

    # Compute BRAVS for each candidate
    results = []
    for cand in candidates[:10]:
        try:
            resp_data = compute_player_bravs_internal(cand["id"], season)
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


if __name__ == "__main__":
    print("\n  BRAVS Web App")
    print("  http://localhost:5000\n")
    app.run(debug=True, port=5000)
