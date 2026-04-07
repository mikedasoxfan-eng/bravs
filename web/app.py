"""BRAVS Web Application — compute BRAVS for any MLB player."""

from __future__ import annotations

import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, render_template, request, jsonify

from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import PlayerSeason, BRAVSResult
from baseball_metric.data.validation import validate_player_season

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


def _build_player_from_form(form: dict) -> PlayerSeason:
    """Build a PlayerSeason from form data."""
    def _int(key: str, default: int = 0) -> int:
        val = form.get(key, "")
        return int(val) if val not in ("", None) else default

    def _float(key: str, default: float = 0.0) -> float:
        val = form.get(key, "")
        return float(val) if val not in ("", None) else default

    def _float_or_none(key: str) -> float | None:
        val = form.get(key, "")
        return float(val) if val not in ("", None) else None

    return PlayerSeason(
        player_id=form.get("player_name", "custom").replace(" ", "_").lower(),
        player_name=form.get("player_name", "Unknown"),
        season=_int("season", 2024),
        team=form.get("team", "UNK"),
        position=form.get("position", "DH"),
        # Batting
        pa=_int("pa"), ab=_int("ab"), hits=_int("hits"),
        doubles=_int("doubles"), triples=_int("triples"), hr=_int("hr"),
        bb=_int("bb"), ibb=_int("ibb"), hbp=_int("hbp"),
        k=_int("k"), sf=_int("sf"), sb=_int("sb"), cs=_int("cs"),
        gidp=_int("gidp"), games=_int("games"),
        # Pitching
        ip=_float("ip"), er=_int("er"), hits_allowed=_int("hits_allowed"),
        hr_allowed=_int("hr_allowed"), bb_allowed=_int("bb_allowed"),
        hbp_allowed=_int("hbp_allowed"), k_pitching=_int("k_pitching"),
        games_pitched=_int("games_pitched"), games_started=_int("games_started"),
        saves=_int("saves"),
        # Fielding
        inn_fielded=_float("inn_fielded"),
        uzr=_float_or_none("uzr"), drs=_float_or_none("drs"), oaa=_float_or_none("oaa"),
        # Catcher
        framing_runs=_float_or_none("framing_runs"),
        blocking_runs=_float_or_none("blocking_runs"),
        throwing_runs=_float_or_none("throwing_runs"),
        catcher_pitches=_int("catcher_pitches"),
        # Context
        avg_leverage_index=_float("avg_leverage_index", 1.0),
        park_factor=_float("park_factor", 1.0),
        league_rpg=_float("league_rpg", 4.5),
        season_games=_int("season_games", 162),
    )


def _result_to_dict(result: BRAVSResult) -> dict:
    """Convert BRAVSResult to JSON-serializable dict."""
    components = []
    for name, comp in sorted(result.components.items()):
        components.append({
            "name": name,
            "runs": round(comp.runs_mean, 1),
            "ci_lo": round(comp.ci_90[0], 1),
            "ci_hi": round(comp.ci_90[1], 1),
        })

    ci90 = result.bravs_ci_90
    return {
        "player_name": result.player.player_name,
        "season": result.player.season,
        "position": result.player.position,
        "team": result.player.team,
        "bravs": round(result.bravs, 1),
        "bravs_era_std": round(result.bravs_era_standardized, 1),
        "bravs_war_eq": round(result.bravs_calibrated, 1),
        "ci90_lo": round(ci90[0], 1),
        "ci90_hi": round(ci90[1], 1),
        "total_runs": round(result.total_runs_mean, 1),
        "rpw": round(result.rpw, 2),
        "leverage_mult": round(result.leverage_multiplier, 3),
        "components": components,
    }


# --- Preset player data for quick access ---
PRESETS = {
    "trout_2016": {
        "player_name": "Mike Trout", "season": 2016, "team": "LAA", "position": "CF",
        "pa": 681, "ab": 549, "hits": 173, "doubles": 24, "triples": 4, "hr": 29,
        "bb": 116, "ibb": 5, "hbp": 7, "k": 137, "sf": 5, "games": 159,
        "sb": 7, "cs": 3, "gidp": 14, "inn_fielded": 1320, "uzr": 0.4, "drs": -2, "oaa": 1,
        "park_factor": 0.98, "league_rpg": 4.48,
    },
    "ohtani_2023": {
        "player_name": "Shohei Ohtani", "season": 2023, "team": "LAA", "position": "DH",
        "pa": 599, "ab": 497, "hits": 151, "doubles": 26, "triples": 8, "hr": 44,
        "bb": 91, "ibb": 5, "hbp": 5, "k": 143, "sf": 6, "games": 135,
        "sb": 20, "cs": 6, "ip": 132, "er": 46, "hits_allowed": 99, "hr_allowed": 18,
        "bb_allowed": 55, "hbp_allowed": 5, "k_pitching": 167, "games_pitched": 23,
        "games_started": 23, "park_factor": 0.98, "league_rpg": 4.62,
    },
    "bonds_2004": {
        "player_name": "Barry Bonds", "season": 2004, "team": "SF", "position": "LF",
        "pa": 617, "ab": 373, "hits": 135, "doubles": 27, "triples": 3, "hr": 45,
        "bb": 232, "ibb": 120, "hbp": 9, "k": 41, "sf": 3, "games": 147,
        "sb": 6, "cs": 1, "gidp": 5, "park_factor": 0.93, "league_rpg": 4.81,
    },
    "degrom_2018": {
        "player_name": "Jacob deGrom", "season": 2018, "team": "NYM", "position": "P",
        "ip": 217, "er": 48, "hits_allowed": 152, "hr_allowed": 10,
        "bb_allowed": 46, "hbp_allowed": 5, "k_pitching": 269, "games_pitched": 32,
        "games_started": 32, "park_factor": 0.95, "league_rpg": 4.45,
    },
    "pedro_1999": {
        "player_name": "Pedro Martinez", "season": 1999, "team": "BOS", "position": "P",
        "ip": 213.3, "er": 49, "hits_allowed": 160, "hr_allowed": 9,
        "bb_allowed": 37, "hbp_allowed": 9, "k_pitching": 313, "games_pitched": 31,
        "games_started": 31, "park_factor": 1.04, "league_rpg": 5.08,
    },
    "judge_2022": {
        "player_name": "Aaron Judge", "season": 2022, "team": "NYY", "position": "RF",
        "pa": 696, "ab": 570, "hits": 177, "doubles": 28, "triples": 0, "hr": 62,
        "bb": 111, "ibb": 19, "hbp": 6, "k": 175, "sf": 4, "games": 157,
        "sb": 16, "cs": 3, "uzr": 3.5, "drs": 5, "oaa": 6, "inn_fielded": 1200,
        "park_factor": 1.05, "league_rpg": 4.28,
    },
    "rivera_2004": {
        "player_name": "Mariano Rivera", "season": 2004, "team": "NYY", "position": "P",
        "ip": 78.7, "er": 16, "hits_allowed": 65, "hr_allowed": 4,
        "bb_allowed": 20, "hbp_allowed": 1, "k_pitching": 66, "games_pitched": 74,
        "games_started": 0, "saves": 53, "avg_leverage_index": 1.85,
        "park_factor": 1.05, "league_rpg": 4.81,
    },
}


@app.route("/")
def index():
    return render_template("index.html", presets=PRESETS)


@app.route("/api/compute", methods=["POST"])
def compute():
    try:
        data = request.json or request.form.to_dict()
        player = _build_player_from_form(data)

        validation = validate_player_season(player)
        warnings = validation.warnings
        if not validation.is_valid:
            return jsonify({"error": "Invalid data", "details": validation.errors}), 400

        result = compute_bravs(player)
        resp = _result_to_dict(result)
        resp["warnings"] = warnings
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/preset/<key>")
def preset(key: str):
    if key not in PRESETS:
        return jsonify({"error": f"Unknown preset: {key}"}), 404
    return jsonify(PRESETS[key])


if __name__ == "__main__":
    print("Starting BRAVS Web App at http://localhost:5000")
    app.run(debug=True, port=5000)
