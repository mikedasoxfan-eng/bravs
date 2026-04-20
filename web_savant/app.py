"""Savant-style baseball analytics web app.

Multi-page Flask app with Player, Compare, Leaderboard, Year-over-Year,
Stat Filter, Season Counter, and Team views. Default port 5055.
"""

from __future__ import annotations

import os
import sys
from urllib.parse import urlencode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, jsonify, redirect, render_template, request, url_for

from web_savant import data as D
from web_savant import glossary as G
from web_savant import leaderboard as LB
from web_savant import percentiles as P


app = Flask(__name__)


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _int(s, default=None):
    try:
        return int(s)
    except (ValueError, TypeError):
        return default


YEARS = list(range(2025, 1899, -1))
MODERN_YEARS = list(range(2025, 2014, -1))


# -----------------------------------------------------------------
# Root / player
# -----------------------------------------------------------------

@app.route("/")
def root():
    return redirect(url_for("player_home"))


@app.route("/player")
def player_home():
    return render_template("player.html", active="player", years=YEARS,
                           card=None, player=None)


@app.route("/player/<int:mlbam>/<int:year>")
def player_view(mlbam: int, year: int):
    card = P.player_card(mlbam, year)
    card["portrait"] = D.portrait_url(mlbam)
    all_years = P.get_years_for_player(mlbam)
    career = _career_arc(mlbam, card["kind"])
    lahman_id = D.mlb_to_lahman().get(mlbam)
    awards = D.career_awards(lahman_id) if lahman_id else {}
    return render_template("player.html", active="player", years=YEARS,
                           card=card, player={"id": mlbam, "year": year},
                           all_years=all_years, career=career, awards=awards)


def _career_arc(mlbam: int, kind: str) -> list[dict]:
    """Year-over-year BRAVS + key rate for the player, for sparklines."""
    df = D.master(kind)
    sub = df[df["mlbam_id"] == mlbam].sort_values("yearID")
    if sub.empty:
        return []
    stats = ["bravs", "xwOBA", "AVG", "OPS"] if kind == "batter" else ["bravs", "ERA", "xERA", "K_pct"]
    rows = []
    for stat_key in stats:
        if stat_key not in sub.columns:
            continue
        series = []
        for _, r in sub.iterrows():
            v = r.get(stat_key)
            series.append({"year": int(r["yearID"]), "v": (None if v != v else float(v))})
        s = D.stat_by_key(kind, stat_key)
        rows.append({
            "key": stat_key,
            "label": s.label if s else stat_key,
            "series": series,
            "latest": next((p["v"] for p in reversed(series) if p["v"] is not None), None),
            "fmt": s.fmt if s else "{:.2f}",
        })
    return rows


# -----------------------------------------------------------------
# Search
# -----------------------------------------------------------------

@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    results = P.search_players(q, limit=15)
    return jsonify(results)


# -----------------------------------------------------------------
# Compare
# -----------------------------------------------------------------

@app.route("/compare")
def compare_home():
    ids_param = request.args.get("ids", "")
    year = _int(request.args.get("year"), 2024)
    mode = request.args.get("mode", "season")  # "season" or "career"
    kind_override = request.args.get("kind")
    ids = [int(x) for x in ids_param.split(",") if x.strip().isdigit()]

    players = []
    for pid in ids[:5]:
        try:
            # For career mode, still classify using latest year
            latest = max(P.get_years_for_player(pid) or [year])
            card = P.player_card(pid, year if mode == "season" else latest)
            card["portrait"] = D.portrait_url(pid)
            card["playerID"] = D.mlb_to_lahman().get(pid)
            players.append(card)
        except Exception:
            pass

    kind = kind_override or (players[0]["kind"] if players else "batter")
    stats = _compare_stats(kind)
    matrix: dict[int, dict] = {}
    awards_by_pid: dict[int, dict[str, int]] = {}

    if players:
        df = D.master(kind)
        for p in players:
            pid = p["mlbam_id"]
            if mode == "career" and p.get("playerID"):
                career_stats = (D.career_totals_batter(p["playerID"]) if kind == "batter"
                                else D.career_totals_pitcher(p["playerID"]))
                player_stats = {}
                for st in stats:
                    v = career_stats.get(st.key)
                    player_stats[st.key] = {
                        "raw": None if v is None else float(v),
                        "fmt": D.fmt_value(st, v) if v is not None else "—",
                    }
                matrix[pid] = player_stats
                awards_by_pid[pid] = D.career_awards(p["playerID"])
            else:
                row = df[(df["mlbam_id"] == pid) & (df["yearID"] == year)]
                if row.empty: continue
                row = row.iloc[0]
                player_stats = {}
                for st in stats:
                    if st.key in row.index:
                        v = row[st.key]
                        player_stats[st.key] = {
                            "raw": None if (v != v) else float(v),
                            "fmt": D.fmt_value(st, v),
                        }
                matrix[pid] = player_stats

    # Identify per-stat leaders
    leaders: dict[str, int] = {}
    for st in stats:
        vals = [(pid, d.get(st.key, {}).get("raw")) for pid, d in matrix.items()]
        vals = [(pid, v) for pid, v in vals if v is not None]
        if len(vals) < 2: continue
        if st.direction == "high":
            pid_lead = max(vals, key=lambda t: t[1])[0]
        else:
            pid_lead = min(vals, key=lambda t: t[1])[0]
        leaders[st.key] = pid_lead

    # Stat groups for layout
    groups: dict[str, list] = {}
    for st in stats:
        groups.setdefault(st.group, []).append(st)

    return render_template("compare.html", active="compare",
                           years=MODERN_YEARS, players=players, year=year,
                           mode=mode, kind=kind, stats=stats, groups=groups,
                           matrix=matrix, leaders=leaders,
                           awards_by_pid=awards_by_pid)


def _compare_stats(kind: str) -> list[D.Stat]:
    # A curated subset for the compare matrix
    keys = (["G", "PA", "AVG", "OBP", "SLG", "OPS", "HR", "RBI", "SB",
             "K_pct", "BB_pct", "xwOBA", "xBA", "xSLG",
             "ev_avg", "barrel_pct", "hardhit_pct", "sprint_speed", "OAA",
             "bravs", "bravs_war_eq", "hitting_runs", "fielding_runs"]
            if kind == "batter" else
            ["G", "GS", "IP", "W", "L", "SV", "ERA", "WHIP", "FIP",
             "K9", "BB9", "K_pct", "BB_pct", "K_BB_pct",
             "xERA", "xwOBA", "velo",
             "bravs", "bravs_war_eq", "pitching_runs"])
    return [D.stat_by_key(kind, k) for k in keys if D.stat_by_key(kind, k)]


# -----------------------------------------------------------------
# Leaderboard
# -----------------------------------------------------------------

@app.route("/leaderboard")
def leaderboard_page():
    kind = request.args.get("kind", "batter")
    stat_key = request.args.get("stat", "bravs")
    year_from = _int(request.args.get("year_from"), 2024)
    year_to = _int(request.args.get("year_to"), year_from)
    n = _int(request.args.get("n"), 20)
    direction_override = request.args.get("asc", "") == "1"
    rows = LB.leaderboard(kind, stat_key, year_from, year_to, n, asc=direction_override)
    return render_template("leaderboard.html", active="leaderboard", years=YEARS,
                           kind=kind, stat_key=stat_key, year_from=year_from,
                           year_to=year_to, n=n, rows=rows,
                           stats=D.stats_for(kind),
                           stat=D.stat_by_key(kind, stat_key))


# -----------------------------------------------------------------
# YoY
# -----------------------------------------------------------------

@app.route("/yoy")
def yoy_page():
    kind = request.args.get("kind", "batter")
    stat_key = request.args.get("stat", "bravs")
    year_a = _int(request.args.get("year_a"), 2023)
    year_b = _int(request.args.get("year_b"), 2024)
    improvers = request.args.get("mode", "improvers") == "improvers"
    n = _int(request.args.get("n"), 15)
    rows = LB.yoy_delta(kind, stat_key, year_a, year_b, n, improvers=improvers)
    return render_template("yoy.html", active="yoy", years=YEARS,
                           kind=kind, stat_key=stat_key, year_a=year_a, year_b=year_b,
                           improvers=improvers, n=n, rows=rows,
                           stats=D.stats_for(kind),
                           stat=D.stat_by_key(kind, stat_key))


# -----------------------------------------------------------------
# Stat Filter
# -----------------------------------------------------------------

@app.route("/filter")
def filter_page():
    kind = request.args.get("kind", "batter")
    year_from = _int(request.args.get("year_from"), 2024)
    year_to = _int(request.args.get("year_to"), year_from)
    n = _int(request.args.get("n"), 30)
    sort_key = request.args.get("sort", "bravs")
    filters = []
    for i in range(1, 5):
        stat = request.args.get(f"stat{i}", "")
        op = request.args.get(f"op{i}", ">=")
        val = request.args.get(f"val{i}", "")
        if stat and val not in ("", None):
            filters.append({"stat": stat, "op": op, "value": val})
    rows = LB.stat_filter(kind, year_from, year_to, filters, n, sort_key) if filters else []
    return render_template("filter.html", active="filter", years=YEARS,
                           kind=kind, year_from=year_from, year_to=year_to,
                           n=n, sort_key=sort_key, filters=filters, rows=rows,
                           stats=D.stats_for(kind))


# -----------------------------------------------------------------
# Season Counter
# -----------------------------------------------------------------

@app.route("/counter")
def counter_page():
    kind = request.args.get("kind", "batter")
    year_from = _int(request.args.get("year_from"), 1900)
    year_to = _int(request.args.get("year_to"), 2025)
    n = _int(request.args.get("n"), 25)
    filters = []
    for i in range(1, 5):
        stat = request.args.get(f"stat{i}", "")
        op = request.args.get(f"op{i}", ">=")
        val = request.args.get(f"val{i}", "")
        if stat and val not in ("", None):
            filters.append({"stat": stat, "op": op, "value": val})
    rows = LB.season_counter(kind, filters, year_from, year_to, n) if filters else []
    return render_template("counter.html", active="counter", years=YEARS,
                           kind=kind, year_from=year_from, year_to=year_to,
                           n=n, filters=filters, rows=rows,
                           stats=D.stats_for(kind))


# -----------------------------------------------------------------
# League Leaders — top N in up to 10 stats at once
# -----------------------------------------------------------------

@app.route("/leaders")
def leaders_page():
    kind = request.args.get("kind", "batter")
    year = _int(request.args.get("year"), 2024)
    n = _int(request.args.get("n"), 5)
    mode = request.args.get("mode", "top")  # "top" or "bottom"
    stat_keys_str = request.args.get("stats", "")
    stat_keys = [s for s in stat_keys_str.split(",") if s]
    if not stat_keys:
        # Preset default
        stat_keys = (["bravs", "OPS", "HR", "AVG", "xwOBA", "barrel_pct"]
                     if kind == "batter"
                     else ["bravs", "ERA", "K9", "xERA", "WHIP", "velo"])

    leaderboards = []
    for sk in stat_keys[:10]:
        st = D.stat_by_key(kind, sk)
        if not st: continue
        asc = (st.direction == "low") if mode == "top" else (st.direction == "high")
        rows = LB.leaderboard(kind, sk, year, year, n, asc=asc)
        for r in rows:
            r["portrait"] = D.portrait_url(r["mlbam_id"])
        leaderboards.append({"stat": st, "rows": rows})

    return render_template("leaders.html", active="leaders", years=YEARS,
                           kind=kind, year=year, n=n, mode=mode,
                           leaderboards=leaderboards,
                           selected_keys=stat_keys,
                           stats=D.stats_for(kind))


# -----------------------------------------------------------------
# Team
# -----------------------------------------------------------------

@app.route("/team")
def team_home():
    year = _int(request.args.get("year"), 2024)
    team_id = request.args.get("team")
    kind = request.args.get("kind", "batter")
    view = request.args.get("view", "roster")  # "roster" | "savant"
    teams = LB.team_list(year)
    roster = LB.team_roster(team_id, year, kind) if team_id else []
    team_savant = (_team_savant_rows(team_id, year, kind)
                   if team_id and view == "savant" else [])
    colors = D.team_colors(team_id)
    return render_template("team.html", active="team", years=YEARS,
                           teams=teams, year=year, team_id=team_id, view=view,
                           roster=roster, kind=kind,
                           team_savant=team_savant, colors=colors)


def _team_savant_rows(team_id: str, year: int, kind: str) -> list[dict]:
    """For each stat, find the team's leader with raw value + league percentile."""
    import numpy as np
    df = D.master(kind)
    team_df = df[(df["yearID"] == year) & (df["teamID"] == team_id)]
    if team_df.empty:
        return []
    # Minimum playing time for inclusion
    if kind == "batter":
        team_df = team_df[team_df["PA"] >= 50]
    else:
        team_df = team_df[team_df["IP"] >= 10]
    league = df[df["yearID"] == year]
    if kind == "batter":
        league = league[league["PA"] >= 300]
    else:
        league = league[league["IP"] >= 40]

    # Stats to display — skip raw counting; keep Statcast-heavy + rate
    display_keys = (["xwOBA", "xBA", "xSLG", "ev_avg", "barrel_pct", "hardhit_pct",
                     "sweetspot_pct", "AVG", "OBP", "SLG", "OPS", "K_pct", "BB_pct",
                     "sprint_speed", "OAA", "bravs"]
                    if kind == "batter" else
                    ["xERA", "xwOBA", "xBA", "xSLG", "ERA", "WHIP", "FIP",
                     "K9", "BB9", "K_pct", "BB_pct", "velo", "bravs"])

    rows = []
    for key in display_keys:
        st = D.stat_by_key(kind, key)
        if not st or key not in team_df.columns:
            continue
        sub = team_df.dropna(subset=[key])
        if sub.empty:
            continue
        asc = st.direction == "low"
        sub = sub.sort_values(key, ascending=asc)
        leader = sub.iloc[0]
        raw = float(leader[key])
        # League percentile
        pool = league.dropna(subset=[key])[key]
        if len(pool) < 10:
            pct = None
        elif st.direction == "high":
            pct = int(round((pool < raw).mean() * 100))
        else:
            pct = int(round((pool > raw).mean() * 100))
        mid = int(leader["mlbam_id"]) if not np.isnan(leader.get("mlbam_id", np.nan)) else None
        rows.append({
            "stat": st,
            "leader_name": leader.get("full_name") or leader["playerID"],
            "leader_mlbam": mid,
            "portrait": D.portrait_url(mid),
            "value_fmt": D.fmt_value(st, raw),
            "percentile": pct,
        })
    return rows


# -----------------------------------------------------------------
# Profile Card (compact share-friendly snapshot)
# -----------------------------------------------------------------

@app.route("/profile")
def profile_home():
    return render_template("profile.html", active="profile", years=YEARS,
                           card=None, player=None)


@app.route("/profile/<int:mlbam>/<int:year>")
def profile_view(mlbam: int, year: int):
    lahman_id = D.mlb_to_lahman().get(mlbam)
    if not lahman_id:
        return render_template("profile.html", active="profile", years=YEARS,
                               card=None, player={"id": mlbam, "year": year}, error="Player not found")
    # Detect kind
    bat = D.batter_master()
    pit = D.pitcher_master()
    brow = bat[(bat["playerID"] == lahman_id) & (bat["yearID"] == year)]
    prow = pit[(pit["playerID"] == lahman_id) & (pit["yearID"] == year)]
    ip = float(prow.iloc[0]["IP"]) if not prow.empty else 0
    pa = float(brow.iloc[0]["PA"]) if not brow.empty else 0
    kind = "pitcher" if (ip > 20 and pa < 300) else "batter"
    row = (prow.iloc[0] if kind == "pitcher" else brow.iloc[0]) if (not prow.empty or not brow.empty) else None
    if row is None:
        return render_template("profile.html", active="profile", years=YEARS,
                               card=None, player={"id": mlbam, "year": year}, error="No season data")
    team_id = row.get("teamID")
    colors = D.team_colors(team_id)
    age = D.player_age(lahman_id, year)
    name = row.get("full_name") or lahman_id

    # Pull the stats block we want to display
    if kind == "batter":
        lines = [
            [("Games", _fmt_int(row.get("G"))), ("PA", _fmt_int(row.get("PA")))],
            [("AVG", _fmt_3(row.get("AVG"))), ("OBP", _fmt_3(row.get("OBP"))), ("SLG", _fmt_3(row.get("SLG")))],
            [("OPS", _fmt_3(row.get("OPS"))), ("HR", _fmt_int(row.get("HR"))), ("RBI", _fmt_int(row.get("RBI")))],
            [("BB%", _fmt_1(row.get("BB_pct"))), ("K%", _fmt_1(row.get("K_pct"))), ("SB", _fmt_int(row.get("SB")))],
        ]
    else:
        lines = [
            [("G", _fmt_int(row.get("G"))), ("GS", _fmt_int(row.get("GS"))), ("SV", _fmt_int(row.get("SV")))],
            [("ERA", _fmt_2(row.get("ERA"))), ("IP", _fmt_1(row.get("IP"))),
             ("W-L", f"{_fmt_int(row.get('W'))}-{_fmt_int(row.get('L'))}")],
            [("K%", _fmt_1(row.get("K_pct"))), ("BB%", _fmt_1(row.get("BB_pct"))),
             ("K/BB", _fmt_2(row.get("SO")/row.get("BB")) if row.get("BB") else "—")],
            [("WHIP", _fmt_2(row.get("WHIP"))), ("HR/9", _fmt_2(row.get("HR9")))],
        ]

    card = {
        "name": name,
        "kind": kind,
        "year": year,
        "age": age,
        "team_id": team_id,
        "team_name": colors["name"],
        "team_primary": colors["primary"],
        "team_secondary": colors["secondary"],
        "portrait": D.portrait_url(mlbam),
        "stat_lines": lines,
    }
    return render_template("profile.html", active="profile", years=YEARS,
                           card=card, player={"id": mlbam, "year": year}, error=None)


def _fmt_int(v):
    try:
        if v != v: return "—"
        return f"{int(v)}"
    except Exception:
        return "—"


def _fmt_1(v):
    try:
        if v != v: return "—"
        return f"{float(v):.1f}"
    except Exception:
        return "—"


def _fmt_2(v):
    try:
        if v != v: return "—"
        return f"{float(v):.2f}"
    except Exception:
        return "—"


def _fmt_3(v):
    try:
        if v != v: return "—"
        return f"{float(v):.3f}".lstrip("0")
    except Exception:
        return "—"


# -----------------------------------------------------------------
# Legacy API (used by search typeahead)
# -----------------------------------------------------------------

@app.route("/api/player/<int:mlbam>/<int:year>")
def api_player(mlbam: int, year: int):
    return jsonify(P.player_card(mlbam, year))


# -----------------------------------------------------------------
# Glossary
# -----------------------------------------------------------------

@app.route("/glossary")
def glossary_page():
    q = request.args.get("q", "").strip()
    cat = request.args.get("cat", "")
    entries = G.all_entries()
    if q:
        ql = q.lower()
        entries = [e for e in entries
                   if ql in e.label.lower()
                   or ql in e.abbrev.lower()
                   or ql in e.definition.lower()
                   or any(ql in a.lower() for a in e.aliases)]
    if cat:
        entries = [e for e in entries if e.category == cat]

    # Group by category for display
    grouped: dict[str, list] = {}
    for e in entries:
        grouped.setdefault(e.category, []).append(e)

    return render_template("glossary.html", active="glossary",
                           grouped=grouped, total=G.total(),
                           shown=len(entries), q=q, cat=cat,
                           all_categories=G.categories_with_counts())


if __name__ == "__main__":
    port = _int(os.environ.get("PORT"), 5055)
    app.run(host="127.0.0.1", port=port, debug=True)
