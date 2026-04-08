"""Live 2026 MVP Race — who's leading right now according to BRAVS?"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from bravs_engine import compute_bravs_fast

MLB_API = "https://statsapi.mlb.com/api/v1"


def safe_int(v, d=0):
    try: return int(v or d)
    except: return d


def parse_ip(v):
    if not v: return 0.0
    s = str(v)
    if "." in s:
        parts = s.split(".")
        return int(parts[0]) + int(parts[1] or 0) / 3.0
    return float(s)


def fetch_and_compute(player_id, name, season=2026):
    """Fetch stats for a player and compute BRAVS."""
    # Hitting
    h_resp = requests.get(f"{MLB_API}/people/{player_id}/stats",
        params={"stats": "season", "season": season, "group": "hitting", "gameType": "R"}, timeout=10)
    h_data = h_resp.json()

    # Pitching
    p_resp = requests.get(f"{MLB_API}/people/{player_id}/stats",
        params={"stats": "season", "season": season, "group": "pitching", "gameType": "R"}, timeout=10)
    p_data = p_resp.json()

    # Fielding for position
    f_resp = requests.get(f"{MLB_API}/people/{player_id}/stats",
        params={"stats": "season", "season": season, "group": "fielding", "gameType": "R"}, timeout=10)
    f_data = f_resp.json()

    pos = "DH"
    inn = 0.0
    if "stats" in f_data:
        for g in f_data["stats"]:
            for sp in g.get("splits", []):
                p_abbr = sp.get("position", {}).get("abbreviation", "")
                p_inn = float((sp.get("stat", {}).get("innings", "0") or "0").replace(",", ""))
                if p_abbr not in ("P", "") and p_inn > inn:
                    pos, inn = p_abbr, p_inn

    # Parse hitting (use total row if traded)
    h = {}
    team = ""
    if "stats" in h_data:
        for g in h_data["stats"]:
            splits = g.get("splits", [])
            chosen = None
            for sp in splits:
                if safe_int(sp.get("numTeams", 1)) > 1:
                    chosen = sp; break
            if not chosen and splits:
                chosen = splits[0] if len(splits) == 1 else max(splits, key=lambda x: safe_int(x.get("stat", {}).get("plateAppearances")))
            if chosen:
                s = chosen.get("stat", {})
                team = chosen.get("team", {}).get("name", "")
                h = {k: safe_int(s.get(k)) for k in [
                    "gamesPlayed", "plateAppearances", "atBats", "hits", "doubles",
                    "triples", "homeRuns", "baseOnBalls", "intentionalWalks",
                    "hitByPitch", "strikeOuts", "sacFlies", "stolenBases",
                    "caughtStealing", "groundIntoDoublePlay"]}
                h["avg"] = s.get("avg", "")
                h["ops"] = s.get("ops", "")

    # Parse pitching
    p = {}
    if "stats" in p_data:
        for g in p_data["stats"]:
            splits = g.get("splits", [])
            chosen = None
            for sp in splits:
                if safe_int(sp.get("numTeams", 1)) > 1:
                    chosen = sp; break
            if not chosen and splits:
                chosen = splits[0] if len(splits) == 1 else max(splits, key=lambda x: parse_ip(x.get("stat", {}).get("inningsPitched")))
            if chosen:
                s = chosen.get("stat", {})
                if not team:
                    team = chosen.get("team", {}).get("name", "")
                p["ip"] = parse_ip(s.get("inningsPitched"))
                for k in ["gamesPlayed", "gamesStarted", "earnedRuns", "hits",
                           "homeRuns", "baseOnBalls", "hitBatsmen", "strikeOuts", "saves"]:
                    p[k] = safe_int(s.get(k))
                p["era"] = s.get("era", "")

    if not h.get("plateAppearances") and not p.get("ip"):
        return None

    # For in-progress seasons, set season_games to the max games any player
    # has played so far. This prorates the durability expectation correctly
    # so early-season snapshots aren't penalized for games not yet played.
    games_played = max(h.get("gamesPlayed", 0), p.get("gamesPlayed", 0))
    # Estimate team games played as slightly more than the player's games
    team_games_est = min(games_played + 5, 162)

    r = compute_bravs_fast(
        pa=h.get("plateAppearances", 0), ab=h.get("atBats", 0),
        hits=h.get("hits", 0), doubles=h.get("doubles", 0),
        triples=h.get("triples", 0), hr=h.get("homeRuns", 0),
        bb=h.get("baseOnBalls", 0), ibb=h.get("intentionalWalks", 0),
        hbp=h.get("hitByPitch", 0), k=h.get("strikeOuts", 0),
        sf=h.get("sacFlies", 0), sb=h.get("stolenBases", 0),
        cs=h.get("caughtStealing", 0), gidp=h.get("groundIntoDoublePlay", 0),
        games=h.get("gamesPlayed", 0),
        ip=p.get("ip", 0.0), er=p.get("earnedRuns", 0),
        hits_allowed=p.get("hits", 0), hr_allowed=p.get("homeRuns", 0),
        bb_allowed=p.get("baseOnBalls", 0), hbp_allowed=p.get("hitBatsmen", 0),
        k_pitching=p.get("strikeOuts", 0), games_pitched=p.get("gamesPlayed", 0),
        games_started=p.get("gamesStarted", 0), saves=p.get("saves", 0),
        inn_fielded=inn, position=pos, season=season,
        park_factor=1.0, league_rpg=4.45, season_games=team_games_est, fast=True,
    )

    return {
        "name": name, "team": team, "pos": pos,
        "g": h.get("gamesPlayed", p.get("gamesPlayed", 0)),
        "pa": h.get("plateAppearances", 0),
        "avg": h.get("avg", ""), "hr": h.get("homeRuns", 0),
        "ops": h.get("ops", ""),
        "ip": p.get("ip", 0), "era": p.get("era", ""),
        "bravs": r["bravs"], "war_eq": r["bravs_war_eq"],
        "era_std": r["bravs_era_std"],
    }


def get_current_leaders(league_id, season=2026):
    """Get current stat leaders from MLB API."""
    categories = {
        "hitting": ["onBasePlusSlugging", "plateAppearances", "homeRuns"],
        "pitching": ["earnedRunAverage", "strikeouts", "inningsPitched"],
    }
    seen = set()
    players = []

    for group, cats in categories.items():
        for cat in cats:
            resp = requests.get(f"{MLB_API}/stats/leaders",
                params={"leaderCategories": cat, "season": season,
                        "leagueId": league_id, "limit": 8,
                        "statGroup": group}, timeout=10)
            data = resp.json()
            if "leagueLeaders" in data:
                for g in data["leagueLeaders"]:
                    for leader in g.get("leaders", []):
                        person = leader.get("person", {})
                        pid = person.get("id")
                        if pid and pid not in seen:
                            seen.add(pid)
                            players.append({"id": pid, "name": person.get("fullName", "?")})
    return players


def run_race(league_name, league_id, season=2026):
    """Run a full MVP race for a league."""
    print(f"\n  Fetching {league_name} leaders...")
    candidates = get_current_leaders(league_id, season)
    print(f"  Found {len(candidates)} candidates. Computing BRAVS...")

    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch_and_compute, c["id"], c["name"], season): c for c in candidates}
        for future in as_completed(futures):
            try:
                r = future.result()
                if r and r["bravs"] is not None:
                    results.append(r)
            except Exception as e:
                pass

    results.sort(key=lambda x: x["bravs"], reverse=True)

    print(f"\n  {'=' * 90}")
    print(f"  {season} {league_name} MVP RACE (as of today)")
    print(f"  {'=' * 90}")
    print(f"\n  {'Rank':<6}{'Player':<24}{'Team':<22}{'Pos':<5}{'G':>4}"
          f"{'BRAVS':>7}{'WAReq':>7}{'Key Stat':>14}")
    print(f"  {'-' * 90}")

    for i, r in enumerate(results[:15], 1):
        if r.get("ip", 0) > 20:
            key = f"  {r['era']} ERA" if r.get("era") else ""
        else:
            key = f"  {r['avg']}/{r['hr']} HR" if r.get("avg") else ""
        marker = " <<" if i == 1 else ""
        print(f"  {i:>3}.  {r['name']:<24}{r['team']:<22}{r['pos']:<5}{r['g']:>4}"
              f"{r['bravs']:>7.1f}{r['war_eq']:>7.1f}{key}{marker}")

    if results:
        leader = results[0]
        print(f"\n  Current BRAVS MVP: {leader['name']} ({leader['bravs']:.1f} BRAVS, {leader['war_eq']:.1f} WAR-eq)")

    return results


def main():
    print("=" * 94)
    print("  LIVE 2026 MVP RACE — WHO'S LEADING RIGHT NOW?")
    print("=" * 94)

    al_results = run_race("American League", 103)
    nl_results = run_race("National League", 104)

    # Combined leaderboard
    all_results = al_results + nl_results
    all_results.sort(key=lambda x: x["bravs"], reverse=True)

    print(f"\n  {'=' * 90}")
    print(f"  COMBINED MLB — TOP 10 MOST VALUABLE PLAYERS RIGHT NOW")
    print(f"  {'=' * 90}")
    print(f"\n  {'Rank':<6}{'Player':<24}{'Team':<22}{'Pos':<5}{'BRAVS':>7}{'WAReq':>7}")
    print(f"  {'-' * 70}")
    for i, r in enumerate(all_results[:10], 1):
        print(f"  {i:>3}.  {r['name']:<24}{r['team']:<22}{r['pos']:<5}"
              f"{r['bravs']:>7.1f}{r['war_eq']:>7.1f}")

    print()


if __name__ == "__main__":
    main()
