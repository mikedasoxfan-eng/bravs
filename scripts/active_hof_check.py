"""Which active players have already accumulated enough career value for the Hall of Fame?"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from bravs_engine import compute_bravs_fast

MLB_API = "https://statsapi.mlb.com/api/v1"
HEADSHOT = "https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/{}/headshot/67/current"

RPG = {
    2005: 4.59, 2006: 4.86, 2007: 4.80, 2008: 4.65, 2009: 4.61,
    2010: 4.38, 2011: 4.28, 2012: 4.32, 2013: 4.17, 2014: 4.07,
    2015: 4.25, 2016: 4.48, 2017: 4.65, 2018: 4.45, 2019: 4.83,
    2020: 4.65, 2021: 4.26, 2022: 4.28, 2023: 4.62, 2024: 4.52,
    2025: 4.45, 2026: 4.45,
}
SHORT_SEASONS = {2020: 60}

# Active veterans most likely to be HOF candidates
CANDIDATES = [
    (545361, "Mike Trout"),
    (660271, "Shohei Ohtani"),
    (605141, "Mookie Betts"),
    (592450, "Aaron Judge"),
    (571448, "Freddie Freeman"),
    (502110, "J.T. Realmuto"),
    (518626, "Jose Altuve"),
    (543829, "Nolan Arenado"),
    (621043, "Trea Turner"),
    (624413, "Jose Ramirez"),
    (543685, "Paul Goldschmidt"),
    (547180, "Bryce Harper"),
    (543037, "Manny Machado"),
    (608336, "Corey Seager"),
    (592178, "Max Scherzer"),
    (477132, "Clayton Kershaw"),
    (519317, "Giancarlo Stanton"),
    (596019, "Francisco Lindor"),
    (543294, "Matt Olson"),
    (666971, "Ronald Acuna Jr."),
    (665742, "Juan Soto"),
    (608369, "Xander Bogaerts"),
    (641355, "Corbin Burnes"),
    (669373, "Bobby Witt Jr."),
    (673540, "Julio Rodriguez"),
]


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


def compute_career(player_id, name):
    """Compute career BRAVS for a player across all MLB seasons."""
    # Get yearByYear hitting
    h_resp = requests.get(f"{MLB_API}/people/{player_id}/stats",
        params={"stats": "yearByYear", "group": "hitting", "gameType": "R"}, timeout=10)
    p_resp = requests.get(f"{MLB_API}/people/{player_id}/stats",
        params={"stats": "yearByYear", "group": "pitching", "gameType": "R"}, timeout=10)
    f_resp = requests.get(f"{MLB_API}/people/{player_id}/stats",
        params={"stats": "yearByYear", "group": "fielding", "gameType": "R"}, timeout=10)
    info_resp = requests.get(f"{MLB_API}/people/{player_id}", timeout=10)

    h_data = h_resp.json()
    p_data = p_resp.json()
    f_data = f_resp.json()
    info = info_resp.json().get("people", [{}])[0]

    birth = info.get("birthDate", "")
    age = 2026 - int(birth[:4]) if birth else 0
    primary_pos = info.get("primaryPosition", {}).get("abbreviation", "")

    # Build position map from fielding
    pos_map = {}
    if "stats" in f_data:
        for g in f_data["stats"]:
            for sp in g.get("splits", []):
                yr = int(sp.get("season", 0))
                pos = sp.get("position", {}).get("abbreviation", "")
                inn = float((sp.get("stat", {}).get("innings", "0") or "0").replace(",", ""))
                if pos in ("P", ""): continue
                if yr not in pos_map or inn > pos_map[yr][1]:
                    pos_map[yr] = (pos, inn)

    # Collect all MLB seasons
    seasons = {}
    if "stats" in h_data:
        for g in h_data["stats"]:
            for sp in g.get("splits", []):
                yr = int(sp.get("season", 0))
                if yr < 2000: continue
                s = sp.get("stat", {})
                pa = safe_int(s.get("plateAppearances"))
                if pa < 20: continue
                if safe_int(sp.get("numTeams", 1)) <= 1 and yr in seasons:
                    continue  # skip team splits if we already have data
                seasons[yr] = {"hitting": s, "pitching": None}

    if "stats" in p_data:
        for g in p_data["stats"]:
            for sp in g.get("splits", []):
                yr = int(sp.get("season", 0))
                if yr < 2000: continue
                s = sp.get("stat", {})
                ip = parse_ip(s.get("inningsPitched", 0))
                if ip < 5: continue
                if yr not in seasons:
                    seasons[yr] = {"hitting": None, "pitching": s}
                else:
                    seasons[yr]["pitching"] = s

    # Compute BRAVS for each season
    career_bravs = 0.0
    career_war = 0.0
    season_results = []
    peak_bravs = 0.0
    total_hr = 0
    total_games = 0

    for yr in sorted(seasons.keys()):
        h = seasons[yr].get("hitting") or {}
        p = seasons[yr].get("pitching") or {}

        pos, inn = pos_map.get(yr, (primary_pos or "DH", 0))
        if parse_ip(p.get("inningsPitched", 0)) > 40 and safe_int(h.get("plateAppearances", 0)) < 30:
            pos = "P"

        rpg = RPG.get(yr, 4.5)
        sg = SHORT_SEASONS.get(yr, 162)

        # Detect in-progress season
        games = max(safe_int(h.get("gamesPlayed")), safe_int(p.get("gamesPlayed")))
        if yr >= 2026 and games < 50:
            sg = max(games + 5, 10)

        r = compute_bravs_fast(
            pa=safe_int(h.get("plateAppearances")), ab=safe_int(h.get("atBats")),
            hits=safe_int(h.get("hits")), doubles=safe_int(h.get("doubles")),
            triples=safe_int(h.get("triples")), hr=safe_int(h.get("homeRuns")),
            bb=safe_int(h.get("baseOnBalls")), ibb=safe_int(h.get("intentionalWalks")),
            hbp=safe_int(h.get("hitByPitch")), k=safe_int(h.get("strikeOuts")),
            sf=safe_int(h.get("sacFlies")), sb=safe_int(h.get("stolenBases")),
            cs=safe_int(h.get("caughtStealing")), gidp=safe_int(h.get("groundIntoDoublePlay")),
            games=games,
            ip=parse_ip(p.get("inningsPitched")), er=safe_int(p.get("earnedRuns")),
            hits_allowed=safe_int(p.get("hits")), hr_allowed=safe_int(p.get("homeRuns")),
            bb_allowed=safe_int(p.get("baseOnBalls")), hbp_allowed=safe_int(p.get("hitBatsmen")),
            k_pitching=safe_int(p.get("strikeOuts")), games_pitched=safe_int(p.get("gamesPlayed")),
            games_started=safe_int(p.get("gamesStarted")), saves=safe_int(p.get("saves")),
            inn_fielded=inn, position=pos, season=yr,
            park_factor=1.0, league_rpg=rpg, season_games=sg, fast=True,
        )

        bravs = r["bravs"]
        war_eq = r["bravs_war_eq"]
        career_bravs += bravs
        career_war += war_eq
        peak_bravs = max(peak_bravs, bravs)
        total_hr += safe_int(h.get("homeRuns"))
        total_games += games
        season_results.append({"year": yr, "bravs": round(bravs, 1), "pos": pos, "g": games})

    # Peak 5 seasons
    top5 = sorted([s["bravs"] for s in season_results], reverse=True)[:5]
    peak5 = sum(top5)

    return {
        "name": name, "id": player_id, "age": age, "pos": primary_pos,
        "career_bravs": round(career_bravs, 1),
        "career_war": round(career_war, 1),
        "peak_bravs": round(peak_bravs, 1),
        "peak5": round(peak5, 1),
        "seasons": len(season_results),
        "total_hr": total_hr,
        "total_games": total_games,
        "season_results": season_results,
    }


def main():
    print("=" * 95)
    print("  ACTIVE PLAYERS: WHO'S ALREADY A HALL OF FAMER?")
    print("  Career BRAVS analysis using Rust engine")
    print("=" * 95)
    print()

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(compute_career, pid, name): (pid, name) for pid, name in CANDIDATES}
        for future in as_completed(futures):
            pid, name = futures[future]
            try:
                r = future.result()
                results.append(r)
                print(f"  Computed: {name} ({r['seasons']} seasons, {r['career_war']:.1f} WAR-eq)")
            except Exception as e:
                print(f"  Failed: {name}: {e}")

    results.sort(key=lambda x: x["career_war"], reverse=True)

    print(f"\n{'=' * 95}")
    print(f"  {'Rank':<5}{'Player':<22}{'Age':>4}{'Pos':<5}{'Yrs':>4}{'Games':>6}{'HR':>5}"
          f"{'BRAVS':>8}{'WAR-eq':>8}{'Peak':>7}{'Pk5':>7}  Verdict")
    print("-" * 95)

    for i, r in enumerate(results, 1):
        war = r["career_war"]
        if war >= 55:
            verdict = "LOCK"
        elif war >= 45:
            verdict = "STRONG"
        elif war >= 35:
            verdict = "BORDERLINE"
        elif war >= 25:
            verdict = "NEEDS MORE"
        else:
            verdict = "NOT YET"

        print(f"  {i:<5}{r['name']:<22}{r['age']:>4}{r['pos']:<5}{r['seasons']:>4}{r['total_games']:>6}"
              f"{r['total_hr']:>5}{r['career_bravs']:>8.1f}{r['career_war']:>8.1f}"
              f"{r['peak_bravs']:>7.1f}{r['peak5']:>7.1f}  {verdict}")

    # Tier breakdown
    locks = [r for r in results if r["career_war"] >= 55]
    strong = [r for r in results if 45 <= r["career_war"] < 55]
    border = [r for r in results if 35 <= r["career_war"] < 45]
    needs = [r for r in results if 25 <= r["career_war"] < 35]
    notyet = [r for r in results if r["career_war"] < 25]

    print(f"""
{'=' * 95}
  HOF TIERS (by career WAR-equivalent)
{'=' * 95}

  LOCK (55+ WAR-eq): {', '.join(r['name'] for r in locks) or 'none'}
    First-ballot certainties. Already accumulated enough value
    that the remaining career is just padding the margin.

  STRONG CASE (45-55 WAR-eq): {', '.join(r['name'] for r in strong) or 'none'}
    Very likely HOFers. A few more good seasons seals it.
    Even if they retired today, the case is compelling.

  BORDERLINE (35-45 WAR-eq): {', '.join(r['name'] for r in border) or 'none'}
    Could go either way. Need either a strong finish or
    memorable narrative moments (championships, milestones).

  NEEDS MORE (25-35 WAR-eq): {', '.join(r['name'] for r in needs) or 'none'}
    On the trajectory but not there yet. Need 3-5 more
    productive seasons to build a real case.

  NOT YET (<25 WAR-eq): {', '.join(r['name'] for r in notyet) or 'none'}
    Too early in career or insufficient peak. Could still
    get there with sustained excellence.

  THRESHOLDS (approximate, from HOF classifier):
    55+ WAR-eq = clear first-ballot
    45-55 = strong case, likely in
    35-45 = borderline, needs narrative
    25-35 = Hall of Very Good
    <25 = not enough yet
""")


if __name__ == "__main__":
    main()
