"""Aging Cliff Analysis — which great players fell off fastest?

Compares the decline curves of legends to identify who aged
gracefully vs who hit a wall. Uses BRAVS to measure value at
each age, which captures the full picture (not just batting avg).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from bravs_engine import compute_bravs_fast

MLB_API = "https://statsapi.mlb.com/api/v1"

RPG = {
    2000: 5.14, 2001: 4.78, 2002: 4.62, 2003: 4.73, 2004: 4.81,
    2005: 4.59, 2006: 4.86, 2007: 4.80, 2008: 4.65, 2009: 4.61,
    2010: 4.38, 2011: 4.28, 2012: 4.32, 2013: 4.17, 2014: 4.07,
    2015: 4.25, 2016: 4.48, 2017: 4.65, 2018: 4.45, 2019: 4.83,
    2020: 4.65, 2021: 4.26, 2022: 4.28, 2023: 4.62, 2024: 4.52, 2025: 4.45,
}
SHORT = {2020: 60}


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


def get_career_by_age(player_id, name):
    """Get BRAVS by age for a player."""
    info = requests.get(f"{MLB_API}/people/{player_id}", timeout=10).json()
    pinfo = info.get("people", [{}])[0]
    birth = pinfo.get("birthDate", "")
    if not birth: return None
    birth_year = int(birth[:4])
    pos = pinfo.get("primaryPosition", {}).get("abbreviation", "DH")

    h_data = requests.get(f"{MLB_API}/people/{player_id}/stats",
        params={"stats": "yearByYear", "group": "hitting", "gameType": "R"}, timeout=10).json()

    results = []
    if "stats" in h_data:
        for g in h_data["stats"]:
            for sp in g.get("splits", []):
                yr = int(sp.get("season", 0))
                s = sp.get("stat", {})
                pa = safe_int(s.get("plateAppearances"))
                if pa < 50 or safe_int(sp.get("numTeams", 1)) > 1:
                    if safe_int(sp.get("numTeams", 1)) <= 1: continue
                age = yr - birth_year

                r = compute_bravs_fast(
                    pa=pa, ab=safe_int(s.get("atBats")),
                    hits=safe_int(s.get("hits")), doubles=safe_int(s.get("doubles")),
                    triples=safe_int(s.get("triples")), hr=safe_int(s.get("homeRuns")),
                    bb=safe_int(s.get("baseOnBalls")), ibb=safe_int(s.get("intentionalWalks")),
                    hbp=safe_int(s.get("hitByPitch")), k=safe_int(s.get("strikeOuts")),
                    sf=safe_int(s.get("sacFlies")), sb=safe_int(s.get("stolenBases")),
                    cs=safe_int(s.get("caughtStealing")), games=safe_int(s.get("gamesPlayed")),
                    position=pos if pos != "P" else "DH", season=yr,
                    park_factor=1.0, league_rpg=RPG.get(yr, 4.5),
                    season_games=SHORT.get(yr, 162), fast=True,
                )
                results.append({"age": age, "year": yr, "bravs": r["bravs"],
                                "war_eq": r["bravs_war_eq"], "pa": pa,
                                "hr": safe_int(s.get("homeRuns")),
                                "avg": s.get("avg", "")})

    results.sort(key=lambda x: x["age"])
    return {"name": name, "id": player_id, "pos": pos, "seasons": results}


PLAYERS = [
    (545361, "Mike Trout"),
    (547180, "Bryce Harper"),
    (502110, "J.T. Realmuto"),
    (518626, "Jose Altuve"),
    (543829, "Nolan Arenado"),
    (543685, "Paul Goldschmidt"),
    (571448, "Freddie Freeman"),
    (519317, "Giancarlo Stanton"),
    (592450, "Aaron Judge"),
    (605141, "Mookie Betts"),
    (543037, "Manny Machado"),
    (471083, "Miguel Cabrera"),
    (425772, "Albert Pujols"),
    (400085, "Ryan Howard"),
    (120074, "David Ortiz"),
    (150029, "Ichiro Suzuki"),
]


def main():
    print("=" * 90)
    print("  AGING CLIFF ANALYSIS: WHO FELL OFF AND WHO AGED GRACEFULLY?")
    print("=" * 90)

    all_results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(get_career_by_age, pid, name): name for pid, name in PLAYERS}
        for future in as_completed(futures):
            name = futures[future]
            try:
                r = future.result()
                if r and r["seasons"]:
                    all_results.append(r)
                    print(f"  Loaded {name}: {len(r['seasons'])} seasons")
            except Exception as e:
                print(f"  Failed {name}: {e}")

    # For each player, find their peak and measure the decline
    print(f"\n{'=' * 90}")
    print(f"  DECLINE RATES FROM PEAK")
    print(f"{'=' * 90}\n")

    decline_data = []
    for p in all_results:
        seasons = p["seasons"]
        if len(seasons) < 4: continue

        # Find peak (highest BRAVS season)
        peak = max(seasons, key=lambda s: s["bravs"])
        peak_age = peak["age"]
        peak_bravs = peak["bravs"]

        # Post-peak seasons
        post_peak = [s for s in seasons if s["age"] > peak_age]
        if len(post_peak) < 2: continue

        # Average BRAVS in years 1-3 after peak and 4-6 after peak
        early_post = [s["bravs"] for s in post_peak[:3]]
        late_post = [s["bravs"] for s in post_peak[3:6]]

        early_avg = sum(early_post) / len(early_post) if early_post else 0
        late_avg = sum(late_post) / len(late_post) if late_post else 0

        # Decline rate: how fast did they fall?
        if peak_bravs > 0:
            early_retention = early_avg / peak_bravs * 100
            late_retention = late_avg / peak_bravs * 100 if late_post else None
        else:
            early_retention = 0
            late_retention = None

        # Did they hit a cliff? (> 60% drop in any consecutive 2-year span)
        cliff_age = None
        for i in range(len(seasons) - 1):
            if seasons[i]["age"] >= peak_age and seasons[i]["bravs"] > 2:
                if i + 1 < len(seasons) and seasons[i + 1]["bravs"] < seasons[i]["bravs"] * 0.3:
                    cliff_age = seasons[i + 1]["age"]
                    break

        decline_data.append({
            "name": p["name"], "peak_age": peak_age, "peak_bravs": peak_bravs,
            "early_retention": early_retention, "late_retention": late_retention,
            "cliff_age": cliff_age, "post_peak_years": len(post_peak),
            "seasons": seasons,
        })

    # Sort by early retention (graceful agers first)
    decline_data.sort(key=lambda x: x["early_retention"], reverse=True)

    print(f"  {'Player':<22}{'Peak':>5}{'PkAge':>6}{'Yrs 1-3':>9}{'Yrs 4-6':>9}{'Cliff':>7}  Aging")
    print("  " + "-" * 70)

    for d in decline_data:
        early = f"{d['early_retention']:.0f}%" if d["early_retention"] else "--"
        late = f"{d['late_retention']:.0f}%" if d["late_retention"] is not None else "--"
        cliff = str(d["cliff_age"]) if d["cliff_age"] else "none"

        if d["early_retention"] >= 60:
            aging = "GRACEFUL"
        elif d["early_retention"] >= 35:
            aging = "NORMAL"
        elif d["early_retention"] >= 10:
            aging = "STEEP"
        else:
            aging = "CLIFF"

        print(f"  {d['name']:<22}{d['peak_bravs']:>5.1f}{d['peak_age']:>6}"
              f"{early:>9}{late:>9}{cliff:>7}  {aging}")

    # Show career curves for the most interesting cases
    print(f"\n{'=' * 90}")
    print(f"  CAREER CURVES (BRAVS by age)")
    print(f"{'=' * 90}")

    for d in decline_data[:6]:
        print(f"\n  {d['name']} (peak age {d['peak_age']}, {d['peak_bravs']:.1f} BRAVS)")
        for s in d["seasons"]:
            bar_len = max(int(s["bravs"] / 1.0), 0) if s["bravs"] > 0 else 0
            neg_bar = abs(int(s["bravs"] / 1.0)) if s["bravs"] < 0 else 0
            bar = "#" * bar_len if bar_len else "-" * neg_bar
            marker = " <<< PEAK" if s["age"] == d["peak_age"] else ""
            color = "" if s["bravs"] >= 0 else "!"
            print(f"    {s['age']:>3}: {s['bravs']:>6.1f}  {color}{bar}{marker}")

    print(f"""
{'=' * 90}
  KEY FINDINGS
{'=' * 90}

  GRACEFUL AGERS maintain 60%+ of peak value for 3 years post-peak.
  These are typically players whose value comes from plate discipline
  and power (skills that age well) rather than speed and defense
  (skills that decline sharply).

  CLIFF FALLERS lose 70%+ of value in a 1-2 year span. Common
  causes: catastrophic injury, sudden loss of bat speed, or
  position change (from premium defense to DH, losing positional
  value instantly).

  BRAVS captures decline more completely than batting average because
  it includes defensive regression, reduced baserunning, and the
  durability penalty for fewer games played. A player hitting .280
  but only playing 100 games with poor defense may show a steeper
  BRAVS decline than their batting line suggests.
""")


if __name__ == "__main__":
    main()
