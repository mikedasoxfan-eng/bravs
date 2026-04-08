"""Is Giancarlo Stanton a Hall of Famer? BRAVS analysis."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from bravs_engine import compute_bravs_fast

MLB_API = "https://statsapi.mlb.com/api/v1"
STANTON_ID = 519317

RPG = {
    2010: 4.38, 2011: 4.28, 2012: 4.32, 2013: 4.17, 2014: 4.07,
    2015: 4.25, 2016: 4.48, 2017: 4.65, 2018: 4.45, 2019: 4.83,
    2020: 4.65, 2021: 4.26, 2022: 4.28, 2023: 4.62, 2024: 4.52,
    2025: 4.45, 2026: 4.45,
}
SEASON_LEN = {2020: 60}


def safe_int(v, d=0):
    try:
        return int(v or d)
    except (ValueError, TypeError):
        return d


def main():
    # Fetch hitting stats
    resp = requests.get(f"{MLB_API}/people/{STANTON_ID}/stats",
                        params={"stats": "yearByYear", "group": "hitting", "gameType": "R"}, timeout=10)
    data = resp.json()

    # Fetch fielding for position
    field_resp = requests.get(f"{MLB_API}/people/{STANTON_ID}/stats",
                              params={"stats": "yearByYear", "group": "fielding", "gameType": "R"}, timeout=10)
    field_data = field_resp.json()

    pos_map = {}
    if "stats" in field_data:
        for group in field_data["stats"]:
            for split in group.get("splits", []):
                yr = int(split.get("season", 0))
                pos = split.get("position", {}).get("abbreviation", "")
                inn = float((split.get("stat", {}).get("innings", "0") or "0").replace(",", ""))
                if pos in ("P", ""):
                    continue
                if yr not in pos_map or inn > pos_map[yr][1]:
                    pos_map[yr] = (pos, inn)

    seasons = []
    if "stats" in data:
        for group in data["stats"]:
            for split in group.get("splits", []):
                if split.get("sport", {}).get("abbreviation", "") != "MLB":
                    continue
                yr = int(split.get("season", 0))
                s = split.get("stat", {})
                pa = safe_int(s.get("plateAppearances"))
                if pa < 30:
                    continue

                pos, inn = pos_map.get(yr, ("DH", 0))

                r = compute_bravs_fast(
                    pa=pa,
                    ab=safe_int(s.get("atBats")),
                    hits=safe_int(s.get("hits")),
                    doubles=safe_int(s.get("doubles")),
                    triples=safe_int(s.get("triples")),
                    hr=safe_int(s.get("homeRuns")),
                    bb=safe_int(s.get("baseOnBalls")),
                    ibb=safe_int(s.get("intentionalWalks")),
                    hbp=safe_int(s.get("hitByPitch")),
                    k=safe_int(s.get("strikeOuts")),
                    sf=safe_int(s.get("sacFlies")),
                    sb=safe_int(s.get("stolenBases")),
                    cs=safe_int(s.get("caughtStealing")),
                    gidp=safe_int(s.get("groundIntoDoublePlay")),
                    games=safe_int(s.get("gamesPlayed")),
                    inn_fielded=inn,
                    position=pos,
                    season=yr,
                    park_factor=1.0,
                    league_rpg=RPG.get(yr, 4.5),
                    season_games=SEASON_LEN.get(yr, 162),
                    fast=True,
                )

                seasons.append({
                    "year": yr, "pos": pos, "g": safe_int(s.get("gamesPlayed")),
                    "pa": pa, "avg": s.get("avg", ""),
                    "hr": safe_int(s.get("homeRuns")),
                    "ops": s.get("ops", ""),
                    "bravs": r["bravs"], "era_std": r["bravs_era_std"],
                    "war_eq": r["bravs_war_eq"],
                })

    seasons.sort(key=lambda x: x["year"])

    career_bravs = sum(s["bravs"] for s in seasons)
    career_war = sum(s["war_eq"] for s in seasons)
    career_era = sum(s["era_std"] for s in seasons)
    total_hr = sum(s["hr"] for s in seasons)
    total_g = sum(s["g"] for s in seasons)
    peak_season = max(seasons, key=lambda s: s["bravs"])
    peak_5 = sum(sorted([s["bravs"] for s in seasons], reverse=True)[:5])

    print("=" * 85)
    print("  GIANCARLO STANTON: HALL OF FAME ANALYSIS")
    print("=" * 85)
    print()
    print(f"  Career: {len(seasons)} MLB seasons ({seasons[0]['year']}-{seasons[-1]['year']})")
    print(f"  Games: {total_g}   HR: {total_hr}")
    print()
    print(f"{'Year':<6}{'Pos':<5}{'G':>4}{'PA':>5}{'AVG':>6}{'HR':>4}{'OPS':>7}"
          f"{'BRAVS':>7}{'ErStd':>7}{'WAReq':>7}")
    print("-" * 85)

    for s in seasons:
        marker = "  <-- peak" if s["year"] == peak_season["year"] else ""
        print(f"{s['year']:<6}{s['pos']:<5}{s['g']:>4}{s['pa']:>5}{s['avg']:>6}"
              f"{s['hr']:>4}{s['ops']:>7}{s['bravs']:>7.1f}{s['era_std']:>7.1f}"
              f"{s['war_eq']:>7.1f}{marker}")

    print("-" * 85)
    print(f"{'TOTAL':<6}{'':>4}{total_g:>4}{'':>5}{'':>6}{total_hr:>4}{'':>7}"
          f"{career_bravs:>7.1f}{career_era:>7.1f}{career_war:>7.1f}")

    positive = [s for s in seasons if s["bravs"] > 3]
    negative = [s for s in seasons if s["bravs"] < 0]
    injured = [s for s in seasons if s["g"] < 100 and s["pa"] > 30]
    avg_games = total_g / len(seasons)
    peak3 = sorted([s["bravs"] for s in seasons], reverse=True)[:3]

    if career_war >= 50:
        verdict = "YES -- clear Hall of Famer"
    elif career_war >= 40:
        verdict = "BORDERLINE -- needs a strong final chapter or 500 HR"
    elif career_war >= 30:
        verdict = "PROBABLY NOT -- Hall of Very Good unless HR total sways voters"
    else:
        verdict = "NO -- below HOF threshold, though HR milestones may help with voters"

    print(f"""
{'=' * 85}
  THE VERDICT
{'=' * 85}

  Career BRAVS:     {career_bravs:.1f}
  Career WAR-eq:    {career_war:.1f}
  Career Era-Std:   {career_era:.1f}
  Peak season:      {peak_season['bravs']:.1f} BRAVS ({peak_season['year']})
  Peak 5 seasons:   {peak_5:.1f} BRAVS
  HR total:         {total_hr}

  HOF COMPARISON (typical career WAR-eq thresholds):

    Clear HOFer:       50+ WAR-eq
    Borderline:        35-50 WAR-eq
    Hall of Very Good: 25-35 WAR-eq
    Below threshold:   <25 WAR-eq

  Stanton at {career_war:.1f} WAR-eq is {
    'in the clear HOF range' if career_war >= 50 else
    'in the borderline range' if career_war >= 35 else
    'in the Hall of Very Good range' if career_war >= 25 else
    'below typical HOF thresholds'
  }.

  THE CASE FOR:
    - {total_hr} career HR {'(over 500!)' if total_hr >= 500 else '(approaching milestones)'}
    - MVP season: {peak_season['bravs']:.1f} BRAVS in {peak_season['year']}
    - {len(positive)} seasons above 3.0 BRAVS (solid value)
    - Top 3 seasons averaged {sum(peak3)/3:.1f} BRAVS (strong peak)

  THE CASE AGAINST:
    - {len(injured)} seasons with <100 games played (durability)
    - {len(negative)} negative-BRAVS seasons (injuries + decline)
    - DH penalty: spent significant time at DH (-17.5 runs/162)
    - Averaged only {avg_games:.0f} games/season over career
    - {safe_int(sum(s['g'] < 120 for s in seasons))} of {len(seasons)} seasons with <120 games

  BRAVS VERDICT: {verdict}

  CONTEXT: Stanton's case mirrors the broader HOF debate about
  peak vs. longevity. His best seasons (MVP year, 59 HR in 2017)
  are genuinely elite. But chronic injuries have stolen hundreds
  of games from his career, and the shift to DH has eliminated
  any defensive value. BRAVS penalizes both of these — durability
  losses and the DH positional penalty are explicit components.

  If Stanton reaches 500 HR, voters will struggle to keep him out
  regardless of what any metric says. But by BRAVS's cold
  probabilistic accounting, his career value is {
    'clearly HOF-worthy' if career_war >= 50 else
    'borderline — it depends on how much weight you give milestones' if career_war >= 35 else
    'below the typical HOF bar, propped up only by the HR total' if career_war >= 25 else
    'well below HOF standards despite the power numbers'
  }.
""")


if __name__ == "__main__":
    main()
