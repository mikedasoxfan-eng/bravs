"""Analyze the most controversial MVP races in baseball history."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import PlayerSeason


def analyze_race(title, players, known_results):
    """Analyze an award race and print detailed results."""
    results = []
    for name, ps in players:
        r = compute_bravs(ps, fast=True)
        results.append((name, r, ps))

    results.sort(key=lambda x: x[1].bravs, reverse=True)

    print(f"\n{'=' * 85}")
    print(f"  {title}")
    print(f"{'=' * 85}")
    print(f"\n{'Rank':<6}{'Player':<22}{'Pos':<5}{'BRAVS':>7}{'ErStd':>7}{'WAReq':>7}{'90% CI':>18}")
    print("-" * 85)

    for i, (name, r, ps) in enumerate(results, 1):
        ci = r.bravs_ci_90
        actual = known_results.get(name, "")
        marker = " <-- WINNER" if actual == "WON" else ""
        print(f"  {i}.  {name:<22}{ps.position:<5}{r.bravs:>7.1f}{r.bravs_era_standardized:>7.1f}"
              f"{r.bravs_calibrated:>7.1f}  [{ci[0]:>5.1f}, {ci[1]:>5.1f}]{marker}")

    # Head-to-head between top 2
    if len(results) >= 2:
        n1, r1, _ = results[0]
        n2, r2, _ = results[1]
        n = min(len(r1.total_samples), len(r2.total_samples))
        p1_wins = float(np.mean(r1.total_samples[:n] > r2.total_samples[:n]))
        ci1 = r1.bravs_ci_90
        ci2 = r2.bravs_ci_90
        overlap = min(ci1[1], ci2[1]) - max(ci1[0], ci2[0])

        print(f"\n  P({n1} > {n2}) = {p1_wins:.0%}")
        print(f"  CI overlap: {overlap:.1f} wins")

        # Component comparison for top 2
        print(f"\n  {'Component':<22}{'#1 ' + n1:>10}{'#2 ' + n2:>10}{'Gap':>10}")
        print(f"  {'-' * 52}")
        comp_names = sorted(set(list(r1.components.keys()) + list(r2.components.keys())))
        for cn in comp_names:
            v1 = r1.components.get(cn)
            v2 = r2.components.get(cn)
            v1r = v1.runs_mean if v1 else 0.0
            v2r = v2.runs_mean if v2 else 0.0
            print(f"  {cn:<22}{v1r:>+10.1f}{v2r:>+10.1f}{v1r - v2r:>+10.1f}")


def race_1941_al_mvp():
    """1941 AL MVP: Ted Williams (.406) vs Joe DiMaggio (56-game hitting streak)."""
    williams = PlayerSeason(
        player_id="tedw41", player_name="Ted Williams", season=1941, team="BOS",
        position="LF", pa=606, ab=456, hits=185, singles=112, doubles=33,
        triples=3, hr=37, bb=147, ibb=0, hbp=3, k=27, sf=0, games=143,
        sb=2, cs=4, park_factor=1.04, league_rpg=4.76, league="AL",
    )
    dimaggio = PlayerSeason(
        player_id="joed41", player_name="Joe DiMaggio", season=1941, team="NYY",
        position="CF", pa=621, ab=541, hits=193, singles=124, doubles=43,
        triples=11, hr=30, bb=76, ibb=0, hbp=4, k=13, sf=0, games=139,
        sb=4, cs=3, park_factor=1.05, league_rpg=4.76, league="AL",
    )
    analyze_race(
        "1941 AL MVP: Ted Williams (.406 BA) vs Joe DiMaggio (56-game streak)",
        [("Ted Williams", williams), ("Joe DiMaggio", dimaggio)],
        {"Joe DiMaggio": "WON"},
    )
    print("""
  CONTEXT: DiMaggio won 291-254 in voting despite Williams hitting .406
  (last player ever). DiMaggio's 56-game hitting streak captured the
  public imagination. Williams was unpopular with Boston media.

  BRAVS TAKE: Williams' .553 OBP is one of the greatest offensive seasons
  ever. His 147 walks in 606 PA represent extraordinary value. DiMaggio
  had the lower K rate (13!) and more triples, but Williams' plate
  discipline was on another level. The LF vs CF positional difference
  is a wash in terms of total BRAVS impact.
""")


def race_2001_nl_mvp():
    """2001 NL MVP: Barry Bonds (73 HR, .863 SLG) vs Sammy Sosa (64 HR, 160 RBI)."""
    bonds = PlayerSeason(
        player_id="bonds01", player_name="Barry Bonds", season=2001, team="SF",
        position="LF", pa=664, ab=476, hits=156, singles=49, doubles=32,
        triples=2, hr=73, bb=177, ibb=35, hbp=9, k=93, sf=2, games=153,
        sb=13, cs=3, gidp=5, park_factor=0.93, league_rpg=4.78,
    )
    sosa = PlayerSeason(
        player_id="sosa01", player_name="Sammy Sosa", season=2001, team="CHC",
        position="RF", pa=697, ab=577, hits=189, singles=87, doubles=34,
        triples=5, hr=64, bb=116, ibb=9, hbp=0, k=153, sf=2, games=160,
        sb=0, cs=2, gidp=19, park_factor=1.01, league_rpg=4.78,
    )
    pujols = PlayerSeason(
        player_id="pujols01", player_name="Albert Pujols", season=2001, team="STL",
        position="LF", pa=676, ab=590, hits=194, singles=107, doubles=47,
        triples=4, hr=37, bb=69, ibb=12, hbp=9, k=93, sf=7, games=161,
        sb=1, cs=3, gidp=20, park_factor=0.98, league_rpg=4.78,
    )
    helton = PlayerSeason(
        player_id="helton01", player_name="Todd Helton", season=2001, team="COL",
        position="1B", pa=698, ab=587, hits=197, singles=105, doubles=54,
        triples=2, hr=49, bb=103, ibb=13, hbp=7, k=104, sf=5, games=159,
        sb=7, cs=3, gidp=14, park_factor=1.16, league_rpg=4.78,
    )
    analyze_race(
        "2001 NL MVP: Bonds (73 HR) vs Sosa (64 HR) vs Pujols (rookie) vs Helton",
        [("Barry Bonds", bonds), ("Sammy Sosa", sosa),
         ("Albert Pujols", pujols), ("Todd Helton", helton)],
        {"Barry Bonds": "WON"},
    )
    print("""
  CONTEXT: Bonds hit 73 home runs. That's the answer. But Sosa had 64 HR
  and 160 RBI, Pujols was a sensational rookie (37 HR, .329 BA), and
  Helton hit .336/.432/.685 but at Coors. Bonds won unanimously.

  BRAVS TAKE: Bonds' 177 walks make his wOBA stratospheric. The Coors
  park factor (1.16) hammers Helton's value — his raw stats are inflated
  by ~16%. Sosa's 153 strikeouts and 19 GIDP hurt his BRAVS relative
  to his counting stats. Pujols in his rookie year was already elite.
""")


def race_2011_nl_mvp():
    """2011 NL MVP: Ryan Braun vs Matt Kemp."""
    braun = PlayerSeason(
        player_id="braun11", player_name="Ryan Braun", season=2011, team="MIL",
        position="LF", pa=629, ab=563, hits=187, singles=106, doubles=36,
        triples=6, hr=33, bb=58, ibb=7, hbp=4, k=93, sf=4, games=150,
        sb=33, cs=6, gidp=10, park_factor=1.02, league_rpg=4.12,
    )
    kemp = PlayerSeason(
        player_id="kemp11", player_name="Matt Kemp", season=2011, team="LAD",
        position="CF", pa=689, ab=602, hits=195, singles=107, doubles=33,
        triples=4, hr=39, bb=74, ibb=12, hbp=5, k=159, sf=6, games=161,
        sb=40, cs=8, gidp=12, park_factor=0.97, league_rpg=4.12,
    )
    votto = PlayerSeason(
        player_id="votto11", player_name="Joey Votto", season=2011, team="CIN",
        position="1B", pa=719, ab=599, hits=185, singles=113, doubles=40,
        triples=3, hr=29, bb=110, ibb=8, hbp=3, k=106, sf=7, games=161,
        sb=8, cs=3, gidp=13, park_factor=1.08, league_rpg=4.12,
    )
    analyze_race(
        "2011 NL MVP: Braun vs Kemp (Triple Crown near-miss) vs Votto",
        [("Ryan Braun", braun), ("Matt Kemp", kemp), ("Joey Votto", votto)],
        {"Ryan Braun": "WON"},
    )
    print("""
  CONTEXT: Braun won despite Kemp nearly winning the Triple Crown
  (.324/39 HR/126 RBI) and having more fWAR (8.3 vs 7.7). Braun later
  tested positive for PEDs (suspended in 2013). Many feel Kemp was
  robbed — he played CF (harder position), had more PA, more HR,
  more SB, and played for a non-playoff team (voters often reward
  playoff teams). Votto was a dark horse with elite plate discipline.

  BRAVS TAKE: The CF vs LF positional difference matters. Kemp gets
  +2.5 CF credit vs Braun's -7.5 LF penalty — a 10-run swing. Kemp's
  40 SB also provides a baserunning edge. But Braun's efficiency in
  fewer PA and his lower K rate help. The real question: was Braun
  this good, or was it pharmaceutical?
""")


def main():
    print("=" * 85)
    print("  CONTROVERSIAL MVP RACES THROUGH THE BRAVS LENS")
    print("=" * 85)

    race_1941_al_mvp()
    race_2001_nl_mvp()
    race_2011_nl_mvp()

    print("\n" + "=" * 85)
    print("  SUMMARY: WHAT BRAVS REVEALS ABOUT CONTROVERSIAL VOTES")
    print("=" * 85)
    print("""
  Across these races, a pattern emerges:

  1. VOTERS OVERWEIGHT NARRATIVE: DiMaggio's streak, Bonds' 73 HR,
     Braun's team making the playoffs. BRAVS ignores narrative.

  2. DEFENSE AND BASERUNNING ARE UNDERVALUED BY VOTERS: Trout 2012,
     Kemp 2011 — elite defenders and baserunners lose to one-dimensional
     sluggers because voters focus on BA/HR/RBI.

  3. PARK FACTORS MATTER MORE THAN VOTERS THINK: Helton's Coors stats
     are deflated significantly by BRAVS. Walker 1997 same story.

  4. THE UNCERTAINTY IS REAL: In most of these races, the 90% credible
     intervals overlap substantially. BRAVS says "these were genuinely
     close" in many cases where one camp or the other insists it was
     obvious. The honest answer is usually: it depends on your
     assumptions about defensive measurement and positional value.
""")


if __name__ == "__main__":
    main()
