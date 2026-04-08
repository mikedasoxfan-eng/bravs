"""Dynasty Rankings — best 5-year stretches in MLB history.

Finds the most dominant half-decade runs ever. Different from career
totals because it rewards sustained peak performance over accumulation.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import PlayerSeason


# Each player gets their best 5 consecutive seasons
DYNASTIES = {
    "Bonds 2000-2004": [
        PlayerSeason(player_id="b00", player_name="Barry Bonds", season=2000, team="SF",
            position="LF", pa=607, ab=480, hits=147, doubles=28, triples=4, hr=49,
            bb=117, ibb=22, hbp=3, k=77, sf=4, games=143, sb=11, cs=3,
            park_factor=0.93, league_rpg=5.14),
        PlayerSeason(player_id="b01", player_name="Barry Bonds", season=2001, team="SF",
            position="LF", pa=664, ab=476, hits=156, doubles=32, triples=2, hr=73,
            bb=177, ibb=35, hbp=9, k=93, sf=2, games=153, sb=13, cs=3,
            park_factor=0.93, league_rpg=4.78),
        PlayerSeason(player_id="b02", player_name="Barry Bonds", season=2002, team="SF",
            position="LF", pa=612, ab=403, hits=149, doubles=31, triples=2, hr=46,
            bb=198, ibb=68, hbp=9, k=47, sf=2, games=143, sb=9, cs=2,
            park_factor=0.93, league_rpg=4.62),
        PlayerSeason(player_id="b03", player_name="Barry Bonds", season=2003, team="SF",
            position="LF", pa=550, ab=390, hits=133, doubles=22, triples=1, hr=45,
            bb=148, ibb=61, hbp=10, k=58, sf=2, games=130, sb=7, cs=0,
            park_factor=0.93, league_rpg=4.73),
        PlayerSeason(player_id="b04", player_name="Barry Bonds", season=2004, team="SF",
            position="LF", pa=617, ab=373, hits=135, doubles=27, triples=3, hr=45,
            bb=232, ibb=120, hbp=9, k=41, sf=3, games=147, sb=6, cs=1,
            park_factor=0.93, league_rpg=4.81),
    ],
    "Ruth 1920-1924": [
        PlayerSeason(player_id="r20", player_name="Babe Ruth", season=1920, team="NYY",
            position="RF", pa=616, ab=458, hits=172, doubles=36, triples=9, hr=54,
            bb=150, ibb=0, hbp=3, k=80, sf=0, games=142, sb=14, cs=14,
            park_factor=1.05, league_rpg=4.39),
        PlayerSeason(player_id="r21", player_name="Babe Ruth", season=1921, team="NYY",
            position="RF", pa=693, ab=540, hits=204, doubles=44, triples=16, hr=59,
            bb=145, ibb=0, hbp=4, k=81, sf=0, games=152, sb=17, cs=13,
            park_factor=1.05, league_rpg=4.83),
        PlayerSeason(player_id="r22", player_name="Babe Ruth", season=1922, team="NYY",
            position="RF", pa=557, ab=406, hits=128, doubles=24, triples=8, hr=35,
            bb=84, ibb=0, hbp=2, k=80, sf=0, games=110, sb=2, cs=5,
            park_factor=1.05, league_rpg=4.72),
        PlayerSeason(player_id="r23", player_name="Babe Ruth", season=1923, team="NYY",
            position="RF", pa=699, ab=522, hits=205, doubles=45, triples=13, hr=41,
            bb=170, ibb=0, hbp=4, k=93, sf=0, games=152, sb=17, cs=21,
            park_factor=1.05, league_rpg=4.82),
        PlayerSeason(player_id="r24", player_name="Babe Ruth", season=1924, team="NYY",
            position="RF", pa=683, ab=529, hits=200, doubles=39, triples=7, hr=46,
            bb=142, ibb=0, hbp=4, k=81, sf=0, games=153, sb=9, cs=13,
            park_factor=1.05, league_rpg=4.70),
    ],
    "Trout 2012-2016": [
        PlayerSeason(player_id="t12", player_name="Mike Trout", season=2012, team="LAA",
            position="CF", pa=639, ab=559, hits=182, doubles=27, triples=8, hr=30,
            bb=67, ibb=8, hbp=4, k=139, sf=7, games=139, sb=49, cs=5,
            park_factor=0.98, league_rpg=4.32),
        PlayerSeason(player_id="t13", player_name="Mike Trout", season=2013, team="LAA",
            position="CF", pa=716, ab=589, hits=190, doubles=39, triples=9, hr=27,
            bb=110, ibb=10, hbp=9, k=136, sf=5, games=157, sb=33, cs=7,
            park_factor=0.98, league_rpg=4.17),
        PlayerSeason(player_id="t14", player_name="Mike Trout", season=2014, team="LAA",
            position="CF", pa=705, ab=602, hits=173, doubles=39, triples=9, hr=36,
            bb=83, ibb=10, hbp=6, k=184, sf=3, games=157, sb=16, cs=2,
            park_factor=0.98, league_rpg=4.07),
        PlayerSeason(player_id="t15", player_name="Mike Trout", season=2015, team="LAA",
            position="CF", pa=682, ab=575, hits=172, doubles=28, triples=2, hr=41,
            bb=92, ibb=11, hbp=7, k=158, sf=7, games=159, sb=11, cs=3,
            park_factor=0.98, league_rpg=4.25),
        PlayerSeason(player_id="t16", player_name="Mike Trout", season=2016, team="LAA",
            position="CF", pa=681, ab=549, hits=173, doubles=24, triples=4, hr=29,
            bb=116, ibb=5, hbp=7, k=137, sf=5, games=159, sb=7, cs=3,
            park_factor=0.98, league_rpg=4.48),
    ],
    "Mays 1962-1966": [
        PlayerSeason(player_id="m62", player_name="Willie Mays", season=1962, team="SF",
            position="CF", pa=685, ab=621, hits=189, doubles=36, triples=5, hr=49,
            bb=78, ibb=0, hbp=2, k=85, sf=4, games=162, sb=18, cs=6,
            inn_fielded=1400.0, total_zone=10.0, park_factor=0.93, league_rpg=4.46),
        PlayerSeason(player_id="m63", player_name="Willie Mays", season=1963, team="SF",
            position="CF", pa=651, ab=596, hits=187, doubles=32, triples=7, hr=38,
            bb=66, ibb=0, hbp=2, k=83, sf=4, games=157, sb=8, cs=3,
            inn_fielded=1350.0, total_zone=9.0, park_factor=0.93, league_rpg=3.95),
        PlayerSeason(player_id="m64", player_name="Willie Mays", season=1964, team="SF",
            position="CF", pa=646, ab=578, hits=171, doubles=21, triples=9, hr=47,
            bb=82, ibb=0, hbp=2, k=72, sf=4, games=157, sb=19, cs=4,
            inn_fielded=1360.0, total_zone=11.0, park_factor=0.93, league_rpg=4.04),
        PlayerSeason(player_id="m65", player_name="Willie Mays", season=1965, team="SF",
            position="CF", pa=638, ab=558, hits=177, doubles=21, triples=3, hr=52,
            bb=76, ibb=0, hbp=0, k=71, sf=4, games=157, sb=9, cs=4,
            inn_fielded=1350.0, total_zone=12.0, park_factor=0.93, league_rpg=4.03),
        PlayerSeason(player_id="m66", player_name="Willie Mays", season=1966, team="SF",
            position="CF", pa=606, ab=552, hits=159, doubles=29, triples=4, hr=37,
            bb=70, ibb=0, hbp=2, k=81, sf=2, games=152, sb=5, cs=4,
            inn_fielded=1300.0, total_zone=8.0, park_factor=0.93, league_rpg=3.89),
    ],
    "Pedro 1997-2002": [
        PlayerSeason(player_id="p97", player_name="Pedro Martinez", season=1997, team="MON",
            position="P", ip=241.3, er=54, hits_allowed=158, hr_allowed=16,
            bb_allowed=67, hbp_allowed=7, k_pitching=305, games_pitched=31,
            games_started=31, park_factor=0.98, league_rpg=4.77),
        PlayerSeason(player_id="p98", player_name="Pedro Martinez", season=1998, team="BOS",
            position="P", ip=233.7, er=58, hits_allowed=188, hr_allowed=11,
            bb_allowed=32, hbp_allowed=6, k_pitching=251, games_pitched=33,
            games_started=33, park_factor=1.04, league_rpg=4.79),
        PlayerSeason(player_id="p99", player_name="Pedro Martinez", season=1999, team="BOS",
            position="P", ip=213.3, er=49, hits_allowed=160, hr_allowed=9,
            bb_allowed=37, hbp_allowed=9, k_pitching=313, games_pitched=31,
            games_started=31, park_factor=1.04, league_rpg=5.08),
        PlayerSeason(player_id="p00", player_name="Pedro Martinez", season=2000, team="BOS",
            position="P", ip=217.0, er=42, hits_allowed=128, hr_allowed=17,
            bb_allowed=32, hbp_allowed=6, k_pitching=284, games_pitched=29,
            games_started=29, park_factor=1.04, league_rpg=5.14),
        PlayerSeason(player_id="p01", player_name="Pedro Martinez", season=2001, team="BOS",
            position="P", ip=116.7, er=24, hits_allowed=84, hr_allowed=5,
            bb_allowed=25, hbp_allowed=3, k_pitching=163, games_pitched=18,
            games_started=18, park_factor=1.04, league_rpg=4.78),
    ],
    "W. Johnson 1912-1916": [
        PlayerSeason(player_id="wj12", player_name="Walter Johnson", season=1912, team="WSH",
            position="P", ip=369.0, er=69, hits_allowed=259, hr_allowed=4,
            bb_allowed=76, hbp_allowed=11, k_pitching=303, games_pitched=50,
            games_started=37, park_factor=0.97, league_rpg=4.10),
        PlayerSeason(player_id="wj13", player_name="Walter Johnson", season=1913, team="WSH",
            position="P", ip=346.0, er=56, hits_allowed=232, hr_allowed=2,
            bb_allowed=38, hbp_allowed=9, k_pitching=243, games_pitched=48,
            games_started=36, park_factor=0.97, league_rpg=3.93),
        PlayerSeason(player_id="wj14", player_name="Walter Johnson", season=1914, team="WSH",
            position="P", ip=371.7, er=72, hits_allowed=287, hr_allowed=3,
            bb_allowed=74, hbp_allowed=6, k_pitching=225, games_pitched=51,
            games_started=40, park_factor=0.97, league_rpg=3.65),
        PlayerSeason(player_id="wj15", player_name="Walter Johnson", season=1915, team="WSH",
            position="P", ip=336.7, er=74, hits_allowed=258, hr_allowed=4,
            bb_allowed=56, hbp_allowed=8, k_pitching=203, games_pitched=47,
            games_started=38, park_factor=0.97, league_rpg=3.68),
        PlayerSeason(player_id="wj16", player_name="Walter Johnson", season=1916, team="WSH",
            position="P", ip=369.7, er=73, hits_allowed=290, hr_allowed=3,
            bb_allowed=82, hbp_allowed=7, k_pitching=228, games_pitched=48,
            games_started=38, park_factor=0.97, league_rpg=3.52),
    ],
}


def main():
    print("=" * 90)
    print("  DYNASTY RANKINGS: BEST 5-YEAR STRETCHES IN MLB HISTORY")
    print("=" * 90)

    dynasty_totals = []

    for name, seasons in DYNASTIES.items():
        total_bravs = 0.0
        total_era = 0.0
        total_war = 0.0
        year_results = []

        for ps in seasons:
            r = compute_bravs(ps, fast=True)
            total_bravs += r.bravs
            total_era += r.bravs_era_standardized
            total_war += r.bravs_calibrated
            year_results.append((ps.season, r.bravs, r.bravs_era_standardized))

        avg_bravs = total_bravs / len(seasons)
        dynasty_totals.append((name, total_bravs, total_era, total_war, avg_bravs, year_results))

    dynasty_totals.sort(key=lambda x: x[2], reverse=True)  # sort by era-standardized

    print(f"\n  {'Rank':<6}{'Dynasty':<28}{'Total':>7}{'ErStd':>7}{'WAReq':>7}{'Avg/yr':>7}")
    print("  " + "-" * 70)

    for i, (name, total, era_std, war_eq, avg, years) in enumerate(dynasty_totals, 1):
        print(f"  {i:>3}.  {name:<28}{total:>7.1f}{era_std:>7.1f}{war_eq:>7.1f}{avg:>7.1f}")

    # Detailed breakdown
    for i, (name, total, era_std, war_eq, avg, years) in enumerate(dynasty_totals[:3], 1):
        print(f"\n  {'=' * 60}")
        print(f"  #{i}: {name}")
        print(f"  {'=' * 60}")
        print(f"  5-year total: {total:.1f} BRAVS / {era_std:.1f} ErStd / {war_eq:.1f} WAR-eq")
        print(f"  Average per season: {avg:.1f} BRAVS")
        print()
        for yr, bravs, era in years:
            bar = "#" * int(max(bravs, 0) / 1.5)
            print(f"    {yr}: {bravs:>6.1f} BRAVS  {bar}")

    print(f"""
  {'=' * 90}
  ANALYSIS
  {'=' * 90}

  The dynasty ranking reveals which players sustained the highest level
  of production over a 5-year window. This rewards consistency and
  durability — a player who puts up 15 BRAVS every year for 5 years
  ranks higher than one who puts up 25 once and 8 the other four.

  Key insight: The era-standardized column is the fairest comparison
  because it removes the RPW advantage that dead-ball era players get.
  Under era-standardization, the ranking may shift significantly from
  the raw BRAVS ranking.
""")


if __name__ == "__main__":
    main()
