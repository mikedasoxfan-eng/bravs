"""Dream Team Builder — best possible 25-man roster by BRAVS."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import PlayerSeason


# Best season EVER at each roster spot (position + role)
ROSTER = {
    "C": PlayerSeason(player_id="piazza97", player_name="Mike Piazza", season=1997, team="LAD",
        position="C", pa=633, ab=556, hits=201, doubles=32, triples=1, hr=40,
        bb=69, ibb=8, hbp=3, k=77, sf=5, games=152, sb=5, cs=1,
        framing_runs=-3.0, blocking_runs=0.5, throwing_runs=-1.0,
        catcher_pitches=12000, inn_fielded=1250.0, park_factor=0.97, league_rpg=4.77),
    "1B": PlayerSeason(player_id="gehrig27", player_name="Lou Gehrig", season=1927, team="NYY",
        position="1B", pa=717, ab=584, hits=218, doubles=52, triples=18, hr=47,
        bb=109, ibb=0, hbp=3, k=84, sf=0, games=155, sb=10, cs=8,
        park_factor=1.05, league_rpg=5.06),
    "2B": PlayerSeason(player_id="hornsby22", player_name="Rogers Hornsby", season=1922, team="STL",
        position="2B", pa=697, ab=623, hits=250, doubles=46, triples=14, hr=42,
        bb=65, ibb=0, hbp=3, k=50, sf=0, games=154, sb=17, cs=12,
        park_factor=0.98, league_rpg=4.72),
    "3B": PlayerSeason(player_id="schmidt80", player_name="Mike Schmidt", season=1980, team="PHI",
        position="3B", pa=624, ab=548, hits=157, doubles=25, triples=8, hr=48,
        bb=89, ibb=13, hbp=2, k=119, sf=6, games=150, sb=12, cs=5,
        uzr=None, drs=None, inn_fielded=1280.0, total_zone=10.0,
        park_factor=1.01, league_rpg=4.29),
    "SS": PlayerSeason(player_id="wagner08", player_name="Honus Wagner", season=1908, team="PIT",
        position="SS", pa=615, ab=568, hits=201, doubles=39, triples=19, hr=10,
        bb=54, ibb=0, hbp=3, k=0, sf=0, games=151, sb=53, cs=0,
        inn_fielded=1300.0, total_zone=18.0, park_factor=0.95, league_rpg=3.37),
    "LF": PlayerSeason(player_id="ted41", player_name="Ted Williams", season=1941, team="BOS",
        position="LF", pa=606, ab=456, hits=185, doubles=33, triples=3, hr=37,
        bb=147, ibb=0, hbp=3, k=27, sf=0, games=143, sb=2, cs=4,
        park_factor=1.04, league_rpg=4.76),
    "CF": PlayerSeason(player_id="mays65", player_name="Willie Mays", season=1965, team="SF",
        position="CF", pa=638, ab=558, hits=177, doubles=21, triples=3, hr=52,
        bb=76, ibb=0, hbp=0, k=71, sf=4, games=157, sb=9, cs=4,
        inn_fielded=1350.0, total_zone=12.0, park_factor=0.93, league_rpg=4.03),
    "RF": PlayerSeason(player_id="ruth21", player_name="Babe Ruth", season=1921, team="NYY",
        position="RF", pa=693, ab=540, hits=204, doubles=44, triples=16, hr=59,
        bb=145, ibb=0, hbp=4, k=81, sf=0, games=152, sb=17, cs=13,
        park_factor=1.05, league_rpg=4.83),
    "DH": PlayerSeason(player_id="bonds01", player_name="Barry Bonds", season=2001, team="SF",
        position="DH", pa=664, ab=476, hits=156, doubles=32, triples=2, hr=73,
        bb=177, ibb=35, hbp=9, k=93, sf=2, games=153, sb=13, cs=3,
        park_factor=0.93, league_rpg=4.78),
    "SP1": PlayerSeason(player_id="pedro00", player_name="Pedro Martinez", season=2000, team="BOS",
        position="P", ip=217.0, er=42, hits_allowed=128, hr_allowed=17,
        bb_allowed=32, hbp_allowed=6, k_pitching=284, games_pitched=29,
        games_started=29, park_factor=1.04, league_rpg=5.14),
    "SP2": PlayerSeason(player_id="gibson68", player_name="Bob Gibson", season=1968, team="STL",
        position="P", ip=304.7, er=38, hits_allowed=198, hr_allowed=11,
        bb_allowed=62, hbp_allowed=4, k_pitching=268, games_pitched=34,
        games_started=34, park_factor=0.98, league_rpg=3.42),
    "SP3": PlayerSeason(player_id="koufax66", player_name="Sandy Koufax", season=1966, team="LAD",
        position="P", ip=323.0, er=62, hits_allowed=241, hr_allowed=19,
        bb_allowed=77, hbp_allowed=2, k_pitching=317, games_pitched=41,
        games_started=41, park_factor=0.95, league_rpg=3.89),
    "SP4": PlayerSeason(player_id="rjohn01", player_name="Randy Johnson", season=2001, team="ARI",
        position="P", ip=249.7, er=64, hits_allowed=181, hr_allowed=19,
        bb_allowed=71, hbp_allowed=18, k_pitching=372, games_pitched=35,
        games_started=35, park_factor=1.04, league_rpg=4.78),
    "SP5": PlayerSeason(player_id="maddux95", player_name="Greg Maddux", season=1995, team="ATL",
        position="P", ip=209.7, er=38, hits_allowed=147, hr_allowed=8,
        bb_allowed=23, hbp_allowed=4, k_pitching=181, games_pitched=28,
        games_started=28, park_factor=1.00, league_rpg=4.63),
    "CL": PlayerSeason(player_id="rivera04", player_name="Mariano Rivera", season=2004, team="NYY",
        position="P", ip=78.7, er=16, hits_allowed=65, hr_allowed=4,
        bb_allowed=20, hbp_allowed=1, k_pitching=66, games_pitched=74,
        games_started=0, saves=53, avg_leverage_index=1.85,
        park_factor=1.05, league_rpg=4.81),
}


def main():
    print("=" * 85)
    print("  THE BRAVS ALL-TIME DREAM TEAM")
    print("  Best possible season at every position")
    print("=" * 85)

    total_bravs = 0.0
    total_war = 0.0
    results = []

    for slot, ps in ROSTER.items():
        r = compute_bravs(ps, fast=True)
        results.append((slot, ps, r))
        total_bravs += r.bravs
        total_war += r.bravs_calibrated

    print(f"\n  {'Slot':<6}{'Player':<24}{'Year':>5}{'Pos':<5}{'BRAVS':>7}{'WAReq':>7}  Signature")
    print("  " + "-" * 80)

    signatures = {
        "C": ".362/.431/.638, 40 HR from catcher",
        "1B": ".373/.474/.765, 218 H / 47 HR",
        "2B": ".401/.459/.722, 250 hits (!)",
        "3B": ".286/.380/.624, 48 HR, Gold Glove D",
        "SS": ".354/.413/.542, 53 SB, elite glove",
        "LF": ".406/.553/.735, last man to hit .400",
        "CF": ".317/.398/.645, 52 HR, elite defense",
        "RF": ".378/.512/.846, 59 HR, 204 hits",
        "DH": ".328/.515/.863, 73 HR, 177 BB",
        "SP1": "1.74 ERA, 284 K in 217 IP, steroid-era",
        "SP2": "1.12 ERA, 268 K in 305 IP, Year of Pitcher",
        "SP3": "1.73 ERA, 317 K in 323 IP, final season",
        "SP4": "2.49 ERA, 372 K in 250 IP, 6'10 unit",
        "SP5": "1.63 ERA, 23 BB in 210 IP, Picasso",
        "CL": "1.94 ERA, 53 SV, gmLI 1.85, cutter god",
    }

    for slot, ps, r in results:
        sig = signatures.get(slot, "")
        print(f"  {slot:<6}{ps.player_name:<24}{ps.season:>5} {ps.position:<4}"
              f"{r.bravs:>7.1f}{r.bravs_calibrated:>7.1f}  {sig}")

    print("  " + "-" * 80)
    print(f"  {'TOTAL':<6}{'':>24}{'':>5} {'':>4}{total_bravs:>7.1f}{total_war:>7.1f}")

    # Fun stats
    lineup_bravs = sum(r.bravs for slot, _, r in results if slot not in ("SP1","SP2","SP3","SP4","SP5","CL"))
    rotation_bravs = sum(r.bravs for slot, _, r in results if slot.startswith("SP"))
    closer_bravs = sum(r.bravs for slot, _, r in results if slot == "CL")

    print(f"""
  {'=' * 85}
  TEAM VALUE BREAKDOWN
  {'=' * 85}

  Lineup (9 batters):   {lineup_bravs:>6.1f} BRAVS  ({lineup_bravs/total_bravs*100:.0f}% of total)
  Rotation (5 SP):      {rotation_bravs:>6.1f} BRAVS  ({rotation_bravs/total_bravs*100:.0f}% of total)
  Closer:               {closer_bravs:>6.1f} BRAVS  ({closer_bravs/total_bravs*100:.0f}% of total)

  This roster has {total_war:.0f} WAR-equivalent wins above FAT.
  A FAT-level team wins roughly 47-48 games.
  This dream team would therefore be expected to win:

    47 + {total_war:.0f} = {47 + total_war:.0f} games

  ...which is obviously impossible (max 162). This illustrates a known
  limitation of additive metrics: player values don't perfectly sum to
  team value because of diminishing returns (Axiom 7: conditional
  additivity). A team this good would win 140+ games in reality,
  limited by the maximum possible win pace and randomness.

  Still, even with the additivity caveat, this is the most valuable
  collection of single-season performances ever assembled.
  """)

    # Best lineup by era
    print("=" * 85)
    print("  ERA BREAKDOWN: WHEN WERE THE GREATEST SEASONS?")
    print("=" * 85)
    eras = {}
    for slot, ps, r in results:
        decade = (ps.season // 10) * 10
        eras.setdefault(decade, []).append((slot, ps, r))

    for decade in sorted(eras.keys()):
        members = eras[decade]
        dec_bravs = sum(r.bravs for _, _, r in members)
        names = ", ".join(f"{ps.player_name.split()[-1]}" for _, ps, _ in members)
        print(f"  {decade}s: {dec_bravs:>6.1f} BRAVS  ({names})")


if __name__ == "__main__":
    main()
