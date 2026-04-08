"""Player Similarity Finder — who had the most similar season to X?"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import PlayerSeason

# Database of notable seasons to compare against
REFERENCE_SEASONS = [
    PlayerSeason(player_id="trout16", player_name="Mike Trout", season=2016, team="LAA",
        position="CF", pa=681, ab=549, hits=173, doubles=24, triples=4, hr=29,
        bb=116, ibb=5, hbp=7, k=137, sf=5, games=159, sb=7, cs=3, gidp=14,
        park_factor=0.98, league_rpg=4.48),
    PlayerSeason(player_id="bonds04", player_name="Barry Bonds", season=2004, team="SF",
        position="LF", pa=617, ab=373, hits=135, doubles=27, triples=3, hr=45,
        bb=232, ibb=120, hbp=9, k=41, sf=3, games=147, sb=6, cs=1,
        park_factor=0.93, league_rpg=4.81),
    PlayerSeason(player_id="ruth27", player_name="Babe Ruth", season=1927, team="NYY",
        position="RF", pa=691, ab=540, hits=192, doubles=29, triples=8, hr=60,
        bb=137, ibb=0, hbp=0, k=89, sf=0, games=151, sb=7, cs=6,
        park_factor=1.05, league_rpg=5.06),
    PlayerSeason(player_id="mays65", player_name="Willie Mays", season=1965, team="SF",
        position="CF", pa=638, ab=558, hits=177, doubles=21, triples=3, hr=52,
        bb=76, ibb=0, hbp=0, k=71, sf=4, games=157, sb=9, cs=4,
        inn_fielded=1350.0, total_zone=12.0, park_factor=0.93, league_rpg=4.03),
    PlayerSeason(player_id="judge22", player_name="Aaron Judge", season=2022, team="NYY",
        position="RF", pa=696, ab=570, hits=177, doubles=28, triples=0, hr=62,
        bb=111, ibb=19, hbp=6, k=175, sf=4, games=157, sb=16, cs=3,
        uzr=3.5, drs=5, oaa=6, inn_fielded=1200.0,
        park_factor=1.05, league_rpg=4.28),
    PlayerSeason(player_id="ohtani23", player_name="Shohei Ohtani", season=2023, team="LAA",
        position="DH", pa=599, ab=497, hits=151, doubles=26, triples=8, hr=44,
        bb=91, ibb=5, hbp=5, k=143, sf=6, games=135,
        ip=132.0, er=46, hits_allowed=99, hr_allowed=18, bb_allowed=55,
        hbp_allowed=5, k_pitching=167, games_pitched=23, games_started=23,
        sb=20, cs=6, park_factor=0.98, league_rpg=4.62),
    PlayerSeason(player_id="pedro00", player_name="Pedro Martinez", season=2000, team="BOS",
        position="P", ip=217.0, er=42, hits_allowed=128, hr_allowed=17,
        bb_allowed=32, hbp_allowed=6, k_pitching=284, games_pitched=29,
        games_started=29, park_factor=1.04, league_rpg=5.14),
    PlayerSeason(player_id="gibson68", player_name="Bob Gibson", season=1968, team="STL",
        position="P", ip=304.7, er=38, hits_allowed=198, hr_allowed=11,
        bb_allowed=62, hbp_allowed=4, k_pitching=268, games_pitched=34,
        games_started=34, park_factor=0.98, league_rpg=3.42),
    PlayerSeason(player_id="degrom18", player_name="Jacob deGrom", season=2018, team="NYM",
        position="P", ip=217.0, er=48, hits_allowed=152, hr_allowed=10,
        bb_allowed=46, hbp_allowed=5, k_pitching=269, games_pitched=32,
        games_started=32, park_factor=0.95, league_rpg=4.45),
    PlayerSeason(player_id="betts18", player_name="Mookie Betts", season=2018, team="BOS",
        position="RF", pa=614, ab=520, hits=180, doubles=47, triples=5, hr=32,
        bb=81, ibb=5, hbp=5, k=94, sf=5, games=136, sb=30, cs=5,
        uzr=10.5, drs=12, oaa=9, inn_fielded=1100.0,
        park_factor=1.04, league_rpg=4.45),
    PlayerSeason(player_id="rivera04", player_name="Mariano Rivera", season=2004, team="NYY",
        position="P", ip=78.7, er=16, hits_allowed=65, hr_allowed=4,
        bb_allowed=20, hbp_allowed=1, k_pitching=66, games_pitched=74,
        games_started=0, saves=53, avg_leverage_index=1.85,
        park_factor=1.05, league_rpg=4.81),
    PlayerSeason(player_id="piazza97", player_name="Mike Piazza", season=1997, team="LAD",
        position="C", pa=633, ab=556, hits=201, doubles=32, triples=1, hr=40,
        bb=69, ibb=8, hbp=3, k=77, sf=5, games=152, sb=5, cs=1,
        framing_runs=-3.0, blocking_runs=0.5, throwing_runs=-1.0,
        catcher_pitches=12000, inn_fielded=1250.0,
        park_factor=0.97, league_rpg=4.77),
    PlayerSeason(player_id="smith87", player_name="Ozzie Smith", season=1987, team="STL",
        position="SS", pa=706, ab=600, hits=182, doubles=25, triples=4, hr=0,
        bb=89, ibb=6, hbp=2, k=36, sf=6, games=158, sb=43, cs=9,
        inn_fielded=1380.0, total_zone=20.0,
        park_factor=0.98, league_rpg=4.52),
    PlayerSeason(player_id="altuve17", player_name="Jose Altuve", season=2017, team="HOU",
        position="2B", pa=662, ab=590, hits=204, doubles=39, triples=4, hr=24,
        bb=58, ibb=5, hbp=7, k=84, sf=3, games=153, sb=32, cs=6, gidp=15,
        park_factor=0.98, league_rpg=4.65),
    PlayerSeason(player_id="griffey97", player_name="Ken Griffey Jr.", season=1997, team="SEA",
        position="CF", pa=694, ab=608, hits=185, doubles=34, triples=3, hr=56,
        bb=76, ibb=12, hbp=3, k=121, sf=6, games=157, sb=15, cs=4,
        park_factor=0.97, league_rpg=4.77),
    PlayerSeason(player_id="kershaw14", player_name="Clayton Kershaw", season=2014, team="LAD",
        position="P", ip=198.3, er=39, hits_allowed=139, hr_allowed=9,
        bb_allowed=31, hbp_allowed=2, k_pitching=239, games_pitched=27,
        games_started=27, park_factor=0.97, league_rpg=4.07),
    PlayerSeason(player_id="cabrera12", player_name="Miguel Cabrera", season=2012, team="DET",
        position="3B", pa=697, ab=622, hits=205, doubles=40, triples=0, hr=44,
        bb=66, ibb=17, hbp=3, k=98, sf=5, games=161, sb=4, cs=1, gidp=28,
        uzr=-7.2, drs=-8, inn_fielded=1250.0,
        park_factor=0.99, league_rpg=4.32),
    PlayerSeason(player_id="henderson82", player_name="Rickey Henderson", season=1982, team="OAK",
        position="LF", pa=627, ab=536, hits=143, doubles=24, triples=4, hr=10,
        bb=116, ibb=12, hbp=2, k=94, sf=3, games=149, sb=130, cs=42,
        park_factor=0.92, league_rpg=4.29),
    PlayerSeason(player_id="pujols06", player_name="Albert Pujols", season=2006, team="STL",
        position="1B", pa=634, ab=535, hits=177, doubles=33, triples=1, hr=49,
        bb=92, ibb=28, hbp=4, k=50, sf=3, games=143, sb=7, cs=2,
        park_factor=0.98, league_rpg=4.86),
    PlayerSeason(player_id="ted41", player_name="Ted Williams", season=1941, team="BOS",
        position="LF", pa=606, ab=456, hits=185, doubles=33, triples=3, hr=37,
        bb=147, ibb=0, hbp=3, k=27, sf=0, games=143, sb=2, cs=4,
        park_factor=1.04, league_rpg=4.76),
]


def compute_similarity(r1_components, r2_components):
    """Compute similarity score between two BRAVS results.

    Uses weighted Euclidean distance on component run values.
    Lower distance = more similar. Converted to 0-100 scale.
    """
    all_names = sorted(set(list(r1_components.keys()) + list(r2_components.keys())))
    weights = {
        "hitting": 3.0, "pitching": 3.0, "baserunning": 2.0,
        "fielding": 1.5, "positional": 1.0, "approach_quality": 1.0,
        "durability": 0.5, "leverage": 0.5, "catcher": 1.5,
    }

    dist_sq = 0.0
    for name in all_names:
        v1 = r1_components.get(name)
        v2 = r2_components.get(name)
        r1v = v1.runs_mean if v1 else 0.0
        r2v = v2.runs_mean if v2 else 0.0
        w = weights.get(name, 1.0)
        dist_sq += w * (r1v - r2v) ** 2

    dist = np.sqrt(dist_sq)
    # Convert to 0-100 similarity (100 = identical, 0 = completely different)
    # Calibrated so 0 distance = 100, 50 run-distance = 50, 100+ = ~0
    similarity = max(0, 100 - dist * 1.2)
    return round(similarity, 1)


def find_similar(target_ps, top_n=5):
    """Find the most similar seasons to a target player-season."""
    target_r = compute_bravs(target_ps, fast=True)

    results = []
    for ref_ps in REFERENCE_SEASONS:
        if ref_ps.player_id == target_ps.player_id:
            continue
        ref_r = compute_bravs(ref_ps, fast=True)
        sim = compute_similarity(target_r.components, ref_r.components)
        results.append((ref_ps, ref_r, sim))

    results.sort(key=lambda x: x[2], reverse=True)
    return target_r, results[:top_n]


def print_comparison(target_ps, target_r, matches):
    """Print similarity analysis."""
    print(f"\n  Target: {target_ps.player_name} {target_ps.season} "
          f"({target_ps.position}, {target_r.bravs:.1f} BRAVS)")
    print()

    print(f"  {'Rank':<6}{'Similarity':>10}  {'Player':<28}{'BRAVS':>7}  Why Similar")
    print(f"  {'-' * 80}")

    for i, (ref_ps, ref_r, sim) in enumerate(matches, 1):
        # Find the most similar components
        similar_comps = []
        for name in sorted(target_r.components.keys()):
            tc = target_r.components.get(name)
            rc = ref_r.components.get(name)
            if tc and rc and abs(tc.runs_mean - rc.runs_mean) < 5:
                similar_comps.append(name)

        why = ", ".join(similar_comps[:3]) if similar_comps else "overall profile"
        bar = "#" * int(sim / 5)
        print(f"  {i:>3}.  {sim:>8.1f}%  {ref_ps.player_name + ' ' + str(ref_ps.season):<28}"
              f"{ref_r.bravs:>7.1f}  {why}")
        print(f"        {bar}")


def main():
    print("=" * 85)
    print("  PLAYER SIMILARITY FINDER")
    print("  Who had the most similar season?")
    print("=" * 85)

    # Test cases: find similar seasons for interesting players
    test_cases = [
        PlayerSeason(player_id="soto20", player_name="Juan Soto", season=2024, team="NYY",
            position="RF", pa=684, ab=557, hits=166, doubles=31, triples=2, hr=41,
            bb=129, ibb=6, hbp=2, k=131, sf=6, games=157, sb=1, cs=2,
            park_factor=1.05, league_rpg=4.52),
        PlayerSeason(player_id="witt25", player_name="Bobby Witt Jr.", season=2025, team="KC",
            position="SS", pa=694, ab=632, hits=193, doubles=37, triples=7, hr=23,
            bb=50, ibb=3, hbp=5, k=116, sf=5, games=157, sb=40, cs=10,
            park_factor=1.00, league_rpg=4.45),
        PlayerSeason(player_id="skubal25", player_name="Tarik Skubal", season=2025, team="DET",
            position="P", ip=207.0, er=51, hits_allowed=143, hr_allowed=18,
            bb_allowed=35, hbp_allowed=5, k_pitching=258, games_pitched=31,
            games_started=31, park_factor=0.97, league_rpg=4.45),
    ]

    for target in test_cases:
        target_r, matches = find_similar(target)
        print_comparison(target, target_r, matches)
        print()

    # Now show the full similarity matrix for a few iconic seasons
    print("=" * 85)
    print("  SIMILARITY MATRIX: ICONIC SEASONS")
    print("=" * 85)

    icons = REFERENCE_SEASONS[:8]  # Trout, Bonds, Ruth, Mays, Judge, Ohtani, Pedro, Gibson
    icon_results = [(ps, compute_bravs(ps, fast=True)) for ps in icons]

    # Header
    short_names = [f"{ps.player_name.split()[-1][:6]} {ps.season % 100:02d}" for ps, _ in icon_results]
    print(f"\n  {'':>14}", end="")
    for sn in short_names:
        print(f"{sn:>10}", end="")
    print()
    print(f"  {'':>14}" + "-" * (10 * len(short_names)))

    for i, (ps_i, r_i) in enumerate(icon_results):
        name = f"{ps_i.player_name.split()[-1][:6]} {ps_i.season % 100:02d}"
        print(f"  {name:>14}", end="")
        for j, (ps_j, r_j) in enumerate(icon_results):
            if i == j:
                print(f"{'---':>10}", end="")
            else:
                sim = compute_similarity(r_i.components, r_j.components)
                print(f"{sim:>9.0f}%", end="")
        print()

    print()


if __name__ == "__main__":
    main()
