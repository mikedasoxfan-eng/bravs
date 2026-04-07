"""Main entry point for BRAVS computation pipeline.

Usage:
    python -m baseball_metric.run --season 2023
    python -m baseball_metric.run --season 2023 --player "Mike Trout"
    python -m baseball_metric.run --historical
    python -m baseball_metric.run --notable-seasons
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import BRAVSResult, PlayerSeason
from baseball_metric.data.sources import (
    batting_row_to_player_season,
    fetch_season_batting,
    fetch_season_pitching,
    pitching_row_to_player_season,
)
from baseball_metric.data.validation import validate_player_season

logger = logging.getLogger("baseball_metric")

# Notable player-seasons for historical validation
NOTABLE_SEASONS: list[PlayerSeason] = [
    # --- Consensus all-time great position player seasons ---
    PlayerSeason(
        player_id="trout01", player_name="Mike Trout", season=2016, team="LAA",
        position="CF", pa=681, ab=549, hits=173, singles=116, doubles=24,
        triples=4, hr=29, bb=116, ibb=5, hbp=7, k=137, sf=5, games=159,
        sb=7, cs=3, gidp=14, uzr=0.4, drs=-2, oaa=1,
        inn_fielded=1320.0, park_factor=0.98, league_rpg=4.48,
    ),
    PlayerSeason(
        player_id="bonds01", player_name="Barry Bonds", season=2004, team="SF",
        position="LF", pa=617, ab=373, hits=135, singles=60, doubles=27,
        triples=3, hr=45, bb=232, ibb=120, hbp=9, k=41, sf=3, games=147,
        sb=6, cs=1, gidp=5, park_factor=0.93, league_rpg=4.81,
    ),
    PlayerSeason(
        player_id="ruth01", player_name="Babe Ruth", season=1927, team="NYY",
        position="RF", pa=691, ab=540, hits=192, singles=95, doubles=29,
        triples=8, hr=60, bb=137, ibb=0, hbp=0, k=89, sf=0, games=151,
        sb=7, cs=6, park_factor=1.05, league_rpg=5.06, league="AL",
    ),
    PlayerSeason(
        player_id="mays01", player_name="Willie Mays", season=1965, team="SF",
        position="CF", pa=638, ab=558, hits=177, singles=95, doubles=21,
        triples=3, hr=52, bb=76, ibb=0, hbp=0, k=71, sf=4, games=157,
        sb=9, cs=4, uzr=None, drs=None, inn_fielded=1350.0,
        total_zone=12.0,  # TotalZone: elite CF defense
        park_factor=0.93, league_rpg=4.03,
    ),
    PlayerSeason(
        player_id="aaron01", player_name="Hank Aaron", season=1971, team="ATL",
        position="RF", pa=619, ab=495, hits=162, singles=83, doubles=22,
        triples=3, hr=47, bb=71, ibb=21, hbp=2, k=58, sf=5, games=139,
        sb=1, cs=1, park_factor=1.02, league_rpg=3.91,
    ),
    # --- Two-way player ---
    PlayerSeason(
        player_id="ohtani01", player_name="Shohei Ohtani", season=2023, team="LAA",
        position="DH", pa=599, ab=497, hits=151, singles=73, doubles=26,
        triples=8, hr=44, bb=91, ibb=5, hbp=5, k=143, sf=6, games=135,
        ip=132.0, er=46, hits_allowed=99, hr_allowed=18, bb_allowed=55,
        hbp_allowed=5, k_pitching=167, games_pitched=23, games_started=23,
        sb=20, cs=6, park_factor=0.98, league_rpg=4.62,
    ),
    # --- Dominant pitcher seasons ---
    PlayerSeason(
        player_id="pedro01", player_name="Pedro Martinez", season=2000, team="BOS",
        position="P", ip=217.0, er=42, hits_allowed=128, hr_allowed=17,
        bb_allowed=32, hbp_allowed=6, k_pitching=284, games_pitched=29,
        games_started=29, park_factor=1.04, league_rpg=5.14, league="AL",
    ),
    PlayerSeason(
        player_id="gibson01", player_name="Bob Gibson", season=1968, team="STL",
        position="P", ip=304.7, er=38, hits_allowed=198, hr_allowed=11,
        bb_allowed=62, hbp_allowed=4, k_pitching=268, games_pitched=34,
        games_started=34, park_factor=0.98, league_rpg=3.42,
    ),
    PlayerSeason(
        player_id="degrom01", player_name="Jacob deGrom", season=2018, team="NYM",
        position="P", ip=217.0, er=48, hits_allowed=152, hr_allowed=10,
        bb_allowed=46, hbp_allowed=5, k_pitching=269, games_pitched=32,
        games_started=32, park_factor=0.95, league_rpg=4.45,
    ),
    PlayerSeason(
        player_id="koufax01", player_name="Sandy Koufax", season=1966, team="LAD",
        position="P", ip=323.0, er=62, hits_allowed=241, hr_allowed=19,
        bb_allowed=77, hbp_allowed=2, k_pitching=317, games_pitched=41,
        games_started=41, park_factor=0.95, league_rpg=3.89,
    ),
    PlayerSeason(
        player_id="clemens01", player_name="Roger Clemens", season=1997, team="TOR",
        position="P", ip=264.0, er=65, hits_allowed=204, hr_allowed=19,
        bb_allowed=68, hbp_allowed=8, k_pitching=292, games_pitched=34,
        games_started=34, park_factor=1.03, league_rpg=4.77, league="AL",
    ),
    PlayerSeason(
        player_id="verlander01", player_name="Justin Verlander", season=2011, team="DET",
        position="P", ip=251.0, er=57, hits_allowed=174, hr_allowed=24,
        bb_allowed=57, hbp_allowed=4, k_pitching=250, games_pitched=34,
        games_started=34, park_factor=0.97, league_rpg=4.28, league="AL",
    ),
    # --- Edge cases ---
    PlayerSeason(
        player_id="martinez_e01", player_name="Edgar Martinez", season=1995, team="SEA",
        position="DH", pa=639, ab=511, hits=182, singles=103, doubles=52,
        triples=0, hr=29, bb=116, ibb=12, hbp=6, k=87, sf=6, games=145,
        sb=4, cs=0, park_factor=1.01, league_rpg=5.06, league="AL",
    ),
    PlayerSeason(
        player_id="rivera01", player_name="Mariano Rivera", season=2004, team="NYY",
        position="P", ip=78.7, er=16, hits_allowed=65, hr_allowed=4,
        bb_allowed=20, hbp_allowed=1, k_pitching=66, games_pitched=74,
        games_started=0, saves=53, avg_leverage_index=1.85,
        park_factor=1.05, league_rpg=4.81, league="AL",
    ),
    PlayerSeason(
        player_id="smith_oz01", player_name="Ozzie Smith", season=1987, team="STL",
        position="SS", pa=706, ab=600, hits=182, singles=140, doubles=25,
        triples=4, hr=0, bb=89, ibb=6, hbp=2, k=36, sf=6, games=158,
        sb=43, cs=9, uzr=None, drs=None, oaa=None, inn_fielded=1380.0,
        total_zone=20.0,  # TotalZone: elite SS defense, the Wizard
        park_factor=0.98, league_rpg=4.52,
    ),
    # --- Short season ---
    PlayerSeason(
        player_id="soto01", player_name="Juan Soto", season=2020, team="WSH",
        position="RF", pa=196, ab=153, hits=54, singles=24, doubles=12,
        triples=0, hr=13, bb=41, ibb=6, hbp=0, k=28, sf=2, games=47,
        sb=6, cs=0, park_factor=0.98, league_rpg=4.65, season_games=60,
    ),
    # --- Controversial WAR case ---
    PlayerSeason(
        player_id="walker01", player_name="Larry Walker", season=1997, team="COL",
        position="RF", pa=613, ab=568, hits=208, singles=107, doubles=46,
        triples=4, hr=49, bb=78, ibb=12, hbp=6, k=90, sf=3, games=153,
        sb=33, cs=6, park_factor=1.16, league_rpg=4.77,
    ),
    # --- Elite catcher ---
    PlayerSeason(
        player_id="piazza01", player_name="Mike Piazza", season=1997, team="LAD",
        position="C", pa=633, ab=556, hits=201, singles=113, doubles=32,
        triples=1, hr=40, bb=69, ibb=8, hbp=3, k=77, sf=5, games=152,
        sb=5, cs=1, framing_runs=-3.0, blocking_runs=0.5, throwing_runs=-1.0,
        catcher_pitches=12000, inn_fielded=1250.0,
        park_factor=0.97, league_rpg=4.77,
    ),
    # --- Recent MVP seasons ---
    PlayerSeason(
        player_id="judge01", player_name="Aaron Judge", season=2022, team="NYY",
        position="RF", pa=696, ab=570, hits=177, singles=74, doubles=28,
        triples=0, hr=62, bb=111, ibb=19, hbp=6, k=175, sf=4, games=157,
        sb=16, cs=3, uzr=3.5, drs=5, oaa=6, inn_fielded=1200.0,
        park_factor=1.05, league_rpg=4.28, league="AL",
    ),
    PlayerSeason(
        player_id="betts01", player_name="Mookie Betts", season=2018, team="BOS",
        position="RF", pa=614, ab=520, hits=180, singles=96, doubles=47,
        triples=5, hr=32, bb=81, ibb=5, hbp=5, k=94, sf=5, games=136,
        sb=30, cs=5, uzr=10.5, drs=12, oaa=9, inn_fielded=1100.0,
        park_factor=1.04, league_rpg=4.45, league="AL",
    ),
    # --- Compiler vs peak: Harold Baines (compiler) ---
    PlayerSeason(
        player_id="baines01", player_name="Harold Baines", season=1985, team="CHW",
        position="RF", pa=673, ab=640, hits=198, singles=131, doubles=29,
        triples=3, hr=22, bb=42, ibb=5, hbp=2, k=89, sf=4, games=160,
        sb=1, cs=5, park_factor=1.01, league_rpg=4.33, league="AL",
    ),
    # --- Walter Johnson dead-ball era ---
    PlayerSeason(
        player_id="johnson_w01", player_name="Walter Johnson", season=1913, team="WSH",
        position="P", ip=346.0, er=56, hits_allowed=232, hr_allowed=2,
        bb_allowed=38, hbp_allowed=9, k_pitching=243, games_pitched=48,
        games_started=36, park_factor=0.97, league_rpg=3.93, league="AL",
    ),
]


def run_notable_seasons() -> list[BRAVSResult]:
    """Compute BRAVS for all notable historical seasons."""
    results = []
    for player in NOTABLE_SEASONS:
        validation = validate_player_season(player)
        if not validation.is_valid:
            logger.error("Skipping %s (%d): %s",
                         player.player_name, player.season, validation.errors)
            continue

        result = compute_bravs(player)
        results.append(result)
        print(result.summary())
        print()

    return results


def run_season(season: int, player_filter: str | None = None) -> list[BRAVSResult]:
    """Compute BRAVS for an entire season."""
    print(f"Fetching data for {season} season...")
    batting_df = fetch_season_batting(season)
    pitching_df = fetch_season_pitching(season)

    results = []

    # Process batters
    for _, row in batting_df.iterrows():
        player = batting_row_to_player_season(row, season)
        if player_filter and player_filter.lower() not in player.player_name.lower():
            continue
        if player.pa < 50:
            continue

        validation = validate_player_season(player)
        if not validation.is_valid:
            continue

        result = compute_bravs(player)
        results.append(result)

    # Process pitchers
    for _, row in pitching_df.iterrows():
        player = pitching_row_to_player_season(row, season)
        if player_filter and player_filter.lower() not in player.player_name.lower():
            continue
        if player.ip < 10:
            continue

        validation = validate_player_season(player)
        if not validation.is_valid:
            continue

        result = compute_bravs(player)
        results.append(result)

    # Sort by BRAVS descending
    results.sort(key=lambda r: r.bravs, reverse=True)

    if player_filter:
        for r in results:
            print(r.summary())
    else:
        # Print top 25
        print(f"\n{'='*70}")
        print(f"  BRAVS Leaderboard — {season} Season")
        print(f"{'='*70}")
        print(f"{'Rank':>4}  {'Player':<25} {'Pos':<4} {'BRAVS':>6}  {'90% CI':>16}")
        print(f"{'-'*70}")
        for i, r in enumerate(results[:25], 1):
            ci = r.bravs_ci_90
            print(f"{i:>4}. {r.player.player_name:<25} {r.player.position:<4} "
                  f"{r.bravs:>6.1f}  [{ci[0]:>5.1f}, {ci[1]:>5.1f}]")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BRAVS — Bayesian Runs Above Value Standard"
    )
    parser.add_argument("--season", type=int, help="Compute BRAVS for a full season")
    parser.add_argument("--player", type=str, help="Filter to a specific player name")
    parser.add_argument("--notable-seasons", action="store_true",
                        help="Run on notable historical seasons")
    parser.add_argument("--historical", action="store_true",
                        help="Run on all notable historical seasons with full output")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--mcmc", action="store_true",
                        help="Use MCMC posteriors (slower, captures skewness)")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(name)s %(levelname)s: %(message)s")

    start_time = time.perf_counter()

    if args.notable_seasons or args.historical:
        results = run_notable_seasons()
        print(f"\n{'='*90}")
        print(f"  All-Time BRAVS Leaderboard (Notable Seasons)")
        print(f"{'='*90}")
        results.sort(key=lambda r: r.bravs, reverse=True)
        print(f"{'Rank':>4}  {'Player':<25} {'Year':>5} {'Pos':<4} "
              f"{'BRAVS':>6} {'ErStd':>6} {'WAReq':>6}  {'90% CI':>16}")
        print(f"{'-'*90}")
        for i, r in enumerate(results, 1):
            ci = r.bravs_ci_90
            print(f"{i:>4}. {r.player.player_name:<25} {r.player.season:>5} "
                  f"{r.player.position:<4} {r.bravs:>6.1f} "
                  f"{r.bravs_era_standardized:>6.1f} {r.bravs_calibrated:>6.1f}"
                  f"  [{ci[0]:>5.1f}, {ci[1]:>5.1f}]")

    elif args.season:
        run_season(args.season, args.player)
    else:
        parser.print_help()

    elapsed = time.perf_counter() - start_time
    print(f"\nCompleted in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
