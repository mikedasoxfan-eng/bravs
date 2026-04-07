"""Data ingestion from pybaseball, Lahman database, and Statcast.

Provides functions to fetch real MLB data and convert it into
PlayerSeason objects for BRAVS computation.
"""

from __future__ import annotations

import logging

import pandas as pd

from baseball_metric.core.types import PlayerSeason

logger = logging.getLogger(__name__)


def fetch_season_batting(season: int) -> pd.DataFrame:
    """Fetch batting stats for a full season using pybaseball.

    Falls back to synthetic data if pybaseball is unavailable or rate-limited.
    """
    try:
        from pybaseball import batting_stats  # type: ignore[import-untyped]

        df = batting_stats(season, qual=0)
        logger.info("Fetched %d batting records for %d", len(df), season)
        return df
    except Exception as e:
        logger.warning("pybaseball fetch failed (%s), using synthetic data", e)
        from baseball_metric.data.synthetic import generate_synthetic_batting

        return generate_synthetic_batting(season)


def fetch_season_pitching(season: int) -> pd.DataFrame:
    """Fetch pitching stats for a full season using pybaseball."""
    try:
        from pybaseball import pitching_stats  # type: ignore[import-untyped]

        df = pitching_stats(season, qual=0)
        logger.info("Fetched %d pitching records for %d", len(df), season)
        return df
    except Exception as e:
        logger.warning("pybaseball fetch failed (%s), using synthetic data", e)
        from baseball_metric.data.synthetic import generate_synthetic_pitching

        return generate_synthetic_pitching(season)


def batting_row_to_player_season(row: pd.Series, season: int) -> PlayerSeason:  # type: ignore[type-arg]
    """Convert a pybaseball batting row to a PlayerSeason."""

    def _get(col: str, default: object = 0) -> object:
        return row.get(col, default)

    # Compute singles from hits minus extra-base hits
    hits = int(_get("H", 0))
    doubles = int(_get("2B", 0))
    triples = int(_get("3B", 0))
    hr = int(_get("HR", 0))
    singles = hits - doubles - triples - hr

    return PlayerSeason(
        player_id=str(_get("IDfg", _get("playerid", "unknown"))),
        player_name=str(_get("Name", "Unknown")),
        season=season,
        team=str(_get("Team", "UNK")),
        position=str(_get("Pos", "DH")),
        pa=int(_get("PA", 0)),
        ab=int(_get("AB", 0)),
        hits=hits,
        singles=max(singles, 0),
        doubles=doubles,
        triples=triples,
        hr=hr,
        bb=int(_get("BB", 0)),
        ibb=int(_get("IBB", 0)),
        hbp=int(_get("HBP", 0)),
        k=int(_get("SO", 0)),
        sf=int(_get("SF", 0)),
        sh=int(_get("SH", 0)),
        sb=int(_get("SB", 0)),
        cs=int(_get("CS", 0)),
        gidp=int(_get("GDP", _get("GIDP", 0))),
        games=int(_get("G", 0)),
    )


def pitching_row_to_player_season(row: pd.Series, season: int) -> PlayerSeason:  # type: ignore[type-arg]
    """Convert a pybaseball pitching row to a PlayerSeason."""

    def _get(col: str, default: object = 0) -> object:
        return row.get(col, default)

    return PlayerSeason(
        player_id=str(_get("IDfg", _get("playerid", "unknown"))),
        player_name=str(_get("Name", "Unknown")),
        season=season,
        team=str(_get("Team", "UNK")),
        position="P",
        ip=float(_get("IP", 0)),
        er=int(_get("ER", 0)),
        hits_allowed=int(_get("H", 0)),
        hr_allowed=int(_get("HR", 0)),
        bb_allowed=int(_get("BB", 0)),
        hbp_allowed=int(_get("HBP", 0)),
        k_pitching=int(_get("SO", _get("K", 0))),
        games_pitched=int(_get("G", 0)),
        games_started=int(_get("GS", 0)),
        saves=int(_get("SV", 0)),
        holds=int(_get("HLD", 0)),
    )
