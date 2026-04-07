"""Shared pytest fixtures for the BRAVS test suite.

Each fixture returns a PlayerSeason populated with realistic stat lines
drawn from actual MLB seasons (or plausible composites).  All counting
stats are internally consistent unless a test specifically requires
otherwise.
"""

from __future__ import annotations

import pytest

from baseball_metric.core.types import PlayerSeason


# ---------------------------------------------------------------------------
# Position-player fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trout_2016() -> PlayerSeason:
    """Mike Trout 2016 — elite CF, MVP-caliber season.

    Source: Baseball-Reference / FanGraphs 2016 season page.
    """
    return PlayerSeason(
        player_id="troutmi01",
        player_name="Mike Trout",
        season=2016,
        team="LAA",
        position="CF",
        pa=681,
        ab=441,
        hits=173,
        singles=116,
        doubles=24,
        triples=4,
        hr=29,
        bb=116,
        ibb=5,
        hbp=7,
        k=137,
        sf=5,
        sh=0,
        sb=7,
        cs=3,
        gidp=14,
        games=159,
        park_factor=0.98,
        league_rpg=4.48,
    )


@pytest.fixture
def degrom_2018() -> PlayerSeason:
    """Jacob deGrom 2018 — historic pitching season (1.70 ERA, NL Cy Young).

    Source: Baseball-Reference / FanGraphs 2018 season page.
    """
    return PlayerSeason(
        player_id="degroja01",
        player_name="Jacob deGrom",
        season=2018,
        team="NYM",
        position="P",
        # Minimal batting line (NL pitcher)
        pa=70,
        ab=63,
        hits=8,
        singles=5,
        doubles=2,
        triples=1,
        hr=0,
        bb=3,
        k=33,
        games=32,
        # Pitching line
        ip=217.0,
        er=48,
        hits_allowed=152,
        hr_allowed=10,
        bb_allowed=46,
        hbp_allowed=5,
        k_pitching=269,
        games_pitched=32,
        games_started=32,
        park_factor=0.95,
        league_rpg=4.45,
    )


@pytest.fixture
def ohtani_2023() -> PlayerSeason:
    """Shohei Ohtani 2023 — two-way player (DH + SP).

    Source: Baseball-Reference / FanGraphs 2023 season page.
    """
    return PlayerSeason(
        player_id="ohtansh01",
        player_name="Shohei Ohtani",
        season=2023,
        team="LAA",
        position="DH",
        # Hitting
        pa=599,
        ab=497,
        hits=151,
        singles=73,
        doubles=26,
        triples=8,
        hr=44,
        bb=91,
        ibb=5,
        hbp=5,
        k=143,
        sf=6,
        sh=0,
        sb=20,
        cs=6,
        games=135,
        # Pitching
        ip=132.0,
        er=46,
        hits_allowed=99,
        hr_allowed=18,
        bb_allowed=55,
        hbp_allowed=5,
        k_pitching=167,
        games_pitched=23,
        games_started=23,
        park_factor=0.98,
        league_rpg=4.62,
    )


@pytest.fixture
def hedges_catcher() -> PlayerSeason:
    """Austin Hedges archetype — elite defensive catcher, weak bat.

    Framing, blocking, and throwing stats are composites from elite
    seasons by Hedges / Jeff Mathis / Yasmani Grandal.
    """
    return PlayerSeason(
        player_id="hedgeau01",
        player_name="Austin Hedges",
        season=2021,
        team="CLE",
        position="C",
        pa=300,
        ab=270,
        hits=54,
        singles=36,
        doubles=10,
        triples=1,
        hr=7,
        bb=20,
        ibb=1,
        hbp=4,
        k=90,
        sf=3,
        sh=3,
        games=90,
        inn_fielded=750.0,
        framing_runs=15.0,
        blocking_runs=2.0,
        throwing_runs=1.5,
        catcher_pitches=10000,
        park_factor=1.0,
        league_rpg=4.53,
    )


@pytest.fixture
def replacement_player() -> PlayerSeason:
    """Generic freely-available-talent (FAT) outfielder.

    Stat line is calibrated so the expected BRAVS output is near zero.
    """
    return PlayerSeason(
        player_id="replac01",
        player_name="Replacement Player",
        season=2023,
        team="AAA",
        position="LF",
        pa=400,
        ab=365,
        hits=82,
        singles=55,
        doubles=15,
        triples=2,
        hr=10,
        bb=25,
        ibb=1,
        hbp=4,
        k=110,
        sf=4,
        sh=2,
        games=100,
        park_factor=1.0,
        league_rpg=4.62,
    )


@pytest.fixture
def short_season_2020() -> PlayerSeason:
    """2020 60-game COVID season — limited sample size.

    The short season should produce wider credible intervals due to
    higher posterior uncertainty from fewer observations.
    """
    return PlayerSeason(
        player_id="short2020",
        player_name="Short Season Guy",
        season=2020,
        team="NYY",
        position="SS",
        pa=220,
        ab=190,
        hits=50,
        singles=28,
        doubles=12,
        triples=2,
        hr=8,
        bb=22,
        ibb=2,
        hbp=3,
        k=55,
        sf=3,
        sh=2,
        games=55,
        park_factor=1.05,
        league_rpg=4.65,
    )


@pytest.fixture
def negative_value_player() -> PlayerSeason:
    """Truly terrible season — .150 BA, no power, lots of K's.

    Should produce clearly negative BRAVS.
    """
    return PlayerSeason(
        player_id="terribl01",
        player_name="Bad Hitter",
        season=2023,
        team="BAL",
        position="1B",
        pa=350,
        ab=320,
        hits=48,
        singles=30,
        doubles=10,
        triples=1,
        hr=7,
        bb=20,
        ibb=0,
        hbp=3,
        k=140,
        sf=4,
        sh=3,
        games=90,
        park_factor=1.0,
        league_rpg=4.62,
    )
