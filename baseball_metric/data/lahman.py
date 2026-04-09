"""Lahman Database integration for BRAVS.

Provides fast, offline access to complete MLB statistics from 1871-2023.
No API calls needed. Falls back to MLB Stats API only for current season.

Data source: Chadwick Bureau's baseballdatabank (cbwinslow fork)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import numpy as np

from baseball_metric.core.types import PlayerSeason

# Try datasets in order of recency:
#   1. lahman2025 (CRAN R package, 1871-2025) — most complete
#   2. stormlightlabs (1871-2024)
#   3. chadwickbureau (1871-2021)
_BASE = Path(__file__).parent.parent.parent / "data"
_2025 = _BASE / "lahman2025"
_STORM = _BASE / "baseball-main" / "data" / "lahman" / "csv"
_CHAD = _BASE / "baseballdatabank-master" / "core"
_CHAD_CONTRIB = _BASE / "baseballdatabank-master" / "contrib"

if _2025.exists() and (_2025 / "Batting.csv").exists():
    DATA_DIR = _2025
    CONTRIB_DIR = _2025
elif _STORM.exists():
    DATA_DIR = _STORM
    CONTRIB_DIR = _STORM if (_STORM / "HallOfFame.csv").exists() else _CHAD_CONTRIB
else:
    DATA_DIR = _CHAD
    CONTRIB_DIR = _CHAD_CONTRIB


@lru_cache(maxsize=1)
def _batting() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "Batting.csv")


@lru_cache(maxsize=1)
def _pitching() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "Pitching.csv")


@lru_cache(maxsize=1)
def _fielding() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "Fielding.csv")


@lru_cache(maxsize=1)
def _appearances() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "Appearances.csv")


@lru_cache(maxsize=1)
def _people() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "People.csv")


@lru_cache(maxsize=1)
def _hof() -> pd.DataFrame:
    return pd.read_csv(CONTRIB_DIR / "HallOfFame.csv")


@lru_cache(maxsize=1)
def _awards() -> pd.DataFrame:
    return pd.read_csv(CONTRIB_DIR / "AwardsPlayers.csv")


def is_available() -> bool:
    """Check if the Lahman database files exist."""
    return (DATA_DIR / "Batting.csv").exists()


def get_primary_position(player_id: str, year: int) -> str:
    """Get a player's primary position for a season from Appearances table.

    Uses games played at each position to determine primary.
    This is far more accurate than the MLB API's primaryPosition field.
    """
    app = _appearances()
    row = app[(app.playerID == player_id) & (app.yearID == year)]
    if row.empty:
        return "DH"

    pos_cols = {
        "G_c": "C", "G_1b": "1B", "G_2b": "2B", "G_3b": "3B",
        "G_ss": "SS", "G_lf": "LF", "G_cf": "CF", "G_rf": "RF",
        "G_dh": "DH", "G_p": "P",
    }

    r = row.iloc[0]
    best_pos = "DH"
    best_g = 0
    for col, pos in pos_cols.items():
        g = int(r.get(col, 0) or 0)
        if g > best_g:
            best_g = g
            best_pos = pos

    return best_pos


def search_player(name: str) -> list[dict]:
    """Search for players by name in the People table."""
    people = _people()
    name_lower = name.lower()

    # Search by last name, first name, or full name
    mask = (
        people.nameLast.str.lower().str.contains(name_lower, na=False) |
        people.nameFirst.str.lower().str.contains(name_lower, na=False) |
        (people.nameFirst.str.lower() + " " + people.nameLast.str.lower()).str.contains(name_lower, na=False)
    )
    matches = people[mask].head(20)

    results = []
    for _, p in matches.iterrows():
        results.append({
            "player_id": p.playerID,
            "name": f"{p.nameFirst} {p.nameLast}",
            "birth_year": int(p.birthYear) if pd.notna(p.birthYear) else None,
            "debut": str(p.debut) if pd.notna(p.debut) else None,
            "final_game": str(p.finalGame) if pd.notna(p.finalGame) else None,
        })
    return results


def get_player_season(player_id: str, year: int) -> PlayerSeason | None:
    """Build a PlayerSeason from Lahman data. No API calls."""
    from baseball_metric.adjustments.era_adjustment import get_rpg
    from baseball_metric.adjustments.park_factors import get_park_factor

    people = _people()
    person = people[people.playerID == player_id]
    if person.empty:
        return None
    person = person.iloc[0]
    name = f"{person.nameFirst} {person.nameLast}"

    # Get batting stats (sum stints for traded players)
    bat = _batting()
    bat_rows = bat[(bat.playerID == player_id) & (bat.yearID == year)]
    if bat_rows.empty:
        h = {}
    else:
        h = bat_rows.sum(numeric_only=True).to_dict()
        h["teamID"] = bat_rows.iloc[-1].teamID
        h["lgID"] = bat_rows.iloc[-1].lgID

    # Get pitching stats (sum stints)
    pit = _pitching()
    pit_rows = pit[(pit.playerID == player_id) & (pit.yearID == year)]
    if pit_rows.empty:
        p = {}
    else:
        p = pit_rows.sum(numeric_only=True).to_dict()
        if not h:
            p["teamID"] = pit_rows.iloc[-1].teamID
            p["lgID"] = pit_rows.iloc[-1].lgID

    if not h and not p:
        return None

    # Get position from appearances
    position = get_primary_position(player_id, year)

    # Team and league
    team = str(h.get("teamID", p.get("teamID", "UNK")))
    league = str(h.get("lgID", p.get("lgID", "MLB")))

    # Context
    league_rpg = get_rpg(year)
    pf = get_park_factor(team, year)

    # Parse IP from pitching (Lahman stores as float, not the .1/.2 notation)
    ip = float(p.get("IPouts", 0)) / 3.0 if p.get("IPouts") else 0.0

    # Known shortened seasons
    season_games = {2020: 60, 1994: 115, 1995: 144, 1981: 107}.get(year, 162)

    # Compute PA (Lahman doesn't have PA directly)
    ab = int(h.get("AB", 0) or 0)
    bb = int(h.get("BB", 0) or 0)
    hbp = int(h.get("HBP", 0) or 0)
    sf = int(h.get("SF", 0) or 0)
    sh = int(h.get("SH", 0) or 0)
    pa = ab + bb + hbp + sf + sh

    hits = int(h.get("H", 0) or 0)
    doubles = int(h.get("2B", 0) or 0)
    triples = int(h.get("3B", 0) or 0)
    hr = int(h.get("HR", 0) or 0)

    return PlayerSeason(
        player_id=player_id,
        player_name=name,
        season=year,
        team=team,
        position=position,
        # Batting
        pa=pa,
        ab=ab,
        hits=hits,
        doubles=doubles,
        triples=triples,
        hr=hr,
        bb=bb,
        ibb=int(h.get("IBB", 0) or 0),
        hbp=hbp,
        k=int(h.get("SO", 0) or 0),
        sf=sf,
        sh=sh,
        sb=int(h.get("SB", 0) or 0),
        cs=int(h.get("CS", 0) or 0),
        gidp=int(h.get("GIDP", 0) or 0),
        games=int(h.get("G", p.get("G", 0)) or 0),
        # Pitching
        ip=ip,
        er=int(p.get("ER", 0) or 0),
        hits_allowed=int(p.get("H", 0) or 0),
        hr_allowed=int(p.get("HR", 0) or 0),
        bb_allowed=int(p.get("BB", 0) or 0),
        hbp_allowed=int(p.get("HBP", 0) or 0),
        k_pitching=int(p.get("SO", 0) or 0),
        games_pitched=int(p.get("G", 0) or 0),
        games_started=int(p.get("GS", 0) or 0),
        saves=int(p.get("SV", 0) or 0),
        # Context
        park_factor=pf.overall,
        league_rpg=league_rpg,
        season_games=season_games,
        league=league,
    )


def get_all_seasons(player_id: str) -> list[int]:
    """Get all years a player appeared in."""
    bat = _batting()
    pit = _pitching()
    years = set()
    years.update(bat[bat.playerID == player_id].yearID.tolist())
    years.update(pit[pit.playerID == player_id].yearID.tolist())
    return sorted(years)


def get_qualified_batters(year: int, min_pa: int = 400) -> list[str]:
    """Get all qualified batters for a season."""
    bat = _batting()
    season = bat[bat.yearID == year].groupby("playerID").sum(numeric_only=True)
    # Compute PA
    season["PA"] = season.AB + season.BB + season.get("HBP", 0).fillna(0) + season.get("SF", 0).fillna(0) + season.get("SH", 0).fillna(0)
    qualified = season[season.PA >= min_pa]
    return qualified.index.tolist()


def get_qualified_pitchers(year: int, min_ip: float = 100.0) -> list[str]:
    """Get all qualified pitchers for a season."""
    pit = _pitching()
    season = pit[pit.yearID == year].groupby("playerID").sum(numeric_only=True)
    season["IP"] = season.IPouts / 3.0
    qualified = season[season.IP >= min_ip]
    return qualified.index.tolist()


def get_hof_inducted() -> list[str]:
    """Get all players inducted into the Hall of Fame."""
    hof = _hof()
    inducted = hof[(hof.inducted == "Y") & (hof.category == "Player")]
    return inducted.playerID.unique().tolist()


def get_hof_voting() -> pd.DataFrame:
    """Get full HOF voting history."""
    return _hof()


def get_awards(year: int | None = None) -> pd.DataFrame:
    """Get awards data, optionally filtered by year."""
    awards = _awards()
    if year:
        return awards[awards.yearID == year]
    return awards


def get_mvp_winners() -> pd.DataFrame:
    """Get all MVP winners."""
    awards = _awards()
    return awards[awards.awardID == "Most Valuable Player"]
