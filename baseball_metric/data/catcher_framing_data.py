"""Catcher framing data overlay for BRAVS.

Loads Statcast catcher framing data (runs_extra_strikes) and maps
catcher MLBAM IDs to Lahman IDs via the Chadwick crosswalk.

Available for 2015-2025 (Statcast era).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

_BASE = Path(__file__).parent.parent.parent / "data"
_STATCAST_DIR = _BASE / "statcast"
_CROSSWALK_PATH = _BASE / "id_crosswalk.csv"


@lru_cache(maxsize=1)
def _load_crosswalk() -> dict[int, str]:
    """Load mlbam_id -> lahman_id mapping from the Chadwick crosswalk."""
    path = _CROSSWALK_PATH
    if not path.exists():
        log.warning("Crosswalk file not found at %s", path)
        return {}
    df = pd.read_csv(path)
    mapping = {}
    for _, row in df.iterrows():
        mlbam = row["mlbam_id"]
        lahman = row["lahman_id"]
        if pd.notna(mlbam) and pd.notna(lahman):
            mapping[int(mlbam)] = str(lahman)
    log.info("Crosswalk loaded: %d mlbam->lahman mappings", len(mapping))
    return mapping


@lru_cache(maxsize=1)
def _load_catcher_framing() -> dict[tuple[str, int], dict]:
    """Load catcher framing CSV and build (lahman_id, year) -> stats lookup.

    The CSV has these key columns:
        player_id:          MLBAM catcher ID
        year:               season
        n_called_pitches:   total called pitches received
        runs_extra_strikes: framing runs above average (the key metric)

    Returns dict mapping (playerID, yearID) to:
        framing_runs:     runs_extra_strikes (runs above average from framing)
        n_called_pitches: total called pitches
    """
    path = _STATCAST_DIR / "catcher_framing_all.csv"
    if not path.exists():
        log.warning("Catcher framing CSV not found at %s", path)
        return {}

    df = pd.read_csv(path)
    crosswalk = _load_crosswalk()

    lookup: dict[tuple[str, int], dict] = {}
    for _, row in df.iterrows():
        player_id = row.get("player_id")
        year = row.get("year")

        # Skip League Average rows (no player_id) and rows with missing data
        if pd.isna(player_id) or pd.isna(year):
            continue

        mlbam_id = int(float(player_id))
        year = int(year)
        lahman_id = crosswalk.get(mlbam_id)
        if lahman_id is None:
            continue

        framing_runs = row.get("runs_extra_strikes")
        if pd.isna(framing_runs):
            continue

        n_called = row.get("n_called_pitches")
        n_called = int(float(n_called)) if pd.notna(n_called) else 0

        lookup[(lahman_id, year)] = {
            "framing_runs": float(framing_runs),
            "n_called_pitches": n_called,
        }

    log.info("Catcher framing loaded: %d catcher-seasons mapped to Lahman IDs", len(lookup))
    return lookup


def get_framing_runs(playerID: str, yearID: int) -> float | None:
    """Return framing runs above average for a catcher-season.

    Uses the Statcast `runs_extra_strikes` metric, which measures how many
    extra runs a catcher saved (or cost) through pitch framing compared to
    the league-average catcher.

    Positive = better framing than average.
    Negative = worse framing than average.

    Returns None if framing data is unavailable for this player-season.
    """
    lookup = _load_catcher_framing()
    entry = lookup.get((playerID, yearID))
    if entry is None:
        return None
    return round(entry["framing_runs"], 1)


def get_framing_stats(playerID: str, yearID: int) -> dict | None:
    """Return the full framing stats dict for a catcher-season.

    Returns None if unavailable.
    """
    lookup = _load_catcher_framing()
    return lookup.get((playerID, yearID))
