"""Statcast expected batting overlay for BRAVS.

Loads Statcast expected batting stats (xwOBA, xBA, xSLG) and converts
the xwOBA-wOBA gap into a luck-adjusted hitting runs value.

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

# wOBA scale — used to convert wOBA differences to runs.
# Standard value from the FanGraphs methodology; the GPU engine uses 1.157.
WOBA_SCALE = 1.157


@lru_cache(maxsize=1)
def _load_crosswalk() -> dict[int, str]:
    """Load mlbam_id -> lahman_id mapping from the Chadwick crosswalk."""
    path = _CROSSWALK_PATH
    if not path.exists():
        log.warning("Crosswalk file not found at %s", path)
        return {}
    df = pd.read_csv(path)
    # mlbam_id,lahman_id
    mapping = {}
    for _, row in df.iterrows():
        mlbam = row["mlbam_id"]
        lahman = row["lahman_id"]
        if pd.notna(mlbam) and pd.notna(lahman):
            mapping[int(mlbam)] = str(lahman)
    log.info("Crosswalk loaded: %d mlbam->lahman mappings", len(mapping))
    return mapping


@lru_cache(maxsize=1)
def _load_expected_batting() -> dict[tuple[str, int], dict]:
    """Load expected batting CSV and build (lahman_id, year) -> stats lookup.

    Returns dict mapping (playerID, yearID) to:
        xwOBA: expected wOBA (est_woba)
        xBA:   expected BA (est_ba)
        xSLG:  expected SLG (est_slg)
        woba:  actual wOBA
        woba_diff: est_woba - woba (positive = unlucky)
        pa:    plate appearances
    """
    path = _STATCAST_DIR / "expected_batting_all.csv"
    if not path.exists():
        log.warning("Expected batting CSV not found at %s", path)
        return {}

    df = pd.read_csv(path)
    crosswalk = _load_crosswalk()

    lookup: dict[tuple[str, int], dict] = {}
    for _, row in df.iterrows():
        mlbam_id = row["player_id"]
        year = row["year"]
        if pd.isna(mlbam_id) or pd.isna(year):
            continue

        mlbam_id = int(mlbam_id)
        year = int(year)
        lahman_id = crosswalk.get(mlbam_id)
        if lahman_id is None:
            continue

        woba = float(row["woba"]) if pd.notna(row["woba"]) else None
        est_woba = float(row["est_woba"]) if pd.notna(row["est_woba"]) else None
        if woba is None or est_woba is None:
            continue

        lookup[(lahman_id, year)] = {
            "xwOBA": est_woba,
            "xBA": float(row["est_ba"]) if pd.notna(row.get("est_ba")) else None,
            "xSLG": float(row["est_slg"]) if pd.notna(row.get("est_slg")) else None,
            "woba": woba,
            "woba_diff": est_woba - woba,
            "pa": int(row["pa"]) if pd.notna(row["pa"]) else 0,
        }

    log.info("Expected batting loaded: %d player-seasons mapped to Lahman IDs", len(lookup))
    return lookup


def get_statcast_hitting_adjustment(playerID: str, yearID: int) -> float | None:
    """Return the xwOBA-based luck adjustment in runs for a player-season.

    The adjustment = (xwOBA - wOBA) / wOBA_scale * PA

    This represents "luck-adjusted hitting runs":
      - Positive value = player was unlucky (deserved more runs)
      - Negative value = player was lucky (got more runs than expected)

    Returns None if Statcast data is unavailable for this player-season.
    """
    lookup = _load_expected_batting()
    entry = lookup.get((playerID, yearID))
    if entry is None:
        return None

    woba_diff = entry["woba_diff"]  # est_woba - woba
    pa = entry["pa"]
    if pa <= 0:
        return 0.0

    # Convert wOBA difference to runs: delta_wOBA / wOBA_scale * PA
    adjustment_runs = woba_diff / WOBA_SCALE * pa
    return round(adjustment_runs, 2)


def get_statcast_stats(playerID: str, yearID: int) -> dict | None:
    """Return the full Statcast expected batting stats for a player-season.

    Returns None if unavailable.
    """
    lookup = _load_expected_batting()
    return lookup.get((playerID, yearID))
