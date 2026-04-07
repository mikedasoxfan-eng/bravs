"""Era normalization for BRAVS.

Adjusts run values to account for different run environments across eras.
All values are normalized relative to an anchor season (2023 by default).

The core insight: in a high-run environment (e.g., 2000: 5.14 R/G),
each marginal run is worth fewer wins than in a low-run environment
(e.g., 1968: 3.42 R/G). This affects both the value of offensive
contributions and the runs-per-win conversion.
"""

from __future__ import annotations

from baseball_metric.utils.constants import ERA_ANCHOR_RPG, ERA_ANCHOR_SEASON

# Historical MLB runs per game by season
# Source: Baseball-Reference league totals
HISTORICAL_RPG: dict[int, float] = {
    1920: 4.39, 1925: 5.06, 1930: 5.55, 1935: 5.08, 1940: 4.65,
    1945: 4.09, 1950: 4.84, 1955: 4.44, 1960: 4.24, 1965: 3.89,
    1968: 3.42, 1969: 4.07, 1970: 4.34, 1975: 4.12, 1980: 4.29,
    1985: 4.33, 1990: 4.26, 1995: 4.85, 1996: 5.04, 1997: 4.77,
    1998: 4.79, 1999: 5.08, 2000: 5.14, 2001: 4.78, 2002: 4.62,
    2003: 4.73, 2004: 4.81, 2005: 4.59, 2006: 4.86, 2007: 4.80,
    2008: 4.65, 2009: 4.61, 2010: 4.38, 2011: 4.28, 2012: 4.32,
    2013: 4.17, 2014: 4.07, 2015: 4.25, 2016: 4.48, 2017: 4.65,
    2018: 4.45, 2019: 4.83, 2020: 4.65, 2021: 4.26, 2022: 4.28,
    2023: 4.62, 2024: 4.52, 2025: 4.45,
}


def get_rpg(season: int) -> float:
    """Get runs per game for a season, interpolating if needed."""
    if season in HISTORICAL_RPG:
        return HISTORICAL_RPG[season]

    # Interpolate between known seasons
    known = sorted(HISTORICAL_RPG.keys())
    if season < known[0]:
        return HISTORICAL_RPG[known[0]]
    if season > known[-1]:
        return HISTORICAL_RPG[known[-1]]

    # Linear interpolation
    for i in range(len(known) - 1):
        if known[i] <= season <= known[i + 1]:
            t = (season - known[i]) / (known[i + 1] - known[i])
            return HISTORICAL_RPG[known[i]] * (1 - t) + HISTORICAL_RPG[known[i + 1]] * t

    return ERA_ANCHOR_RPG


def era_run_multiplier(season: int) -> float:
    """Compute the era adjustment multiplier for run values.

    Scales run values so that contributions in different run environments
    are comparable. In a low-run environment, each run is worth more,
    so raw run values are scaled up (multiplier > 1).

    multiplier = anchor_RPG / season_RPG

    Args:
        season: The season to compute the multiplier for.

    Returns:
        Era adjustment multiplier (1.0 for anchor season).
    """
    season_rpg = get_rpg(season)
    return ERA_ANCHOR_RPG / season_rpg


def era_adjusted_runs(runs: float, season: int) -> float:
    """Adjust a run value to the anchor era.

    Args:
        runs: Raw runs value in the original era.
        season: The season in which the runs were produced.

    Returns:
        Era-adjusted runs (normalized to anchor season run environment).
    """
    return runs * era_run_multiplier(season)
