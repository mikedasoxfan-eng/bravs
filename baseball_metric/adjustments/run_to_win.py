"""Dynamic runs-per-win conversion for BRAVS.

Replaces the static ~10 runs = 1 win assumption with a context-dependent
conversion derived from the Pythagorean expectation.

The key insight: in a low-scoring environment, each marginal run is
worth more wins than in a high-scoring environment. This matters for
cross-era comparisons and for comparing players on different teams.
"""

from __future__ import annotations

from baseball_metric.adjustments.era_adjustment import get_rpg
from baseball_metric.utils.math_helpers import pythagorean_rpw


def dynamic_rpw(
    season: int,
    park_factor: float = 1.0,
    team_rpg: float | None = None,
) -> float:
    """Compute context-dependent runs-per-win.

    Uses the derivative of the Pythagorean win formula:
    RPW = 2 × RPG / exponent

    Where RPG is adjusted for park factor and optionally team-specific
    run environment.

    Args:
        season: Season year (for league run environment).
        park_factor: Park factor (>1 = hitter-friendly).
        team_rpg: Optional team-specific runs per game. If None,
                  uses league average.

    Returns:
        Dynamic runs-per-win value.
    """
    league_rpg = get_rpg(season)

    if team_rpg is not None:
        rpg = team_rpg
    else:
        rpg = league_rpg

    # Adjust for park factor
    rpg_adjusted = rpg * park_factor

    # Pythagorean RPW with Pythagenpat exponent
    # Pythagenpat: exponent = RPG^0.287 (Smyth/Patriot)
    exponent = rpg_adjusted ** 0.287

    return pythagorean_rpw(rpg_adjusted, exponent)
