"""League-level calibration for BRAVS.

Handles AL/NL differences (particularly in the pre-universal DH era),
inter-league play effects, and overall league quality adjustments.
"""

from __future__ import annotations

# AL-NL run scoring differential by era
# Source: Baseball-Reference league totals
# Positive means AL scored more runs per game than NL
LEAGUE_DIFFERENTIAL: dict[str, float] = {
    "pre_dh": 0.0,       # pre-1973: same rules
    "dh_split": 0.25,    # 1973-2021: AL had DH, ~0.25 RPG higher
    "universal_dh": 0.0, # 2022+: universal DH, no structural difference
}


def league_adjustment(
    league: str,
    season: int,
) -> float:
    """Compute league adjustment factor.

    In the DH-split era (1973-2021), AL position players benefited from
    not having the pitcher bat, inflating offensive stats slightly.
    NL pitchers benefited from facing lineups with the pitcher batting.

    Returns adjustment in runs (positive = credit, negative = penalty).
    For a full-season player, this is typically small (±2 runs max).

    Args:
        league: "AL" or "NL".
        season: Season year.

    Returns:
        Run adjustment (add to player's value).
    """
    if season < 1973 or season >= 2022:
        return 0.0

    # DH-split era: adjust for league context
    # An AL hitter faces weaker lineups (no pitcher in opposing lineup)
    # but this is small and largely captured in park/league stats
    if league == "NL":
        return 1.0  # small credit for NL hitters facing pitcher-included lineups
    elif league == "AL":
        return -1.0  # small penalty for AL hitters in DH-enhanced lineups
    return 0.0
