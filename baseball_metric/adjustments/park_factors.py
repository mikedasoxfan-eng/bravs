"""Multi-dimensional park factor model for BRAVS.

Traditional park factors apply a single scalar multiplier. This module
provides park factors broken down by:
- Overall (batting runs)
- Home run factor (HR-specific adjustment)
- Handedness (LHB vs RHB)

When detailed park data is unavailable, falls back to the overall scalar.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParkFactor:
    """Multi-dimensional park factor for a specific park-season."""

    park_id: str
    season: int
    overall: float = 1.0       # general batting factor (1.0 = neutral)
    hr_factor: float = 1.0     # home run factor
    lhb_factor: float = 1.0    # left-handed batter factor
    rhb_factor: float = 1.0    # right-handed batter factor

    def for_batter(self, bats: str = "R") -> float:
        """Get park factor for a specific batter handedness."""
        if bats == "L":
            return self.lhb_factor
        elif bats == "R":
            return self.rhb_factor
        else:
            return self.overall  # switch hitters use overall


# Known park factors for selected parks/seasons
# Source: FanGraphs park factor data, ESPN park factors
# Format: (park_id, season) -> ParkFactor
PARK_FACTOR_DB: dict[tuple[str, int], ParkFactor] = {
    ("COL", 2023): ParkFactor("COL", 2023, 1.16, 1.25, 1.18, 1.14),  # Coors Field
    ("NYY", 2023): ParkFactor("NYY", 2023, 1.05, 1.15, 1.10, 1.02),  # Yankee Stadium
    ("OAK", 2023): ParkFactor("OAK", 2023, 0.92, 0.85, 0.91, 0.93),  # Oakland Coliseum
    ("SF", 2023): ParkFactor("SF", 2023, 0.93, 0.80, 0.90, 0.95),    # Oracle Park
    ("MIL", 2023): ParkFactor("MIL", 2023, 1.02, 1.08, 1.03, 1.01),  # AmFam Field
    ("CIN", 2023): ParkFactor("CIN", 2023, 1.08, 1.12, 1.10, 1.06),  # GABP
    ("HOU", 2023): ParkFactor("HOU", 2023, 0.98, 1.02, 0.97, 0.99),  # Minute Maid
    ("LAD", 2023): ParkFactor("LAD", 2023, 0.97, 0.92, 0.96, 0.98),  # Dodger Stadium
}


def get_park_factor(team: str, season: int) -> ParkFactor:
    """Look up park factor for a team-season.

    Falls back to neutral (1.0) if no data is available.
    """
    key = (team, season)
    if key in PARK_FACTOR_DB:
        return PARK_FACTOR_DB[key]
    # Try team with default season
    for (t, _s), pf in PARK_FACTOR_DB.items():
        if t == team:
            return pf
    # Neutral park
    return ParkFactor(team, season)
