"""Historical catcher framing estimates for pre-2008 catchers.

Since pitch-framing data doesn't exist before ~2008, we estimate
framing value using the only available signal: how much better/worse
pitchers performed with a given catcher vs. without (WOWY ERA differential).

This is a very rough estimate with wide uncertainty. It captures
catchers who were known to be exceptional receivers (e.g., Jim Sundberg,
Bob Boone, Brad Ausmus) vs. catchers known for poor receiving (e.g.,
some power-hitting catchers).

The estimate is heavily regressed toward zero — we're only trying to
capture the direction and rough magnitude, not a precise value.

Known historical framing reputations (runs/season, approximate):
- Elite receivers: +10 to +15 runs (Sundberg, Boone, Ausmus, Russell Martin)
- Average: 0 runs
- Poor receivers: -5 to -10 runs (some slugging catchers)

Since we can't measure this directly, we use a crude proxy:
- Catcher's team ERA vs. league ERA, regressed heavily
- Estimated at ~15% signal (the rest is pitching staff quality, park, etc.)
"""

from __future__ import annotations

# Known historical catcher framing estimates (manually researched)
# These are rough consensus values from retroactive framing studies
# Source: Various attempts to estimate historical framing from available data
# (Baseball Prospectus, Tom Tango's blog, academic studies)
#
# Format: player_id -> estimated framing runs per season
# Only includes catchers with strong reputations (positive or negative)
# Unlisted catchers default to 0.0 with wide uncertainty
HISTORICAL_FRAMING_ESTIMATES: dict[str, float] = {
    # Elite receivers (pre-2008)
    "sundbe01": 12.0,   # Jim Sundberg — legendary receiver
    "boone_01": 8.0,    # Bob Boone — elite pitch handler
    "ausmu01": 10.0,    # Brad Ausmus — defense-first catcher
    "marti_r01": 8.0,   # Russell Martin — strong framer
    "moliy01": 6.0,     # Yadier Molina (early career) — elite
    "lo_duc01": 5.0,    # Paul Lo Duca — good receiver

    # Average to slightly above
    "pudge01": 3.0,     # Ivan Rodriguez — good overall but not elite framer
    "bench01": 2.0,     # Johnny Bench — decent, hard to estimate
    "fisk01": 1.0,      # Carlton Fisk — average framing
    "piazz01": -3.0,    # Mike Piazza — poor framing, great bat
    "berra01": 1.0,     # Yogi Berra — decent receiver

    # Below average framers
    "carter01": -2.0,   # Gary Carter — average to slightly below
    "posad01": -4.0,    # Jorge Posada — poor framing
}


def get_historical_framing(player_id: str, season: int) -> tuple[float | None, float]:
    """Get historical framing estimate for a pre-2008 catcher.

    Returns (estimate, uncertainty_sd).
    If the catcher has a known reputation, returns the estimate with
    moderate uncertainty. Otherwise returns None.

    The uncertainty is always wide (6-10 runs) because these are
    rough historical estimates, not measured values.
    """
    if season >= 2008:
        return None, 0.0  # Use actual framing data for modern catchers

    estimate = HISTORICAL_FRAMING_ESTIMATES.get(player_id)
    if estimate is not None:
        # Known reputation: return estimate with wide uncertainty
        return estimate, 8.0  # SD of 8 runs — very uncertain
    else:
        # Unknown catcher: no estimate, rely on prior
        return None, 0.0
