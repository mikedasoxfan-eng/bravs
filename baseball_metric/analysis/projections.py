"""BRAVS Projection System — Aging curves and future value estimation.

Uses established aging curve research (Tango, Lichtman, JC Bradbury)
to project future BRAVS values from current performance.

Key assumptions:
- Hitters peak at age 27, decline ~0.5 wins/year after 30
- Pitchers peak at age 27, decline ~0.3 wins/year after 30
- Speed declines faster than power (steeper baserunning curve)
- Defense declines sharply after 32
- Injury risk increases with age (durability component)
"""

from __future__ import annotations

import numpy as np


# Aging multipliers by age relative to peak production
# Source: Tango/Lichtman aging curves, adapted for BRAVS components
# Index: age - 20 (so index 0 = age 20, index 7 = age 27 = peak)
HITTING_AGING = {
    20: 0.70, 21: 0.78, 22: 0.85, 23: 0.90, 24: 0.94,
    25: 0.97, 26: 0.99, 27: 1.00, 28: 0.99, 29: 0.97,
    30: 0.94, 31: 0.90, 32: 0.86, 33: 0.81, 34: 0.76,
    35: 0.70, 36: 0.63, 37: 0.56, 38: 0.48, 39: 0.40,
    40: 0.32, 41: 0.25, 42: 0.18,
}

PITCHING_AGING = {
    20: 0.65, 21: 0.75, 22: 0.83, 23: 0.89, 24: 0.93,
    25: 0.96, 26: 0.98, 27: 1.00, 28: 0.99, 29: 0.97,
    30: 0.95, 31: 0.92, 32: 0.88, 33: 0.84, 34: 0.79,
    35: 0.73, 36: 0.66, 37: 0.58, 38: 0.50, 39: 0.42,
    40: 0.34, 41: 0.26, 42: 0.18,
}

SPEED_AGING = {
    20: 0.95, 21: 0.97, 22: 0.99, 23: 1.00, 24: 1.00,
    25: 0.99, 26: 0.97, 27: 0.95, 28: 0.91, 29: 0.87,
    30: 0.82, 31: 0.76, 32: 0.69, 33: 0.62, 34: 0.55,
    35: 0.47, 36: 0.39, 37: 0.31, 38: 0.24, 39: 0.17,
    40: 0.10,
}

DEFENSE_AGING = {
    20: 0.90, 21: 0.93, 22: 0.96, 23: 0.98, 24: 0.99,
    25: 1.00, 26: 1.00, 27: 0.99, 28: 0.97, 29: 0.94,
    30: 0.90, 31: 0.85, 32: 0.78, 33: 0.70, 34: 0.60,
    35: 0.50, 36: 0.39, 37: 0.28, 38: 0.18, 39: 0.08,
    40: 0.00,
}

# Injury probability by age (chance of missing 30+ games)
INJURY_RISK = {
    20: 0.10, 21: 0.10, 22: 0.10, 23: 0.10, 24: 0.12,
    25: 0.13, 26: 0.14, 27: 0.15, 28: 0.17, 29: 0.19,
    30: 0.22, 31: 0.25, 32: 0.29, 33: 0.33, 34: 0.38,
    35: 0.43, 36: 0.48, 37: 0.54, 38: 0.60, 39: 0.66,
    40: 0.72,
}


def _get_aging_factor(curve: dict[int, float], age: int) -> float:
    """Get aging factor for a given age, interpolating if needed."""
    if age in curve:
        return curve[age]
    ages = sorted(curve.keys())
    if age < ages[0]:
        return curve[ages[0]]
    if age > ages[-1]:
        return max(curve[ages[-1]] - 0.05 * (age - ages[-1]), 0.0)
    # Interpolate
    for i in range(len(ages) - 1):
        if ages[i] <= age <= ages[i + 1]:
            t = (age - ages[i]) / (ages[i + 1] - ages[i])
            return curve[ages[i]] * (1 - t) + curve[ages[i + 1]] * t
    return 0.5


def project_bravs(
    current_bravs: float,
    current_age: int,
    is_pitcher: bool = False,
    years_forward: int = 5,
    current_components: dict[str, float] | None = None,
) -> list[dict]:
    """Project future BRAVS values using aging curves.

    Args:
        current_bravs: Current season BRAVS value.
        current_age: Player's current age.
        is_pitcher: Whether the player is primarily a pitcher.
        years_forward: How many years to project.
        current_components: Optional component breakdown for more precise projection.

    Returns:
        List of dicts with projected BRAVS for each future year.
    """
    rng = np.random.default_rng(42)
    projections = []

    # Get current aging factor to normalize
    if is_pitcher:
        current_factor = _get_aging_factor(PITCHING_AGING, current_age)
    else:
        current_factor = _get_aging_factor(HITTING_AGING, current_age)

    # Estimate "peak talent" level
    if current_factor > 0:
        peak_bravs = current_bravs / current_factor
    else:
        peak_bravs = current_bravs

    for yr in range(1, years_forward + 1):
        future_age = current_age + yr

        if is_pitcher:
            age_factor = _get_aging_factor(PITCHING_AGING, future_age)
        else:
            age_factor = _get_aging_factor(HITTING_AGING, future_age)

        # Base projection
        projected = peak_bravs * age_factor

        # Injury risk reduces expected value
        injury_prob = _get_aging_factor(INJURY_RISK, future_age)
        # If injured, lose ~40% of season value
        expected_bravs = projected * (1 - injury_prob * 0.40)

        # Uncertainty increases with projection distance
        uncertainty_sd = abs(current_bravs) * 0.15 * np.sqrt(yr)

        # 90% CI
        ci_lo = expected_bravs - 1.645 * uncertainty_sd
        ci_hi = expected_bravs + 1.645 * uncertainty_sd

        projections.append({
            "year": yr,
            "age": future_age,
            "projected_bravs": round(expected_bravs, 1),
            "ci_lo": round(ci_lo, 1),
            "ci_hi": round(ci_hi, 1),
            "aging_factor": round(age_factor, 3),
            "injury_risk": round(injury_prob, 2),
        })

    return projections


def remaining_career_value(
    current_bravs: float,
    current_age: int,
    is_pitcher: bool = False,
    retirement_threshold: float = 0.5,
) -> dict:
    """Estimate remaining career BRAVS value.

    Projects until BRAVS drops below the retirement threshold.
    """
    total = 0.0
    years = 0
    projections = project_bravs(current_bravs, current_age, is_pitcher, years_forward=15)

    for proj in projections:
        if proj["projected_bravs"] < retirement_threshold:
            break
        total += proj["projected_bravs"]
        years += 1

    return {
        "remaining_bravs": round(total, 1),
        "remaining_war_eq": round(total * 0.62, 1),
        "expected_years": years,
        "retirement_age": current_age + years,
        "projections": projections[:years + 1],
    }
