"""Leverage/context adjustment component of BRAVS.

Applies damped leverage weighting to skill-based value components.
Uses sqrt(gmLI) as the leverage multiplier, which is the geometric
mean of ignoring leverage entirely (WAR) and using it fully (WPA).
"""

from __future__ import annotations

import numpy as np

from baseball_metric.core.types import ComponentResult, PlayerSeason
from baseball_metric.utils.constants import LEVERAGE_DAMPING_EXPONENT
from baseball_metric.utils.math_helpers import damped_leverage


def compute_leverage_multiplier(player: PlayerSeason) -> float:
    """Compute the damped leverage multiplier for a player-season.

    The leverage multiplier adjusts skill-based value to account for
    when that value was deployed. A closer pitching in high-leverage
    situations gets a boost; a mop-up reliever gets a discount.

    The damping (sqrt) prevents extreme leverage from dominating,
    keeping the metric more stable than WPA while still being
    context-sensitive unlike WAR.

    Args:
        player: PlayerSeason with average leverage index.

    Returns:
        Leverage multiplier (1.0 = average leverage).
    """
    raw_li = player.avg_leverage_index
    damped = damped_leverage(raw_li, LEVERAGE_DAMPING_EXPONENT)

    # Normalize so that league-average leverage = 1.0
    # Since E[sqrt(LI)] ≈ 0.97 for the league (slightly below 1 due to Jensen's inequality),
    # we normalize to keep the average multiplier at 1.0
    avg_damped_li = 0.97  # empirical average of sqrt(gmLI) across MLB
    return damped / avg_damped_li


def compute_leverage_adjustment(
    player: PlayerSeason,
    skill_runs: float,
    skill_var: float,
    rng: np.random.Generator | None = None,
    n_samples: int = 10000,
) -> ComponentResult:
    """Compute the leverage adjustment as a separate component.

    This reports the additional runs (positive or negative) from
    leverage context beyond what the raw skill components capture.

    Leverage adjustment = skill_runs × (leverage_multiplier - 1.0)

    A positive value means the player deployed his skills in
    above-average leverage situations. A negative value means
    below-average leverage.

    Args:
        player: PlayerSeason with leverage data.
        skill_runs: Total skill-based runs (pre-leverage).
        skill_var: Variance of skill-based runs.
        rng: Random generator for reproducibility.
        n_samples: Number of posterior samples.

    Returns:
        ComponentResult with leverage adjustment runs.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    multiplier = compute_leverage_multiplier(player)
    adjustment_runs = skill_runs * (multiplier - 1.0)
    adjustment_var = skill_var * (multiplier - 1.0) ** 2

    # Samples
    skill_samples = rng.normal(skill_runs, np.sqrt(max(skill_var, 0.01)), size=n_samples)
    adjustment_samples = skill_samples * (multiplier - 1.0)

    from baseball_metric.utils.math_helpers import credible_interval

    ci50 = credible_interval(adjustment_samples, 0.50)
    ci90 = credible_interval(adjustment_samples, 0.90)

    return ComponentResult(
        name="leverage",
        runs_mean=adjustment_runs,
        runs_var=adjustment_var,
        ci_50=ci50,
        ci_90=ci90,
        samples=adjustment_samples,
        metadata={
            "raw_gmLI": round(player.avg_leverage_index, 2),
            "damped_multiplier": round(multiplier, 3),
            "adjustment_pct": round((multiplier - 1.0) * 100, 1),
        },
    )
