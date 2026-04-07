"""Posterior computation and sampling for BRAVS.

Handles combining component posteriors into the total BRAVS posterior,
including uncertainty propagation and joint posterior estimation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from baseball_metric.core.types import BRAVSResult, ComponentResult
from baseball_metric.utils.math_helpers import credible_interval


def combine_component_posteriors(
    components: dict[str, ComponentResult],
    leverage_multiplier: float = 1.0,
    rpw: float = 9.8,
    rng: np.random.Generator | None = None,
    n_samples: int = 10000,
) -> tuple[float, float, NDArray[np.floating[object]]]:
    """Combine component posteriors into a total BRAVS posterior.

    Since components are approximately independent, we sum their samples
    element-wise to get the joint posterior of total runs. For components
    where we have samples, we use those directly. For components with
    only mean/variance, we generate normal samples.

    The leverage multiplier is applied to skill-based components
    (hitting, pitching, baserunning, fielding, catcher, AQI).
    Durability and positional adjustments are not leverage-adjusted.

    Args:
        components: Dict of component name -> ComponentResult.
        leverage_multiplier: Damped leverage multiplier.
        rpw: Runs per win for conversion.
        rng: Random generator for reproducibility.
        n_samples: Number of posterior samples.

    Returns:
        Tuple of (total_runs_mean, total_runs_var, total_runs_samples).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    total_samples = np.zeros(n_samples)

    # Sum all component samples directly.
    # The leverage component already captures the adjustment delta
    # (skill_runs * (multiplier - 1)), so we do NOT re-multiply skill
    # components here — that would double-count the leverage effect.
    for name, comp in components.items():
        if comp.samples is not None and len(comp.samples) >= n_samples:
            total_samples += comp.samples[:n_samples]
        else:
            total_samples += rng.normal(comp.runs_mean, comp.runs_sd, size=n_samples)

    total_mean = float(np.mean(total_samples))
    total_var = float(np.var(total_samples))

    return total_mean, total_var, total_samples


def posterior_summary(
    samples: NDArray[np.floating[object]],
    rpw: float = 9.8,
) -> dict[str, object]:
    """Generate a comprehensive summary of the posterior distribution.

    Args:
        samples: Posterior samples in runs.
        rpw: Runs per win conversion.

    Returns:
        Dict with summary statistics.
    """
    wins_samples = samples / rpw

    return {
        "runs_mean": round(float(np.mean(samples)), 1),
        "runs_median": round(float(np.median(samples)), 1),
        "runs_sd": round(float(np.std(samples)), 1),
        "runs_ci_50": credible_interval(samples, 0.50),
        "runs_ci_90": credible_interval(samples, 0.90),
        "wins_mean": round(float(np.mean(wins_samples)), 1),
        "wins_median": round(float(np.median(wins_samples)), 1),
        "wins_sd": round(float(np.std(wins_samples)), 2),
        "wins_ci_50": credible_interval(wins_samples, 0.50),
        "wins_ci_90": credible_interval(wins_samples, 0.90),
        "prob_positive": round(float(np.mean(wins_samples > 0)), 3),
        "prob_above_3": round(float(np.mean(wins_samples > 3)), 3),
        "prob_above_5": round(float(np.mean(wins_samples > 5)), 3),
        "prob_above_8": round(float(np.mean(wins_samples > 8)), 3),
    }
