"""Posterior computation and sampling for BRAVS.

Handles combining component posteriors into the total BRAVS posterior,
including uncertainty propagation with component correlations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from baseball_metric.core.types import BRAVSResult, ComponentResult
from baseball_metric.utils.math_helpers import credible_interval

# Known correlations between component run values.
# Estimated from historical data: fast players hit for more infield singles
# (hitting-baserunning r≈0.25), good hitters draw walks and see more pitches
# (hitting-approach r≈0.35), catchers who frame well also block well
# (framing-blocking not modeled here since they're sub-components).
# Pairs not listed are assumed independent (r=0).
COMPONENT_CORRELATIONS: dict[tuple[str, str], float] = {
    ("hitting", "baserunning"): 0.25,   # speed → infield hits + stolen bases
    ("hitting", "approach_quality"): 0.20,  # residual correlation after orthogonalization
    ("baserunning", "approach_quality"): 0.10,  # marginal: patient hitters see more pitches
    ("fielding", "baserunning"): 0.15,   # athleticism drives both
}


def combine_component_posteriors(
    components: dict[str, ComponentResult],
    leverage_multiplier: float = 1.0,
    rpw: float = 9.8,
    rng: np.random.Generator | None = None,
    n_samples: int = 10000,
) -> tuple[float, float, NDArray[np.floating[object]]]:
    """Combine component posteriors into a total BRAVS posterior.

    Uses component correlation structure to generate a joint posterior
    via a Cholesky-correlated sampling approach. Components with known
    correlations have their samples drawn from a multivariate normal,
    then mapped back through their marginal distributions.

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

    comp_names = list(components.keys())
    n_comp = len(comp_names)

    # Build correlation matrix
    corr_matrix = np.eye(n_comp)
    for i, name_i in enumerate(comp_names):
        for j, name_j in enumerate(comp_names):
            if i == j:
                continue
            pair = (name_i, name_j)
            pair_rev = (name_j, name_i)
            if pair in COMPONENT_CORRELATIONS:
                corr_matrix[i, j] = COMPONENT_CORRELATIONS[pair]
            elif pair_rev in COMPONENT_CORRELATIONS:
                corr_matrix[i, j] = COMPONENT_CORRELATIONS[pair_rev]

    # Extract means and SDs
    means = np.array([components[n].runs_mean for n in comp_names])
    sds = np.array([max(components[n].runs_sd, 0.01) for n in comp_names])

    # Build covariance matrix from correlation matrix and marginal SDs
    cov_matrix = np.outer(sds, sds) * corr_matrix

    # Ensure positive semi-definite (numerical safety)
    eigvals = np.linalg.eigvalsh(cov_matrix)
    if np.any(eigvals < -1e-10):
        # Fall back to diagonal (no correlation) if matrix is not PSD
        cov_matrix = np.diag(sds ** 2)

    # Draw correlated samples
    try:
        joint_samples = rng.multivariate_normal(means, cov_matrix, size=n_samples)
    except np.linalg.LinAlgError:
        # Fallback: independent sampling
        joint_samples = np.column_stack([
            rng.normal(means[i], sds[i], size=n_samples) for i in range(n_comp)
        ])

    # For components that have their own posterior samples (non-normal),
    # blend: use the correlated structure for the noise but anchor to
    # the component's own samples for the marginal distribution
    for i, name in enumerate(comp_names):
        comp = components[name]
        if comp.samples is not None and len(comp.samples) >= n_samples:
            # Rank-preserve: sort the correlated samples to match the
            # rank order of the component's own samples
            own_sorted = np.sort(comp.samples[:n_samples])
            rank_order = np.argsort(np.argsort(joint_samples[:, i]))
            joint_samples[:, i] = own_sorted[rank_order]

    # Sum across components for total
    total_samples = np.sum(joint_samples, axis=1)

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
