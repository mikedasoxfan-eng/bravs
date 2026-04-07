"""Fielding component of BRAVS.

Estimates defensive value using a Bayesian ensemble of UZR, DRS, and OAA.
The ensemble weights are based on estimated reliability of each system,
and the combined estimate is heavily regressed toward zero for small samples.
"""

from __future__ import annotations

import numpy as np

from baseball_metric.core.types import ComponentResult, PlayerSeason
from baseball_metric.utils.constants import (
    DEFAULT_FIELDING_WEIGHTS,
    MIN_INNINGS_FOR_FIELDING,
    PRIOR_FIELDING_MEAN,
    PRIOR_FIELDING_SD,
)
from baseball_metric.utils.math_helpers import (
    bayesian_update_normal,
    credible_interval,
    ensemble_average,
    normal_posterior_samples,
)

# Per-season observation variance for defensive metrics
# Defensive metrics are notoriously noisy — YoY correlation ~0.4 for UZR
# This implies large observation variance relative to true talent variance
# Source: Lichtman UZR methodology papers, stabilization ~4320 innings
FIELDING_OBS_VARIANCE = 60.0  # variance per 1400-inning season


def compute_fielding(
    player: PlayerSeason,
    rng: np.random.Generator | None = None,
    n_samples: int = 10000,
) -> ComponentResult:
    """Compute the fielding component of BRAVS.

    Uses Bayesian model averaging of available defensive metrics (UZR, DRS, OAA).
    When multiple metrics are available, they are combined using reliability-based
    weights. The combined estimate is then shrunk toward zero based on innings played.

    Args:
        player: PlayerSeason with fielding statistics.
        rng: Random generator for reproducibility.
        n_samples: Number of posterior samples.

    Returns:
        ComponentResult with fielding runs above average.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Pitchers and DHs get zero fielding value (handled by positional adjustment)
    if player.position in ("DH", "P") or player.inn_fielded < 50:
        return ComponentResult(
            name="fielding",
            runs_mean=0.0,
            runs_var=PRIOR_FIELDING_SD ** 2,
            ci_50=(-3.4, 3.4),
            ci_90=(-8.2, 8.2),
            samples=normal_posterior_samples(0.0, PRIOR_FIELDING_SD ** 2, n_samples, rng),
            metadata={"note": "no fielding data (DH/P or < 50 innings)"},
        )

    # Collect available defensive metrics
    estimates: dict[str, float] = {}
    if player.uzr is not None:
        estimates["UZR"] = player.uzr
    if player.drs is not None:
        estimates["DRS"] = player.drs
    if player.oaa is not None:
        estimates["OAA"] = player.oaa

    if not estimates:
        # No defensive data available — return prior
        return ComponentResult(
            name="fielding",
            runs_mean=0.0,
            runs_var=PRIOR_FIELDING_SD ** 2,
            ci_50=(-3.4, 3.4),
            ci_90=(-8.2, 8.2),
            samples=normal_posterior_samples(0.0, PRIOR_FIELDING_SD ** 2, n_samples, rng),
            metadata={"note": "no defensive metrics available, using prior"},
        )

    # Ensemble average of available metrics
    obs_fielding = ensemble_average(estimates, DEFAULT_FIELDING_WEIGHTS)

    # Scale observation by playing time (full season = ~1400 innings fielded)
    playing_time_fraction = min(player.inn_fielded / 1400.0, 1.0)

    # The observation is already in runs per season, so we need to account
    # for the fact that partial-season players have noisier estimates
    effective_n = max(int(player.inn_fielded / MIN_INNINGS_FOR_FIELDING), 1)

    # Bayesian update
    post_mean, post_var = bayesian_update_normal(
        prior_mean=PRIOR_FIELDING_MEAN,
        prior_var=PRIOR_FIELDING_SD ** 2,
        data_mean=obs_fielding,
        data_var=FIELDING_OBS_VARIANCE,
        n=effective_n,
    )

    # Posterior samples
    samples = normal_posterior_samples(post_mean, post_var, n_samples, rng)

    ci50 = credible_interval(samples, 0.50)
    ci90 = credible_interval(samples, 0.90)

    return ComponentResult(
        name="fielding",
        runs_mean=post_mean,
        runs_var=post_var,
        ci_50=ci50,
        ci_90=ci90,
        samples=samples,
        metadata={
            "available_metrics": list(estimates.keys()),
            "ensemble_value": round(obs_fielding, 1),
            "innings_fielded": player.inn_fielded,
            "shrinkage_pct": round((1.0 - post_var / (PRIOR_FIELDING_SD ** 2)) * 100, 1),
        },
    )
