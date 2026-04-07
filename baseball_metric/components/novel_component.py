"""Approach Quality Index (AQI) — Novel component of BRAVS.

Measures batter decision quality on a per-pitch basis. For each pitch,
we estimate the run value difference between the decision the batter made
(swing or take) and the optimal decision given the count, pitch location,
and pitch type.

This goes beyond traditional plate discipline metrics (O-Swing%, Z-Contact%)
by weighting each decision by its run-value consequence. A chase on a
3-0 count costs more than a chase on an 0-2 count. Taking a hittable
fastball down the middle on a hitter's count is a missed opportunity
that standard metrics don't capture.

When Statcast pitch-level data is unavailable, AQI is estimated from
proxy statistics (BB%, K%, O-Swing%, Z-Contact%) with wider uncertainty.
"""

from __future__ import annotations

import numpy as np

from baseball_metric.core.types import ComponentResult, PlayerSeason
from baseball_metric.utils.constants import (
    MIN_PA_FOR_AQI,
    PRIOR_AQI_MEAN,
    PRIOR_AQI_SD,
)
from baseball_metric.utils.math_helpers import (
    bayesian_update_normal,
    credible_interval,
    normal_posterior_samples,
)

# Observation variance for AQI
# AQI is moderately noisy — stabilizes at ~300-400 PA
AQI_OBS_VARIANCE = 12.0

# Proxy model coefficients: estimate AQI from traditional stats
# when pitch-level data is unavailable
# Derived from regression of pitch-level AQI on traditional stats
PROXY_COEFF_BB_RATE = 15.0    # high BB% → good approach (+)
PROXY_COEFF_K_RATE = -8.0     # high K% → poor approach (-)
PROXY_COEFF_CHASE_RATE = -12.0  # high O-Swing% → poor approach (-)
PROXY_COEFF_ZONE_CONTACT = 5.0  # high Z-Contact% → good approach (+)
PROXY_INTERCEPT = 0.0


def _estimate_aqi_from_proxy(player: PlayerSeason) -> float | None:
    """Estimate AQI from traditional plate discipline statistics.

    Uses a linear model relating BB%, K%, chase rate, and zone contact
    rate to pitch-level AQI values. This proxy is noisier than direct
    pitch-level computation but provides coverage for historical players.

    Returns None if insufficient data.
    """
    if player.pa < MIN_PA_FOR_AQI:
        return None

    bb_rate = player.bb / max(player.pa, 1)
    k_rate = player.k / max(player.pa, 1)

    # League average rates for centering
    avg_bb_rate = 0.085
    avg_k_rate = 0.220
    avg_chase_rate = 0.30
    avg_zone_contact = 0.85

    aqi = PROXY_INTERCEPT
    aqi += PROXY_COEFF_BB_RATE * (bb_rate - avg_bb_rate)
    aqi += PROXY_COEFF_K_RATE * (k_rate - avg_k_rate)

    if player.chase_rate is not None:
        aqi += PROXY_COEFF_CHASE_RATE * (player.chase_rate - avg_chase_rate)

    if player.zone_contact_rate is not None:
        aqi += PROXY_COEFF_ZONE_CONTACT * (player.zone_contact_rate - avg_zone_contact)

    # Scale to runs per 600 PA
    return aqi * (player.pa / 600.0) * 10.0


def compute_aqi(
    player: PlayerSeason,
    rng: np.random.Generator | None = None,
    n_samples: int = 10000,
) -> ComponentResult:
    """Compute the Approach Quality Index component of BRAVS.

    If pitch-level AQI data is available (player.aqi_raw), uses that directly.
    Otherwise, estimates AQI from proxy statistics (BB%, K%, O-Swing%, Z-Contact%).

    The proxy estimation carries wider uncertainty, reflected in the posterior.

    Args:
        player: PlayerSeason with plate discipline data.
        rng: Random generator for reproducibility.
        n_samples: Number of posterior samples.

    Returns:
        ComponentResult with AQI runs above average.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if player.pa < MIN_PA_FOR_AQI:
        return ComponentResult(
            name="approach_quality",
            runs_mean=0.0,
            runs_var=PRIOR_AQI_SD ** 2,
            ci_50=(-2.0, 2.0),
            ci_90=(-4.9, 4.9),
            samples=normal_posterior_samples(0.0, PRIOR_AQI_SD ** 2, n_samples, rng),
            metadata={"note": f"insufficient PA ({player.pa} < {MIN_PA_FOR_AQI})"},
        )

    # Determine AQI observation
    if player.aqi_raw is not None:
        # Direct pitch-level AQI available
        obs_aqi = player.aqi_raw
        obs_variance = AQI_OBS_VARIANCE
        source = "statcast_pitch_level"
    else:
        # Proxy estimation
        proxy_aqi = _estimate_aqi_from_proxy(player)
        if proxy_aqi is None:
            return ComponentResult(
                name="approach_quality",
                runs_mean=0.0,
                runs_var=PRIOR_AQI_SD ** 2,
                ci_50=(-2.0, 2.0),
                ci_90=(-4.9, 4.9),
                samples=normal_posterior_samples(0.0, PRIOR_AQI_SD ** 2, n_samples, rng),
                metadata={"note": "proxy estimation failed"},
            )
        obs_aqi = proxy_aqi
        obs_variance = AQI_OBS_VARIANCE * 2.0  # wider uncertainty for proxy
        source = "proxy_model"

    # Bayesian update
    pa_scale = max(player.pa / 600.0, 0.1)
    effective_n = max(int(pa_scale * 5), 1)

    post_mean, post_var = bayesian_update_normal(
        prior_mean=PRIOR_AQI_MEAN,
        prior_var=PRIOR_AQI_SD ** 2,
        data_mean=obs_aqi,
        data_var=obs_variance,
        n=effective_n,
    )

    # Posterior samples
    samples = normal_posterior_samples(post_mean, post_var, n_samples, rng)

    ci50 = credible_interval(samples, 0.50)
    ci90 = credible_interval(samples, 0.90)

    return ComponentResult(
        name="approach_quality",
        runs_mean=post_mean,
        runs_var=post_var,
        ci_50=ci50,
        ci_90=ci90,
        samples=samples,
        metadata={
            "source": source,
            "obs_aqi_runs": round(obs_aqi, 1),
            "post_aqi_mean": round(post_mean, 1),
            "post_aqi_sd": round(float(np.sqrt(post_var)), 2),
        },
    )
