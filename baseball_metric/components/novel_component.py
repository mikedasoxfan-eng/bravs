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
wOBA-residualized proxy statistics to avoid double-counting with the
hitting component. The proxy uses the portion of plate discipline stats
(chase rate, zone contact) that is NOT already captured by wOBA.
"""

from __future__ import annotations

import numpy as np

from baseball_metric.core.types import ComponentResult, PlayerSeason
from baseball_metric.utils.constants import (
    LEAGUE_AVG_WOBA,
    MIN_PA_FOR_AQI,
    PRIOR_AQI_MEAN,
    PRIOR_AQI_SD,
    WOBA_SCALE,
)
from baseball_metric.utils.math_helpers import (
    bayesian_update_normal,
    compute_woba,
    credible_interval,
    normal_posterior_samples,
)

# Observation variance for AQI
# AQI is moderately noisy — stabilizes at ~300-400 PA
AQI_OBS_VARIANCE = 12.0

# Proxy model coefficients — RESIDUALIZED against wOBA.
# These capture the portion of approach quality that is INDEPENDENT
# of batting outcomes. BB% and K% are NOT used directly because they
# are already captured by the hitting component via wOBA.
# Instead we use:
#   - chase rate (O-Swing%): swinging at pitches outside the zone
#   - zone contact rate (Z-Contact%): making contact on hittable pitches
#   - wOBA residual: the gap between expected wOBA (from approach) and actual
# These are orthogonal to wOBA by construction.
PROXY_COEFF_CHASE_RATE = -10.0    # high O-Swing% → poor approach
PROXY_COEFF_ZONE_CONTACT = 6.0    # high Z-Contact% → good approach
PROXY_COEFF_WOBA_RESIDUAL = -8.0  # stronger penalty: skills > decisions, less AQI credit
PROXY_INTERCEPT = 0.0

# Expected wOBA model: what wOBA would we expect from BB%/K% alone?
# Used to compute the residual. Coefficients from regression of wOBA on BB%/K%.
EXPECTED_WOBA_BB_COEFF = 0.80   # each pct of BB% adds ~0.008 wOBA
EXPECTED_WOBA_K_COEFF = -0.45   # each pct of K% subtracts ~0.0045 wOBA
EXPECTED_WOBA_INTERCEPT = 0.310


def _estimate_aqi_from_proxy(player: PlayerSeason) -> float | None:
    """Estimate AQI from residualized plate discipline statistics.

    The proxy model is orthogonalized against wOBA to prevent double-counting
    with the hitting component. Instead of using BB% and K% directly (which
    drive wOBA), we:
    1. Compute expected wOBA from BB%/K% alone
    2. Compute the wOBA residual (actual - expected)
    3. Use chase rate, zone contact, and the wOBA residual as inputs

    This ensures AQI captures approach quality BEYOND what outcomes measure.

    Returns None if insufficient data.
    """
    if player.pa < MIN_PA_FOR_AQI:
        return None

    bb_rate = player.bb / max(player.pa, 1)
    k_rate = player.k / max(player.pa, 1)

    # Compute actual wOBA
    actual_woba = compute_woba(
        bb=player.ubb, hbp=player.hbp, singles=player.singles,
        doubles=player.doubles, triples=player.triples, hr=player.hr,
        ab=player.ab, sf=player.sf,
    )

    # Expected wOBA from BB%/K% alone (what the hitting component already captures)
    expected_woba = (EXPECTED_WOBA_INTERCEPT
                     + EXPECTED_WOBA_BB_COEFF * (bb_rate - 0.085)
                     + EXPECTED_WOBA_K_COEFF * (k_rate - 0.220))

    # wOBA residual: positive = outcomes exceed what plate discipline predicts
    # (e.g., great bat speed compensating for poor approach)
    woba_residual = actual_woba - expected_woba

    # League averages for centering
    avg_chase_rate = 0.30
    avg_zone_contact = 0.85

    aqi = PROXY_INTERCEPT

    # Primary signals: approach metrics that are NOT captured by wOBA
    if player.chase_rate is not None:
        aqi += PROXY_COEFF_CHASE_RATE * (player.chase_rate - avg_chase_rate)
    else:
        # Without chase rate data, use a dampened BB%/K% signal
        # but at 1/3 the weight to minimize double-counting
        # Minimal fallback: very dampened to avoid double-counting with wOBA
        aqi += 1.5 * (bb_rate - 0.085) + (-1.0) * (k_rate - 0.220)

    if player.zone_contact_rate is not None:
        aqi += PROXY_COEFF_ZONE_CONTACT * (player.zone_contact_rate - avg_zone_contact)

    # Residual adjustment: if outcomes exceed approach prediction, discount AQI
    # (the player has good bat skills, not necessarily good decisions)
    aqi += PROXY_COEFF_WOBA_RESIDUAL * woba_residual

    # Scale to runs per 600 PA (dampened to 3.0 for better orthogonalization)
    return aqi * (player.pa / 600.0) * 3.0


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
