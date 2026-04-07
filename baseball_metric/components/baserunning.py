"""Baserunning component of BRAVS.

Estimates baserunning value in runs above average using stolen base
run values, extra-base advance rates, and GIDP avoidance.
"""

from __future__ import annotations

import numpy as np

from baseball_metric.core.types import ComponentResult, PlayerSeason
from baseball_metric.utils.constants import (
    PRIOR_BASERUNNING_MEAN,
    PRIOR_BASERUNNING_SD,
    RUN_VALUE_CAUGHT_STEALING,
    RUN_VALUE_STOLEN_BASE,
)
from baseball_metric.utils.math_helpers import (
    bayesian_update_normal,
    credible_interval,
    normal_posterior_samples,
)

# Average extra-base advance rate (taking extra base on hit)
# Source: FanGraphs baserunning stats, league average ~40% of opportunities
AVG_EXTRA_BASE_RATE = 0.40

# Run value per extra base taken vs not taken
# Source: RE24 difference between advancing an extra base or not (~0.18 runs)
RUN_VALUE_EXTRA_BASE = 0.18

# Run value per out on bases (negative)
RUN_VALUE_OUT_ON_BASES = -0.45

# Average GIDP rate per opportunity (approximately)
AVG_GIDP_RATE = 0.11
RUN_VALUE_GIDP = -0.37  # additional run cost of GIDP vs regular out

# Observation variance for baserunning (per 600 PA equivalent)
BASERUNNING_OBS_VARIANCE = 9.0


def compute_baserunning(
    player: PlayerSeason,
    rng: np.random.Generator | None = None,
    n_samples: int = 10000,
) -> ComponentResult:
    """Compute the baserunning component of BRAVS.

    Combines three sub-components:
    1. Stolen base runs: SB * runSB + CS * runCS
    2. Advance runs: extra bases taken/opportunities above average
    3. GIDP avoidance: GIDP above/below expected rate

    All sub-components are summed and then Bayesian-shrunk toward
    a population prior to handle small samples.

    Args:
        player: PlayerSeason with baserunning statistics.
        rng: Random generator for reproducibility.
        n_samples: Number of posterior samples.

    Returns:
        ComponentResult with baserunning runs above average.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Sub-component 1: Stolen base value
    sb_runs = player.sb * RUN_VALUE_STOLEN_BASE + player.cs * RUN_VALUE_CAUGHT_STEALING

    # Sub-component 2: Extra-base advance value
    if player.extra_base_opportunities > 0:
        player_rate = player.extra_bases_taken / player.extra_base_opportunities
        advance_runs = (player_rate - AVG_EXTRA_BASE_RATE) * player.extra_base_opportunities * RUN_VALUE_EXTRA_BASE
    else:
        advance_runs = 0.0

    # Out-on-bases penalty
    oob_runs = player.outs_on_bases * RUN_VALUE_OUT_ON_BASES

    # Sub-component 3: GIDP avoidance
    # Estimate GIDP opportunities as ~30% of PA with runner on 1st (rough proxy: PA * 0.15)
    gidp_opportunities = max(player.pa * 0.15, 1.0)
    expected_gidp = gidp_opportunities * AVG_GIDP_RATE
    gidp_runs = (expected_gidp - player.gidp) * abs(RUN_VALUE_GIDP)

    # Raw total
    raw_runs = sb_runs + advance_runs + oob_runs + gidp_runs

    # Bayesian shrinkage
    # Scale observation to per-600-PA rate, update, scale back
    pa_scale = max(player.pa, 1) / 600.0
    obs_mean = raw_runs / pa_scale if pa_scale > 0 else 0.0

    post_mean_per600, post_var_per600 = bayesian_update_normal(
        prior_mean=PRIOR_BASERUNNING_MEAN,
        prior_var=PRIOR_BASERUNNING_SD ** 2,
        data_mean=obs_mean,
        data_var=BASERUNNING_OBS_VARIANCE,
        n=max(int(pa_scale * 10), 1),  # effective observations
    )

    total_runs_mean = post_mean_per600 * pa_scale
    total_runs_var = post_var_per600 * (pa_scale ** 2)

    # Posterior samples
    samples_per600 = normal_posterior_samples(post_mean_per600, post_var_per600, n_samples, rng)
    runs_samples = samples_per600 * pa_scale

    ci50 = credible_interval(runs_samples, 0.50)
    ci90 = credible_interval(runs_samples, 0.90)

    return ComponentResult(
        name="baserunning",
        runs_mean=total_runs_mean,
        runs_var=total_runs_var,
        ci_50=ci50,
        ci_90=ci90,
        samples=runs_samples,
        metadata={
            "sb_runs": round(sb_runs, 1),
            "advance_runs": round(advance_runs, 1),
            "oob_runs": round(oob_runs, 1),
            "gidp_runs": round(gidp_runs, 1),
            "raw_total": round(raw_runs, 1),
        },
    )
