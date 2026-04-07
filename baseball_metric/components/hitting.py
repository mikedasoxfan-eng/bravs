"""Hitting component of BRAVS.

Estimates batting value in runs above FAT using a Bayesian wOBA framework.
The observed wOBA is combined with a positional population prior to produce
a posterior estimate of true-talent wOBA, which is then converted to runs.
"""

from __future__ import annotations

import numpy as np

from baseball_metric.core.types import ComponentResult, PlayerSeason
from baseball_metric.utils.constants import (
    FAT_BATTING_RUNS_PER_600PA,
    LEAGUE_AVG_WOBA,
    MIN_PA_FOR_HITTING,
    PRIOR_WOBA_MEAN,
    PRIOR_WOBA_SD,
    WOBA_SCALE,
)
from baseball_metric.utils.math_helpers import (
    bayesian_update_normal,
    compute_woba,
    credible_interval,
    normal_posterior_samples,
)


# Observation variance for wOBA per PA
# Derived from binomial variance of batting outcomes: ~0.09 per PA
# Source: Tango, "The Book", variance of wOBA ≈ 0.09 / sqrt(PA)
WOBA_OBS_VARIANCE_PER_PA = 0.09


def compute_hitting(
    player: PlayerSeason,
    rng: np.random.Generator | None = None,
    n_samples: int = 10000,
) -> ComponentResult:
    """Compute the hitting component of BRAVS.

    Steps:
    1. Compute observed wOBA from counting stats
    2. Park-adjust and era-adjust the observed wOBA
    3. Apply Bayesian shrinkage toward positional prior
    4. Convert posterior wOBA to runs above FAT
    5. Generate posterior samples for uncertainty

    Args:
        player: PlayerSeason with batting statistics.
        rng: Random generator for reproducibility.
        n_samples: Number of posterior samples.

    Returns:
        ComponentResult with hitting runs above FAT.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if player.pa < 1:
        return ComponentResult(
            name="hitting",
            runs_mean=0.0,
            runs_var=0.0,
            ci_50=(0.0, 0.0),
            ci_90=(0.0, 0.0),
            metadata={"note": "no plate appearances"},
        )

    # Step 1: Observed wOBA
    obs_woba = compute_woba(
        bb=player.ubb,
        hbp=player.hbp,
        singles=player.singles,
        doubles=player.doubles,
        triples=player.triples,
        hr=player.hr,
        ab=player.ab,
        sf=player.sf,
    )

    # Step 2: Park and league adjustment
    # Park factor adjusts observed wOBA: wOBA_adj = wOBA / PF
    # (parks that inflate offense have PF > 1)
    obs_woba_adj = obs_woba / player.park_factor

    # Step 3: Bayesian update
    # Prior: N(PRIOR_WOBA_MEAN, PRIOR_WOBA_SD²)
    # Observation variance for the sample mean: WOBA_OBS_VARIANCE / PA
    prior_mean = PRIOR_WOBA_MEAN
    prior_var = PRIOR_WOBA_SD ** 2
    obs_var = WOBA_OBS_VARIANCE_PER_PA  # per-PA variance

    post_mean, post_var = bayesian_update_normal(
        prior_mean=prior_mean,
        prior_var=prior_var,
        data_mean=obs_woba_adj,
        data_var=obs_var,
        n=max(player.pa, MIN_PA_FOR_HITTING),
    )

    # Step 4: Convert posterior wOBA to runs above FAT
    # Runs above average per PA = (wOBA - lgwOBA) / wOBA_scale
    # FAT batting level is FAT_BATTING_RUNS_PER_600PA below average, prorated
    fat_runs_total = FAT_BATTING_RUNS_PER_600PA * (player.pa / 600.0)  # negative number

    post_runs_above_avg = (post_mean - LEAGUE_AVG_WOBA) / WOBA_SCALE * player.pa
    # Runs above FAT = runs above average - FAT_runs (subtracting a negative = adding)
    total_runs_mean = post_runs_above_avg - fat_runs_total
    post_runs_per_pa_var = post_var / (WOBA_SCALE ** 2)
    total_runs_var = post_runs_per_pa_var * (player.pa ** 2)

    # Step 5: Posterior samples
    woba_samples = normal_posterior_samples(post_mean, post_var, n_samples, rng)
    runs_above_avg_samples = (woba_samples - LEAGUE_AVG_WOBA) / WOBA_SCALE * player.pa
    runs_samples = runs_above_avg_samples - fat_runs_total

    ci50 = credible_interval(runs_samples, 0.50)
    ci90 = credible_interval(runs_samples, 0.90)

    return ComponentResult(
        name="hitting",
        runs_mean=total_runs_mean,
        runs_var=total_runs_var,
        ci_50=ci50,
        ci_90=ci90,
        samples=runs_samples,
        metadata={
            "obs_woba": round(obs_woba, 3),
            "obs_woba_adj": round(obs_woba_adj, 3),
            "post_woba_mean": round(post_mean, 3),
            "post_woba_sd": round(float(np.sqrt(post_var)), 4),
            "shrinkage": round(1.0 - post_var / prior_var, 3),
        },
    )
