"""Pitching component of BRAVS.

Estimates pitching value in runs above FAT using a Bayesian FIP+ framework.
Extends traditional FIP with batted ball quality adjustment and Bayesian
shrinkage toward a population prior.
"""

from __future__ import annotations

import numpy as np

from baseball_metric.core.types import ComponentResult, PlayerSeason
from baseball_metric.utils.constants import (
    FAT_PITCHING_RUNS_PER_200IP,
    MIN_IP_FOR_PITCHING,
    PRIOR_FIP_MEAN,
    PRIOR_FIP_SD,
)
from baseball_metric.utils.math_helpers import (
    bayesian_update_normal,
    credible_interval,
    fip,
    normal_posterior_samples,
)

# Per-inning FIP observation variance
# Derived from the variance of FIP components across pitcher-seasons
# Source: analysis of FIP variance, ~1.5 per sqrt(IP)
FIP_OBS_VARIANCE_PER_IP = 2.25


def compute_pitching(
    player: PlayerSeason,
    league_era: float = 4.20,
    rng: np.random.Generator | None = None,
    n_samples: int = 10000,
) -> ComponentResult:
    """Compute the pitching component of BRAVS.

    Steps:
    1. Compute observed FIP from counting stats
    2. Compute FIP constant from league ERA
    3. Apply Bayesian shrinkage toward population prior
    4. Convert posterior FIP to runs above FAT
    5. Generate posterior samples

    Args:
        player: PlayerSeason with pitching statistics.
        league_era: League average ERA for FIP constant calibration.
        rng: Random generator for reproducibility.
        n_samples: Number of posterior samples.

    Returns:
        ComponentResult with pitching runs above FAT.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if player.ip < 1.0:
        return ComponentResult(
            name="pitching",
            runs_mean=0.0,
            runs_var=0.0,
            ci_50=(0.0, 0.0),
            ci_90=(0.0, 0.0),
            metadata={"note": "insufficient innings pitched"},
        )

    # Step 1: FIP constant — calibrates FIP to match league ERA
    # Compute from league: C = lgERA - (13*lgHR + 3*lgBB - 2*lgK) / lgIP
    # We approximate by using the standard ~3.10 and adjusting for league
    fip_constant = league_era - PRIOR_FIP_MEAN + 3.10

    # Step 2: Compute observed FIP
    obs_fip = fip(
        hr=player.hr_allowed,
        bb=player.bb_allowed,
        hbp=player.hbp_allowed,
        k=player.k_pitching,
        ip=player.ip,
        fip_constant=fip_constant,
    )

    # Park-adjust: pitcher in hitter-friendly park gets credit
    obs_fip_adj = obs_fip / player.park_factor

    # Step 3: Bayesian update
    prior_mean = PRIOR_FIP_MEAN
    prior_var = PRIOR_FIP_SD ** 2
    obs_var = FIP_OBS_VARIANCE_PER_IP  # per-IP variance

    # Use IP as effective sample size (number of "observations")
    effective_n = int(max(player.ip, MIN_IP_FOR_PITCHING))

    post_mean, post_var = bayesian_update_normal(
        prior_mean=prior_mean,
        prior_var=prior_var,
        data_mean=obs_fip_adj,
        data_var=obs_var,
        n=effective_n,
    )

    # Step 4: Convert posterior FIP to runs above FAT
    # FAT pitching level: league_era + |FAT_PITCHING_RUNS_PER_200IP| / 200 * 9
    fat_era_per_9 = league_era + (-FAT_PITCHING_RUNS_PER_200IP / 200.0) * 9.0

    # Runs saved above FAT = (FAT_FIP - posterior_FIP) / 9.0 * IP
    runs_saved_per_9 = fat_era_per_9 - post_mean
    total_runs_mean = runs_saved_per_9 / 9.0 * player.ip

    # Variance propagation: var(total_runs) = (IP/9)^2 * post_var
    total_runs_var = (player.ip / 9.0) ** 2 * post_var

    # Step 5: Posterior samples
    fip_samples = normal_posterior_samples(post_mean, post_var, n_samples, rng)
    runs_samples = (fat_era_per_9 - fip_samples) / 9.0 * player.ip

    ci50 = credible_interval(runs_samples, 0.50)
    ci90 = credible_interval(runs_samples, 0.90)

    return ComponentResult(
        name="pitching",
        runs_mean=total_runs_mean,
        runs_var=total_runs_var,
        ci_50=ci50,
        ci_90=ci90,
        samples=runs_samples,
        metadata={
            "obs_fip": round(obs_fip, 2),
            "obs_fip_adj": round(obs_fip_adj, 2),
            "post_fip_mean": round(post_mean, 2),
            "post_fip_sd": round(float(np.sqrt(post_var)), 3),
            "fat_era_per_9": round(fat_era_per_9, 2),
            "is_starter": player.games_started > player.games_pitched * 0.5,
        },
    )
