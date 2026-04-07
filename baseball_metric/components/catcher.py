"""Catcher-specific component of BRAVS.

Estimates catcher value beyond standard fielding, including:
- Pitch framing (strikes gained × run value)
- Pitch blocking (wild pitches/passed balls prevented)
- Throwing (caught stealing and pickoff contributions)
- Game-calling (pitcher WOWY differential, heavily regressed)
"""

from __future__ import annotations

import numpy as np

from baseball_metric.core.types import ComponentResult, PlayerSeason
from baseball_metric.utils.constants import (
    MIN_PITCHES_FOR_FRAMING,
    PRIOR_FRAMING_MEAN,
    PRIOR_FRAMING_SD,
    RUN_VALUE_STRIKE_GAINED,
)
from baseball_metric.utils.math_helpers import (
    bayesian_update_normal,
    credible_interval,
    normal_posterior_samples,
)

# Observation variance for framing runs per season
# Framing is relatively stable (YoY r ≈ 0.6-0.7) but still noisy
FRAMING_OBS_VARIANCE = 25.0

# Default values for blocking and throwing if not available
# Source: average contribution ranges from Statcast catcher data
DEFAULT_BLOCKING_RUNS = 0.0
DEFAULT_THROWING_RUNS = 0.0

# Game-calling prior: wide because WOWY is noisy, but we can do better
# than pure prior by incorporating multi-year rolling data and
# pitcher-specific effects
GAME_CALLING_PRIOR_SD = 4.0
GAME_CALLING_OBS_VARIANCE_SINGLE_YEAR = 100.0  # very high for 1 year of WOWY
GAME_CALLING_OBS_VARIANCE_MULTI_YEAR = 40.0  # better with 3+ years of data


def compute_catcher(
    player: PlayerSeason,
    rng: np.random.Generator | None = None,
    n_samples: int = 10000,
) -> ComponentResult:
    """Compute the catcher-specific component of BRAVS.

    Only applies to catchers. Non-catchers return zero.

    Sub-components:
    1. Framing: runs from extra strikes gained (Bayesian-shrunk)
    2. Blocking: runs from preventing wild pitches/passed balls
    3. Throwing: runs from CS and pickoffs above average
    4. Game-calling: WOWY differential (heavily regressed)

    Args:
        player: PlayerSeason with catcher statistics.
        rng: Random generator for reproducibility.
        n_samples: Number of posterior samples.

    Returns:
        ComponentResult with catcher-specific runs above average.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if not player.is_catcher:
        return ComponentResult(
            name="catcher",
            runs_mean=0.0,
            runs_var=0.0,
            ci_50=(0.0, 0.0),
            ci_90=(0.0, 0.0),
            metadata={"note": "not a catcher"},
        )

    # --- Sub-component 1: Framing ---
    if player.framing_runs is not None:
        framing_obs = player.framing_runs
    else:
        # If no framing data, use prior only
        framing_obs = PRIOR_FRAMING_MEAN

    effective_n_framing = max(player.catcher_pitches // MIN_PITCHES_FOR_FRAMING, 1)

    framing_post_mean, framing_post_var = bayesian_update_normal(
        prior_mean=PRIOR_FRAMING_MEAN,
        prior_var=PRIOR_FRAMING_SD ** 2,
        data_mean=framing_obs,
        data_var=FRAMING_OBS_VARIANCE,
        n=effective_n_framing,
    )

    # --- Sub-component 2: Blocking ---
    blocking_runs = player.blocking_runs if player.blocking_runs is not None else DEFAULT_BLOCKING_RUNS

    # --- Sub-component 3: Throwing ---
    throwing_runs = player.throwing_runs if player.throwing_runs is not None else DEFAULT_THROWING_RUNS

    # --- Sub-component 4: Game-calling (multi-year WOWY) ---
    # If game_calling_runs is provided (from external WOWY analysis),
    # use it with variance scaled by years of data. More years = tighter.
    # If not provided, fall back to prior (0 with wide uncertainty).
    if player.game_calling_runs is not None:
        gc_obs = player.game_calling_runs
        # Multi-year data is much more reliable than single-year
        if player.game_calling_years >= 3:
            gc_obs_var = GAME_CALLING_OBS_VARIANCE_MULTI_YEAR
        else:
            gc_obs_var = GAME_CALLING_OBS_VARIANCE_SINGLE_YEAR

        game_calling_mean, game_calling_var = bayesian_update_normal(
            prior_mean=0.0,
            prior_var=GAME_CALLING_PRIOR_SD ** 2,
            data_mean=gc_obs,
            data_var=gc_obs_var,
            n=max(player.game_calling_years, 1),
        )
    else:
        game_calling_mean = 0.0
        game_calling_var = GAME_CALLING_PRIOR_SD ** 2

    # --- Aggregate ---
    total_mean = framing_post_mean + blocking_runs + throwing_runs + game_calling_mean
    total_var = framing_post_var + game_calling_var  # blocking/throwing treated as known

    # Posterior samples (dominated by framing uncertainty)
    framing_samples = normal_posterior_samples(framing_post_mean, framing_post_var, n_samples, rng)
    gc_samples = normal_posterior_samples(game_calling_mean, game_calling_var, n_samples, rng)
    total_samples = framing_samples + blocking_runs + throwing_runs + gc_samples

    ci50 = credible_interval(total_samples, 0.50)
    ci90 = credible_interval(total_samples, 0.90)

    return ComponentResult(
        name="catcher",
        runs_mean=total_mean,
        runs_var=total_var,
        ci_50=ci50,
        ci_90=ci90,
        samples=total_samples,
        metadata={
            "framing_runs": round(framing_post_mean, 1),
            "blocking_runs": round(blocking_runs, 1),
            "throwing_runs": round(throwing_runs, 1),
            "game_calling_runs": round(game_calling_mean, 1),
            "framing_shrinkage_pct": round(
                (1.0 - framing_post_var / (PRIOR_FRAMING_SD ** 2)) * 100, 1
            ),
        },
    )
