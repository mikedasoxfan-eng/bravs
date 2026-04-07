"""Optional MCMC posterior estimation for BRAVS.

Provides a Metropolis-Hastings sampler for non-conjugate posteriors,
giving richer (potentially skewed, multimodal) distributions than
the default conjugate normal approximation.

Uses only numpy/scipy — no PyMC or Stan dependency required.

Usage:
    python -m baseball_metric --notable-seasons --mcmc
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats  # type: ignore[import-untyped]

from baseball_metric.core.types import ComponentResult, PlayerSeason
from baseball_metric.utils.constants import (
    LEAGUE_AVG_WOBA,
    PRIOR_WOBA_MEAN,
    PRIOR_WOBA_SD,
    WOBA_SCALE,
)


def _log_normal_pdf(x: float, mean: float, sd: float) -> float:
    """Log PDF of normal distribution."""
    return -0.5 * ((x - mean) / sd) ** 2 - np.log(sd)


def _log_beta_binomial(rate: float, successes: int, trials: int,
                       alpha: float, beta: float) -> float:
    """Log posterior for a Beta-Binomial model.

    Captures skewness for batting rates at small sample sizes.
    """
    if rate <= 0.0 or rate >= 1.0:
        return -np.inf
    log_prior = stats.beta.logpdf(rate, alpha, beta)
    log_lik = stats.binom.logpmf(successes, trials, rate)
    return log_prior + log_lik


def mcmc_hitting_posterior(
    player: PlayerSeason,
    n_samples: int = 20000,
    burn_in: int = 5000,
    seed: int = 42,
) -> ComponentResult:
    """MCMC posterior for the hitting component using Metropolis-Hastings.

    Instead of the conjugate normal approximation to wOBA, this models
    the on-base rate as a Beta-Binomial, preserving skewness at small
    sample sizes. For large samples (>200 PA), results converge to the
    conjugate solution.

    Args:
        player: PlayerSeason with batting statistics.
        n_samples: Total MCMC samples (including burn-in).
        burn_in: Samples to discard as warm-up.
        seed: Random seed.

    Returns:
        ComponentResult with hitting runs from MCMC posterior.
    """
    rng = np.random.default_rng(seed)

    if player.pa < 1:
        return ComponentResult(
            name="hitting",
            runs_mean=0.0,
            runs_var=0.0,
            metadata={"method": "mcmc", "note": "no PA"},
        )

    # Model on-base rate as Beta-Binomial
    # Prior: Beta(alpha, beta) calibrated to match normal prior on wOBA
    # Mean = alpha/(alpha+beta) ≈ PRIOR_WOBA_MEAN, concentration ≈ 1/PRIOR_WOBA_SD²
    prior_mean = PRIOR_WOBA_MEAN
    concentration = 1.0 / (PRIOR_WOBA_SD ** 2)  # ~816
    alpha_prior = prior_mean * concentration
    beta_prior = (1.0 - prior_mean) * concentration

    # Observed: approximate "successes" as wOBA-weighted events / max_weight
    from baseball_metric.utils.math_helpers import compute_woba
    obs_woba = compute_woba(
        bb=player.ubb, hbp=player.hbp, singles=player.singles,
        doubles=player.doubles, triples=player.triples, hr=player.hr,
        ab=player.ab, sf=player.sf,
    )
    obs_woba_adj = obs_woba / player.park_factor

    # Map wOBA to a pseudo-rate for Beta-Binomial
    # wOBA ranges ~0.2-0.5, map to (0,1) via wOBA / 0.6
    obs_rate = min(max(obs_woba_adj / 0.6, 0.01), 0.99)
    pseudo_successes = int(obs_rate * player.pa)
    pseudo_trials = player.pa

    # Metropolis-Hastings
    current = obs_rate
    samples_raw: list[float] = []
    proposal_sd = 0.02  # tuned for ~30-40% acceptance

    for i in range(n_samples):
        proposal = current + rng.normal(0, proposal_sd)
        if proposal <= 0.0 or proposal >= 1.0:
            samples_raw.append(current)
            continue

        log_current = _log_beta_binomial(current, pseudo_successes, pseudo_trials,
                                          alpha_prior, beta_prior)
        log_proposal = _log_beta_binomial(proposal, pseudo_successes, pseudo_trials,
                                           alpha_prior, beta_prior)

        log_accept = log_proposal - log_current
        if np.log(rng.random()) < log_accept:
            current = proposal

        samples_raw.append(current)

    # Discard burn-in and convert back to wOBA scale
    rate_samples = np.array(samples_raw[burn_in:])
    woba_samples = rate_samples * 0.6  # map back to wOBA scale

    # Convert to runs above FAT
    from baseball_metric.utils.constants import FAT_BATTING_RUNS_PER_600PA
    fat_runs_total = FAT_BATTING_RUNS_PER_600PA * (player.pa / 600.0)
    runs_above_avg = (woba_samples - LEAGUE_AVG_WOBA) / WOBA_SCALE * player.pa
    runs_samples = runs_above_avg - fat_runs_total

    from baseball_metric.utils.math_helpers import credible_interval
    ci50 = credible_interval(runs_samples, 0.50)
    ci90 = credible_interval(runs_samples, 0.90)

    acceptance = sum(
        1 for i in range(1, len(samples_raw))
        if samples_raw[i] != samples_raw[i - 1]
    ) / max(len(samples_raw) - 1, 1)

    return ComponentResult(
        name="hitting",
        runs_mean=float(np.mean(runs_samples)),
        runs_var=float(np.var(runs_samples)),
        ci_50=ci50,
        ci_90=ci90,
        samples=runs_samples,
        metadata={
            "method": "mcmc_metropolis_hastings",
            "n_samples": len(runs_samples),
            "burn_in": burn_in,
            "acceptance_rate": round(acceptance, 3),
            "post_woba_mean": round(float(np.mean(woba_samples)), 3),
            "post_woba_sd": round(float(np.std(woba_samples)), 4),
            "skewness": round(float(stats.skew(runs_samples)), 3),
        },
    )


def mcmc_compute_bravs(
    player: PlayerSeason,
    n_samples: int = 20000,
    burn_in: int = 5000,
    seed: int = 42,
) -> None:
    """Compute BRAVS with MCMC posteriors for hitting component.

    Replaces the conjugate normal hitting posterior with a Beta-Binomial
    MCMC posterior. All other components use their standard computation.
    This is slower (~100ms vs ~5ms) but captures skewness in small samples.

    This function patches the hitting component in-place before calling
    the standard compute_bravs pipeline.
    """
    from baseball_metric.core.model import compute_bravs

    # Compute standard BRAVS first
    result = compute_bravs(player, n_samples=n_samples - burn_in, seed=seed)

    # Replace hitting component with MCMC version
    if player.pa >= 1:
        mcmc_hitting = mcmc_hitting_posterior(player, n_samples, burn_in, seed)
        result.components["hitting"] = mcmc_hitting

        # Recompute total from updated components
        from baseball_metric.core.posterior import combine_component_posteriors
        rng = np.random.default_rng(seed)
        total_mean, total_var, total_samples = combine_component_posteriors(
            result.components,
            result.leverage_multiplier,
            result.rpw,
            rng,
            n_samples - burn_in,
        )
        result.total_runs_mean = total_mean
        result.total_runs_var = total_var
        result.total_samples = total_samples

    return result  # type: ignore[return-value]
