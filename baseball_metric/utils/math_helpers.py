"""Statistical utility functions for BRAVS computation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats  # type: ignore[import-untyped]


def bayesian_update_normal(
    prior_mean: float,
    prior_var: float,
    data_mean: float,
    data_var: float,
    n: int,
) -> tuple[float, float]:
    """Bayesian update for normal-normal conjugate model.

    Given a normal prior N(prior_mean, prior_var) and n observations
    with sample mean data_mean and known per-observation variance data_var,
    returns the posterior mean and variance.

    Args:
        prior_mean: Prior distribution mean.
        prior_var: Prior distribution variance.
        data_mean: Observed sample mean.
        data_var: Per-observation variance (sigma² of likelihood).
        n: Number of observations.

    Returns:
        Tuple of (posterior_mean, posterior_var).
    """
    if n <= 0:
        return prior_mean, prior_var

    precision_prior = 1.0 / prior_var
    precision_data = n / data_var

    posterior_precision = precision_prior + precision_data
    posterior_var = 1.0 / posterior_precision
    posterior_mean = posterior_var * (precision_prior * prior_mean + precision_data * data_mean)

    return posterior_mean, posterior_var


def shrinkage_factor(n: int, data_var: float, prior_var: float) -> float:
    """Compute the Bayesian shrinkage factor toward the prior.

    Returns a value in [0, 1] where 0 means full shrinkage to prior
    and 1 means full weight on observed data.

    Args:
        n: Number of observations.
        data_var: Per-observation variance.
        prior_var: Prior variance.

    Returns:
        Shrinkage factor alpha in [0, 1].
    """
    if n <= 0:
        return 0.0
    return (n * prior_var) / (n * prior_var + data_var)


def credible_interval(
    samples: NDArray[np.floating[object]],
    level: float = 0.90,
) -> tuple[float, float]:
    """Compute highest-density credible interval from posterior samples.

    Uses percentile method (equal-tailed interval) for simplicity.
    For unimodal, roughly symmetric posteriors this is close to HDI.

    Args:
        samples: Array of posterior samples.
        level: Credible level (e.g., 0.90 for 90% CI).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    alpha = 1.0 - level
    lower = float(np.percentile(samples, 100 * alpha / 2))
    upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lower, upper


def pythagorean_rpw(runs_per_game: float, exponent: float = 2.0) -> float:
    """Dynamic runs-per-win from Pythagorean expectation derivative.

    Derived from W = R^e / (R^e + RA^e). At equilibrium (R = RA),
    the marginal win cost of a run is: RPW = 2 * R / (e * W_total).
    Simplified: RPW ≈ (2 * RPG) / e for balanced teams.

    More precisely: RPW = RPG / (e * 0.5) = 2 * RPG / e

    For the standard exponent e=2: RPW ≈ RPG.
    But we use the full formula for flexibility with non-standard exponents.

    Args:
        runs_per_game: Average runs per game in the league (both teams combined / 2,
                       i.e., runs per team per game).
        exponent: Pythagorean exponent (default 2.0, Pythagenpat uses variable).

    Returns:
        Runs per win value.
    """
    return 2.0 * runs_per_game / exponent


def woba_to_runs_per_pa(woba: float, league_woba: float, woba_scale: float) -> float:
    """Convert wOBA to runs above average per plate appearance.

    Formula: (wOBA - lgwOBA) / wOBA_scale

    Args:
        woba: Player's weighted on-base average.
        league_woba: League average wOBA.
        woba_scale: wOBA scale factor (converts to run scale).

    Returns:
        Runs above average per PA.
    """
    return (woba - league_woba) / woba_scale


def damped_leverage(leverage_index: float, exponent: float = 0.5) -> float:
    """Apply damping to leverage index.

    Uses LI^exponent to moderate the effect of leverage on value.
    With exponent=0.5 (sqrt), this is the geometric mean of
    "ignore leverage" (exponent=0) and "full leverage" (exponent=1).

    Args:
        leverage_index: Raw average leverage index (gmLI).
        exponent: Damping exponent (0.5 = sqrt damping).

    Returns:
        Damped leverage multiplier.
    """
    return float(np.power(max(leverage_index, 0.01), exponent))


def normal_posterior_samples(
    mean: float,
    var: float,
    n_samples: int = 10000,
    rng: np.random.Generator | None = None,
) -> NDArray[np.floating[object]]:
    """Draw samples from a normal posterior distribution.

    Args:
        mean: Posterior mean.
        var: Posterior variance.
        n_samples: Number of samples to draw.
        rng: NumPy random generator for reproducibility.

    Returns:
        Array of posterior samples.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.normal(mean, np.sqrt(var), size=n_samples)


def compute_woba(
    bb: int,
    hbp: int,
    singles: int,
    doubles: int,
    triples: int,
    hr: int,
    ab: int,
    sf: int,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted on-base average.

    wOBA = (w_BB*BB + w_HBP*HBP + w_1B*1B + w_2B*2B + w_3B*3B + w_HR*HR) /
           (AB + BB - IBB + SF + HBP)

    We treat all BB as unintentional here (IBB data handled upstream).

    Args:
        bb: Walks (unintentional).
        hbp: Hit by pitch.
        singles: Singles.
        doubles: Doubles.
        triples: Triples.
        hr: Home runs.
        ab: At bats.
        sf: Sacrifice flies.
        weights: Optional custom wOBA weights dict.

    Returns:
        wOBA value.
    """
    from baseball_metric.utils.constants import (
        WOBA_WEIGHT_1B,
        WOBA_WEIGHT_2B,
        WOBA_WEIGHT_3B,
        WOBA_WEIGHT_BB,
        WOBA_WEIGHT_HBP,
        WOBA_WEIGHT_HR,
    )

    w = weights or {}
    w_bb = w.get("BB", WOBA_WEIGHT_BB)
    w_hbp = w.get("HBP", WOBA_WEIGHT_HBP)
    w_1b = w.get("1B", WOBA_WEIGHT_1B)
    w_2b = w.get("2B", WOBA_WEIGHT_2B)
    w_3b = w.get("3B", WOBA_WEIGHT_3B)
    w_hr = w.get("HR", WOBA_WEIGHT_HR)

    numerator = w_bb * bb + w_hbp * hbp + w_1b * singles + w_2b * doubles + w_3b * triples + w_hr * hr
    denominator = ab + bb + sf + hbp

    if denominator == 0:
        return 0.0

    return numerator / denominator


def fip(
    hr: int,
    bb: int,
    hbp: int,
    k: int,
    ip: float,
    fip_constant: float = 3.10,
) -> float:
    """Compute Fielding Independent Pitching.

    FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + C

    Args:
        hr: Home runs allowed.
        bb: Walks issued.
        hbp: Hit batters.
        k: Strikeouts.
        ip: Innings pitched.
        fip_constant: League-calibrated constant (typically ~3.10).

    Returns:
        FIP value.
    """
    if ip <= 0:
        return fip_constant + 5.0  # degenerate case
    return (FIP_HR_COEFF * hr + FIP_BB_COEFF * (bb + hbp) - FIP_K_COEFF * k) / ip + fip_constant


# Import constants used in fip function
from baseball_metric.utils.constants import FIP_BB_COEFF, FIP_HR_COEFF, FIP_K_COEFF


def beta_binomial_posterior(
    successes: int,
    trials: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> tuple[float, float]:
    """Beta-binomial conjugate posterior for rate stats.

    Given a Beta(alpha, beta) prior and binomial data,
    the posterior is Beta(alpha + successes, beta + failures).

    Args:
        successes: Number of successes observed.
        trials: Number of trials.
        prior_alpha: Prior alpha parameter.
        prior_beta: Prior beta parameter.

    Returns:
        Tuple of (posterior_alpha, posterior_beta).
    """
    post_alpha = prior_alpha + successes
    post_beta = prior_beta + (trials - successes)
    return post_alpha, post_beta


def ensemble_average(
    estimates: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Weighted ensemble average of multiple metric estimates.

    Args:
        estimates: Dict mapping metric name to estimate value.
        weights: Dict mapping metric name to weight.

    Returns:
        Weighted average of available estimates.
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for name, value in estimates.items():
        if name in weights and np.isfinite(value):
            w = weights[name]
            weighted_sum += w * value
            total_weight += w

    if total_weight == 0.0:
        return 0.0

    return weighted_sum / total_weight
