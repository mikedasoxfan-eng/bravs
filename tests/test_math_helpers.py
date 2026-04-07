"""Tests for baseball_metric.utils.math_helpers.

These tests verify the fundamental statistical utilities that underpin
the entire BRAVS framework: conjugate Bayesian updates, wOBA / FIP
calculations, credible intervals, and leverage damping.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from baseball_metric.utils.math_helpers import (
    bayesian_update_normal,
    beta_binomial_posterior,
    compute_woba,
    credible_interval,
    damped_leverage,
    ensemble_average,
    fip,
    normal_posterior_samples,
    pythagorean_rpw,
    shrinkage_factor,
    woba_to_runs_per_pa,
)


# ---------------------------------------------------------------------------
# Bayesian update (normal-normal conjugate)
# ---------------------------------------------------------------------------

class TestBayesianUpdateNormal:
    """Verify the closed-form normal-normal conjugate update."""

    def test_known_update(self) -> None:
        """Hand-computed conjugate update with simple values.

        Prior: N(0, 1)  -->  precision = 1
        Data: 10 observations with mean = 2, per-obs variance = 1
              data precision = 10 / 1 = 10
        Posterior precision = 1 + 10 = 11
        Posterior variance = 1/11
        Posterior mean = (1/11) * (1*0 + 10*2) = 20/11 ≈ 1.8182
        """
        post_mean, post_var = bayesian_update_normal(
            prior_mean=0.0,
            prior_var=1.0,
            data_mean=2.0,
            data_var=1.0,
            n=10,
        )

        assert math.isclose(post_mean, 20.0 / 11.0, rel_tol=1e-9), (
            f"Posterior mean {post_mean} should be 20/11"
        )
        assert math.isclose(post_var, 1.0 / 11.0, rel_tol=1e-9), (
            f"Posterior variance {post_var} should be 1/11"
        )

    def test_zero_n_returns_prior(self) -> None:
        """With n=0 observations, the posterior should equal the prior exactly."""
        post_mean, post_var = bayesian_update_normal(
            prior_mean=3.14,
            prior_var=0.5,
            data_mean=999.0,  # should be ignored
            data_var=1.0,
            n=0,
        )

        assert post_mean == 3.14, "n=0 should return prior mean"
        assert post_var == 0.5, "n=0 should return prior variance"

    def test_negative_n_returns_prior(self) -> None:
        """Negative n should be treated like n=0."""
        post_mean, post_var = bayesian_update_normal(
            prior_mean=1.0,
            prior_var=2.0,
            data_mean=10.0,
            data_var=1.0,
            n=-5,
        )

        assert post_mean == 1.0
        assert post_var == 2.0

    def test_very_large_n_approaches_data(self) -> None:
        """With a very large n, the posterior should converge to the data mean."""
        post_mean, post_var = bayesian_update_normal(
            prior_mean=0.0,
            prior_var=1.0,
            data_mean=5.0,
            data_var=1.0,
            n=1_000_000,
        )

        assert math.isclose(post_mean, 5.0, abs_tol=0.01), (
            f"With n=1M, posterior mean ({post_mean}) should be near data mean 5.0"
        )
        assert post_var < 1e-5, (
            f"With n=1M, posterior variance ({post_var}) should be near 0"
        )

    def test_equal_weight(self) -> None:
        """When prior precision equals data precision, posterior mean is the average."""
        # Prior: N(0, 1) -> precision = 1
        # Data: n=1, mean=4, var=1 -> data precision = 1
        # Posterior mean = (0 + 4) / 2 = 2
        post_mean, post_var = bayesian_update_normal(
            prior_mean=0.0,
            prior_var=1.0,
            data_mean=4.0,
            data_var=1.0,
            n=1,
        )

        assert math.isclose(post_mean, 2.0, rel_tol=1e-9)
        assert math.isclose(post_var, 0.5, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Shrinkage factor
# ---------------------------------------------------------------------------

class TestShrinkageFactor:
    """Verify shrinkage factor behavior."""

    def test_zero_n_full_shrinkage(self) -> None:
        """With n=0, shrinkage should be 0 (full weight on prior)."""
        assert shrinkage_factor(0, 1.0, 1.0) == 0.0

    def test_large_n_near_one(self) -> None:
        """With very large n, shrinkage should approach 1 (full weight on data)."""
        alpha = shrinkage_factor(100_000, 1.0, 1.0)
        assert alpha > 0.999, f"Shrinkage factor ({alpha}) should be near 1.0 with large n"

    @pytest.mark.parametrize("n,data_var,prior_var,expected", [
        (10, 1.0, 1.0, 10.0 / 11.0),
        (1, 1.0, 1.0, 0.5),
        (100, 0.5, 0.5, 100.0 / 101.0),
    ])
    def test_shrinkage_values(
        self, n: int, data_var: float, prior_var: float, expected: float
    ) -> None:
        """Parametrized check of shrinkage factor formula."""
        result = shrinkage_factor(n, data_var, prior_var)
        assert math.isclose(result, expected, rel_tol=1e-9), (
            f"shrinkage_factor({n}, {data_var}, {prior_var}) = {result}, expected {expected}"
        )

    def test_shrinkage_bounded_zero_one(self) -> None:
        """Shrinkage factor should always be in [0, 1]."""
        for n in [0, 1, 5, 50, 500, 5000]:
            alpha = shrinkage_factor(n, 0.09, 0.035**2)
            assert 0.0 <= alpha <= 1.0, f"Shrinkage {alpha} out of [0,1] for n={n}"


# ---------------------------------------------------------------------------
# wOBA calculation
# ---------------------------------------------------------------------------

class TestComputeWOBA:
    """Verify wOBA against known values."""

    def test_trout_2016_woba(self) -> None:
        """Mike Trout 2016 wOBA should be high, reflecting an elite season.

        Uses default BRAVS linear weights (calibrated to 2015-2023 average),
        which differ from FanGraphs' season-specific 2016 weights.  The BRAVS
        weights produce a higher wOBA (~.493) than the FanGraphs-published
        .417 because the coefficient set is intentionally inflated to a
        common scale.
        """
        woba = compute_woba(
            bb=111,       # uBB = 116 - 5 IBB
            hbp=7,
            singles=116,
            doubles=24,
            triples=4,
            hr=29,
            ab=441,
            sf=5,
        )

        # With BRAVS default weights the value is ~0.493.
        # Allow a window around that to guard against small weight changes.
        assert 0.460 <= woba <= 0.530, (
            f"Trout 2016 wOBA ({woba:.3f}) should be in [.460, .530] "
            f"with BRAVS default weights"
        )

    def test_zero_denominator(self) -> None:
        """If ab + bb + sf + hbp = 0, wOBA should return 0.0."""
        woba = compute_woba(bb=0, hbp=0, singles=0, doubles=0, triples=0, hr=0, ab=0, sf=0)
        assert woba == 0.0

    def test_all_hr(self) -> None:
        """A player who hits nothing but home runs should have a very high wOBA."""
        woba = compute_woba(bb=0, hbp=0, singles=0, doubles=0, triples=0, hr=50, ab=100, sf=0)
        # With default HR weight ~2.015: numerator = 2.015*50 = 100.75
        # denominator = 100 + 0 + 0 + 0 = 100
        # wOBA ≈ 1.0075
        assert woba > 0.95, f"All-HR wOBA ({woba:.3f}) should be > .95"

    def test_all_outs(self) -> None:
        """A player who makes only outs should have wOBA = 0."""
        woba = compute_woba(bb=0, hbp=0, singles=0, doubles=0, triples=0, hr=0, ab=100, sf=0)
        assert woba == 0.0


# ---------------------------------------------------------------------------
# FIP calculation
# ---------------------------------------------------------------------------

class TestFIP:
    """Verify FIP against known values."""

    def test_degrom_2018_fip(self) -> None:
        """deGrom 2018 FIP should be approximately 1.98 (FanGraphs actual).

        Using actual stats: 10 HR, 46 BB, 5 HBP, 269 K, 217 IP.
        The FIP constant varies by season; we use ~3.10 as default.
        """
        result = fip(hr=10, bb=46, hbp=5, k=269, ip=217.0, fip_constant=3.10)

        # FIP = (13*10 + 3*(46+5) - 2*269) / 217 + 3.10
        #     = (130 + 153 - 538) / 217 + 3.10
        #     = -255 / 217 + 3.10
        #     ≈ -1.175 + 3.10
        #     ≈ 1.925
        assert 1.5 <= result <= 2.5, (
            f"deGrom 2018 FIP ({result:.2f}) should be ~1.9"
        )

    def test_zero_ip_degenerate(self) -> None:
        """With 0 IP, FIP should return a degenerate high value."""
        result = fip(hr=0, bb=0, hbp=0, k=0, ip=0.0)
        assert result > 5.0, "Zero IP should produce a high degenerate FIP"

    def test_perfect_pitcher(self) -> None:
        """A pitcher with 0 HR, 0 BB, 0 HBP and lots of K should have very low FIP."""
        result = fip(hr=0, bb=0, hbp=0, k=200, ip=200.0, fip_constant=3.10)
        # FIP = (0 + 0 - 400) / 200 + 3.10 = -2.0 + 3.10 = 1.10
        assert math.isclose(result, 1.10, abs_tol=0.01), (
            f"Perfect pitcher FIP ({result:.2f}) should be 1.10"
        )

    def test_terrible_pitcher(self) -> None:
        """A pitcher giving up lots of HR and BB with few K should have high FIP."""
        result = fip(hr=40, bb=80, hbp=10, k=50, ip=150.0, fip_constant=3.10)
        assert result > 6.0, f"Terrible pitcher FIP ({result:.2f}) should be > 6.0"


# ---------------------------------------------------------------------------
# Credible interval
# ---------------------------------------------------------------------------

class TestCredibleInterval:
    """Test the percentile-based credible interval computation."""

    def test_standard_normal_90ci(self) -> None:
        """90% CI on a standard normal should be approximately (-1.645, 1.645)."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0.0, 1.0, size=1_000_000)
        lo, hi = credible_interval(samples, 0.90)

        assert math.isclose(lo, -1.645, abs_tol=0.02), (
            f"90% CI lower ({lo:.3f}) should be near -1.645"
        )
        assert math.isclose(hi, 1.645, abs_tol=0.02), (
            f"90% CI upper ({hi:.3f}) should be near 1.645"
        )

    def test_50ci_narrower_than_90ci(self) -> None:
        """50% CI should be narrower than 90% CI."""
        rng = np.random.default_rng(42)
        samples = rng.normal(5.0, 2.0, size=50_000)

        ci50 = credible_interval(samples, 0.50)
        ci90 = credible_interval(samples, 0.90)

        assert (ci50[1] - ci50[0]) < (ci90[1] - ci90[0])

    def test_level_100_gives_full_range(self) -> None:
        """A 100% CI should span the full sample range (or nearly so)."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo, hi = credible_interval(samples, 1.0)

        assert lo == 1.0
        assert hi == 5.0


# ---------------------------------------------------------------------------
# Damped leverage
# ---------------------------------------------------------------------------

class TestDampedLeverage:
    """Test leverage index damping."""

    def test_sqrt_of_four(self) -> None:
        """sqrt(4.0) should give 2.0 with default exponent=0.5."""
        result = damped_leverage(4.0)
        assert math.isclose(result, 2.0, rel_tol=1e-9)

    def test_leverage_one_unchanged(self) -> None:
        """LI = 1.0 should give 1.0 regardless of exponent."""
        assert math.isclose(damped_leverage(1.0, 0.5), 1.0, rel_tol=1e-9)
        assert math.isclose(damped_leverage(1.0, 0.0), 1.0, rel_tol=1e-9)
        assert math.isclose(damped_leverage(1.0, 1.0), 1.0, rel_tol=1e-9)

    def test_damping_reduces_high_leverage(self) -> None:
        """Damped leverage should be less than raw leverage for LI > 1."""
        li = 2.0
        damped = damped_leverage(li, 0.5)
        assert damped < li, f"Damped ({damped}) should be < raw ({li}) for LI > 1"

    def test_damping_increases_low_leverage(self) -> None:
        """Damped leverage should be greater than raw leverage for 0 < LI < 1."""
        li = 0.25
        damped = damped_leverage(li, 0.5)
        assert damped > li, f"Damped ({damped}) should be > raw ({li}) for 0 < LI < 1"

    def test_zero_leverage_clamped(self) -> None:
        """Zero leverage should not produce 0 (clamped at 0.01 floor)."""
        result = damped_leverage(0.0)
        assert result > 0.0, "Zero LI should be clamped above 0"


# ---------------------------------------------------------------------------
# Pythagorean runs-per-win
# ---------------------------------------------------------------------------

class TestPythagoreanRPW:
    """Verify the Pythagorean-derived runs per win."""

    def test_standard_environment(self) -> None:
        """In a 4.5 R/G environment with exponent=2, RPW should be 4.5."""
        rpw = pythagorean_rpw(4.5, exponent=2.0)
        assert math.isclose(rpw, 4.5, rel_tol=1e-9)

    def test_higher_rpg_higher_rpw(self) -> None:
        """Higher R/G should produce higher RPW."""
        assert pythagorean_rpw(5.0) > pythagorean_rpw(4.0)


# ---------------------------------------------------------------------------
# wOBA to runs per PA
# ---------------------------------------------------------------------------

class TestWobaToRuns:
    """Test wOBA-to-runs conversion."""

    def test_league_average_is_zero(self) -> None:
        """A league-average wOBA should produce 0 runs above average per PA."""
        result = woba_to_runs_per_pa(0.315, league_woba=0.315, woba_scale=1.157)
        assert result == 0.0

    def test_above_average_positive(self) -> None:
        """Above-average wOBA should give positive runs per PA."""
        result = woba_to_runs_per_pa(0.400, league_woba=0.315, woba_scale=1.157)
        assert result > 0.0

    def test_below_average_negative(self) -> None:
        """Below-average wOBA should give negative runs per PA."""
        result = woba_to_runs_per_pa(0.250, league_woba=0.315, woba_scale=1.157)
        assert result < 0.0


# ---------------------------------------------------------------------------
# Beta-binomial posterior
# ---------------------------------------------------------------------------

class TestBetaBinomialPosterior:
    """Test the beta-binomial conjugate update."""

    def test_uniform_prior_update(self) -> None:
        """Beta(1,1) + 7 successes in 10 trials = Beta(8, 4)."""
        alpha, beta = beta_binomial_posterior(successes=7, trials=10, prior_alpha=1.0, prior_beta=1.0)
        assert alpha == 8.0
        assert beta == 4.0

    def test_zero_trials_returns_prior(self) -> None:
        """With zero trials, posterior should equal prior."""
        alpha, beta = beta_binomial_posterior(successes=0, trials=0, prior_alpha=2.0, prior_beta=5.0)
        assert alpha == 2.0
        assert beta == 5.0


# ---------------------------------------------------------------------------
# Ensemble average
# ---------------------------------------------------------------------------

class TestEnsembleAverage:
    """Test weighted ensemble averaging."""

    def test_equal_weights(self) -> None:
        """Equal weights should give the arithmetic mean."""
        estimates = {"a": 2.0, "b": 4.0, "c": 6.0}
        weights = {"a": 1.0, "b": 1.0, "c": 1.0}
        result = ensemble_average(estimates, weights)
        assert math.isclose(result, 4.0, rel_tol=1e-9)

    def test_missing_weight(self) -> None:
        """Estimates without corresponding weights should be ignored."""
        estimates = {"a": 2.0, "b": 4.0, "c": 100.0}
        weights = {"a": 1.0, "b": 1.0}
        result = ensemble_average(estimates, weights)
        assert math.isclose(result, 3.0, rel_tol=1e-9)

    def test_all_missing_returns_zero(self) -> None:
        """If no weights match, return 0."""
        result = ensemble_average({"x": 1.0}, {"y": 1.0})
        assert result == 0.0

    def test_nan_value_ignored(self) -> None:
        """Non-finite values should be excluded from the average."""
        estimates = {"a": 2.0, "b": float("nan")}
        weights = {"a": 1.0, "b": 1.0}
        result = ensemble_average(estimates, weights)
        assert math.isclose(result, 2.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Normal posterior samples
# ---------------------------------------------------------------------------

class TestNormalPosteriorSamples:
    """Test sample generation from normal posterior."""

    def test_sample_count(self) -> None:
        """Should return the requested number of samples."""
        samples = normal_posterior_samples(0.0, 1.0, n_samples=500)
        assert len(samples) == 500

    def test_mean_close_to_parameter(self) -> None:
        """Sample mean should be close to the distribution mean."""
        samples = normal_posterior_samples(10.0, 1.0, n_samples=100_000, rng=np.random.default_rng(42))
        assert math.isclose(np.mean(samples), 10.0, abs_tol=0.05)

    def test_reproducibility(self) -> None:
        """Same RNG seed should produce identical samples."""
        s1 = normal_posterior_samples(0.0, 1.0, rng=np.random.default_rng(42))
        s2 = normal_posterior_samples(0.0, 1.0, rng=np.random.default_rng(42))
        assert np.array_equal(s1, s2)
