"""Tests for the hitting component of BRAVS.

Validates that compute_hitting produces sensible run values across a
range of player archetypes and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from baseball_metric.components.hitting import compute_hitting
from baseball_metric.core.types import ComponentResult, PlayerSeason


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestHittingValues:
    """Core hitting-value sanity checks against known stat lines."""

    def test_trout_hitting_positive(self, trout_2016: PlayerSeason) -> None:
        """Mike Trout 2016 should produce a large positive hitting value.

        Trout's .441 wOBA / 10.6 fWAR season should yield well above +30
        batting runs above FAT.
        """
        result = compute_hitting(trout_2016)

        assert isinstance(result, ComponentResult)
        assert result.runs_mean > 30.0, (
            f"Trout 2016 hitting runs ({result.runs_mean:.1f}) should be > 30; "
            f"wOBA metadata: {result.metadata}"
        )

    def test_replacement_hitting_near_zero(self, replacement_player: PlayerSeason) -> None:
        """A FAT-calibrated hitter should produce hitting runs near zero.

        The replacement-player fixture is designed to approximate freely
        available talent, so hitting runs should be within +/-10 of zero.
        """
        result = compute_hitting(replacement_player)

        assert -10.0 <= result.runs_mean <= 10.0, (
            f"Replacement hitter runs ({result.runs_mean:.1f}) should be near 0"
        )

    def test_negative_hitter(self, negative_value_player: PlayerSeason) -> None:
        """A .150 BA hitter over 350 PA should produce negative hitting runs."""
        result = compute_hitting(negative_value_player)

        assert result.runs_mean < 0.0, (
            f"Negative-value hitter runs ({result.runs_mean:.1f}) should be < 0"
        )

    def test_ohtani_hitting_positive(self, ohtani_2023: PlayerSeason) -> None:
        """Ohtani 2023 had elite offensive production even as a two-way player."""
        result = compute_hitting(ohtani_2023)

        assert result.runs_mean > 20.0, (
            f"Ohtani 2023 hitting runs ({result.runs_mean:.1f}) should be > 20"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestHittingEdgeCases:
    """Boundary conditions and degenerate inputs."""

    def test_zero_pa_returns_zero(self) -> None:
        """A player with zero plate appearances should get exactly 0.0 hitting value."""
        player = PlayerSeason(
            player_id="zeropa",
            player_name="No PA Player",
            season=2023,
            team="XXX",
            position="RF",
            pa=0,
            ab=0,
        )
        result = compute_hitting(player)

        assert result.runs_mean == 0.0, "Zero PA should yield exactly 0.0 runs"
        assert result.runs_var == 0.0, "Zero PA should yield exactly 0.0 variance"
        assert result.ci_90 == (0.0, 0.0), "Zero PA CIs should be (0, 0)"

    def test_one_pa(self) -> None:
        """A single plate appearance should produce a result dominated by the prior."""
        player = PlayerSeason(
            player_id="onepa",
            player_name="One PA",
            season=2023,
            team="XXX",
            position="CF",
            pa=1,
            ab=1,
            hits=1,
            singles=1,
        )
        result = compute_hitting(player)

        # With 1 PA the posterior should be heavily shrunk toward the prior,
        # so the total runs (scaled by 1 PA) should be tiny.
        assert abs(result.runs_mean) < 5.0, (
            f"1 PA hitting runs ({result.runs_mean:.2f}) should be near-zero"
        )


# ---------------------------------------------------------------------------
# Bayesian shrinkage behavior
# ---------------------------------------------------------------------------

class TestHittingShrinkage:
    """Verify that Bayesian shrinkage behaves as expected."""

    def test_shrinkage_with_small_sample(self) -> None:
        """With only 50 PA, the posterior wOBA should be pulled toward the prior.

        An impossibly high observed wOBA (~.600) in 50 PA should be dragged
        well below the raw value by the prior.
        """
        # Construct a player who went 30-for-40 with 10 HR in 50 PA (absurd).
        player = PlayerSeason(
            player_id="smalls",
            player_name="Small Sample Hero",
            season=2023,
            team="XXX",
            position="2B",
            pa=50,
            ab=40,
            hits=30,
            singles=10,
            doubles=5,
            triples=5,
            hr=10,
            bb=8,
            hbp=2,
            k=5,
            sf=0,
        )
        result = compute_hitting(player)
        shrinkage = result.metadata.get("shrinkage", 0.0)

        # Shrinkage toward prior should be substantial (> 0.3) with only 50 PA.
        assert shrinkage > 0.2, (
            f"Shrinkage ({shrinkage}) should be > 0.2 for 50 PA"
        )

        # The posterior wOBA should be well below the absurd observed value.
        obs_woba = result.metadata.get("obs_woba", 0.0)
        post_woba = result.metadata.get("post_woba_mean", 0.0)
        assert post_woba < obs_woba, (
            f"Posterior wOBA ({post_woba:.3f}) should be shrunk below "
            f"observed ({obs_woba:.3f})"
        )

    def test_large_sample_less_shrinkage(self, trout_2016: PlayerSeason) -> None:
        """With 681 PA the posterior should heavily weight observed data."""
        result = compute_hitting(trout_2016)
        shrinkage = result.metadata.get("shrinkage", 0.0)

        # With ~680 PA the posterior is almost entirely data-driven.
        assert shrinkage > 0.85, (
            f"Shrinkage ({shrinkage}) should exceed 0.85 with 681 PA"
        )


# ---------------------------------------------------------------------------
# Park factor adjustment
# ---------------------------------------------------------------------------

class TestHittingParkFactor:
    """Park factor should deflate stats from hitter-friendly parks."""

    def test_park_factor_adjustment(self) -> None:
        """A player in Coors Field (PF=1.16) should have lower adjusted
        hitting value than the same stat line in a neutral park.
        """
        base = dict(
            player_id="coors",
            player_name="Coors Player",
            season=2023,
            team="COL",
            position="RF",
            pa=500,
            ab=450,
            hits=135,
            singles=85,
            doubles=25,
            triples=10,
            hr=15,
            bb=40,
            hbp=5,
            k=100,
            sf=5,
        )

        coors = PlayerSeason(**base, park_factor=1.16)
        neutral = PlayerSeason(**{**base, "team": "MIL"}, park_factor=1.00)

        result_coors = compute_hitting(coors, rng=np.random.default_rng(99))
        result_neutral = compute_hitting(neutral, rng=np.random.default_rng(99))

        assert result_coors.runs_mean < result_neutral.runs_mean, (
            f"Coors runs ({result_coors.runs_mean:.1f}) should be lower than "
            f"neutral ({result_neutral.runs_mean:.1f}) after park adjustment"
        )

    def test_pitcher_friendly_park_boosts_hitter(self) -> None:
        """A hitter in a pitcher-friendly park (PF < 1) should get credit."""
        base = dict(
            player_id="petco",
            player_name="Petco Player",
            season=2023,
            team="SDP",
            position="CF",
            pa=500,
            ab=450,
            hits=120,
            singles=75,
            doubles=25,
            triples=5,
            hr=15,
            bb=40,
            hbp=5,
            k=110,
            sf=5,
        )

        pitcher_park = PlayerSeason(**base, park_factor=0.92)
        neutral = PlayerSeason(**{**base, "team": "MIL"}, park_factor=1.00)

        result_pitcher = compute_hitting(pitcher_park, rng=np.random.default_rng(99))
        result_neutral = compute_hitting(neutral, rng=np.random.default_rng(99))

        assert result_pitcher.runs_mean > result_neutral.runs_mean, (
            f"Pitcher-park runs ({result_pitcher.runs_mean:.1f}) should exceed "
            f"neutral ({result_neutral.runs_mean:.1f})"
        )


# ---------------------------------------------------------------------------
# Credible intervals
# ---------------------------------------------------------------------------

class TestHittingCredibleIntervals:
    """Posterior CI properties that should always hold."""

    def test_credible_intervals_contain_mean(self, trout_2016: PlayerSeason) -> None:
        """The 90% credible interval must contain the posterior mean."""
        result = compute_hitting(trout_2016)

        lower, upper = result.ci_90
        assert lower <= result.runs_mean <= upper, (
            f"90% CI [{lower:.1f}, {upper:.1f}] does not contain "
            f"mean {result.runs_mean:.1f}"
        )

    def test_ci_50_narrower_than_ci_90(self, trout_2016: PlayerSeason) -> None:
        """The 50% CI should be strictly narrower than the 90% CI."""
        result = compute_hitting(trout_2016)

        width_50 = result.ci_50[1] - result.ci_50[0]
        width_90 = result.ci_90[1] - result.ci_90[0]
        assert width_50 < width_90, (
            f"50% CI width ({width_50:.2f}) should be less than "
            f"90% CI width ({width_90:.2f})"
        )

    def test_ci_lower_less_than_upper(self, trout_2016: PlayerSeason) -> None:
        """Lower bound should always be strictly less than upper bound."""
        result = compute_hitting(trout_2016)

        assert result.ci_90[0] < result.ci_90[1], "CI lower must be < upper"
        assert result.ci_50[0] < result.ci_50[1], "CI lower must be < upper"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestHittingReproducibility:
    """Same seed should yield identical results."""

    def test_same_seed_same_result(self, trout_2016: PlayerSeason) -> None:
        """Two calls with the same RNG seed must produce bit-identical output."""
        r1 = compute_hitting(trout_2016, rng=np.random.default_rng(12345))
        r2 = compute_hitting(trout_2016, rng=np.random.default_rng(12345))

        assert r1.runs_mean == r2.runs_mean, "Mean should be identical across seeds"
        assert r1.runs_var == r2.runs_var, "Variance should be identical across seeds"
        assert r1.ci_90 == r2.ci_90, "CIs should be identical across seeds"

    def test_different_seed_different_samples(self, trout_2016: PlayerSeason) -> None:
        """Different seeds should produce different posterior sample arrays."""
        r1 = compute_hitting(trout_2016, rng=np.random.default_rng(1))
        r2 = compute_hitting(trout_2016, rng=np.random.default_rng(2))

        # Means will be the same (analytic), but CIs from samples may differ slightly.
        assert r1.samples is not None and r2.samples is not None
        assert not np.array_equal(r1.samples, r2.samples), (
            "Different seeds should produce different sample arrays"
        )
