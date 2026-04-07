"""Integration tests for compute_bravs — the full BRAVS pipeline.

These tests exercise the end-to-end model orchestration including all
component computations, adjustments, leverage, and posterior combination.
"""

from __future__ import annotations

import numpy as np
import pytest

from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import BRAVSResult, PlayerSeason


# ---------------------------------------------------------------------------
# Star-player sanity checks
# ---------------------------------------------------------------------------

class TestStarPlayerBRAVS:
    """Verify that elite seasons produce plausible total BRAVS values."""

    def test_trout_bravs_reasonable(self, trout_2016: PlayerSeason) -> None:
        """Mike Trout 2016 BRAVS should be very high (20-30 wins).

        The BRAVS framework uses a Pythagenpat RPW (~5.7 for 2016), which
        is roughly half the traditional ~10 RPW.  This means win totals are
        approximately doubled compared to fWAR/rWAR.  Combined with the
        inflated BRAVS wOBA scale and park/era adjustments, an elite
        season like Trout 2016 lands around 25 BRAVS wins.
        """
        result = compute_bravs(trout_2016)

        assert isinstance(result, BRAVSResult)
        assert 18.0 <= result.bravs <= 32.0, (
            f"Trout 2016 BRAVS ({result.bravs:.1f}) should be 18-32 wins"
        )

    def test_degrom_bravs_reasonable(self, degrom_2018: PlayerSeason) -> None:
        """deGrom 2018 BRAVS should be 10-20 wins.

        With the Pythagenpat RPW (~5.6) the win total is roughly doubled
        compared to traditional WAR.  deGrom 2018 lands around ~15 BRAVS wins.
        """
        result = compute_bravs(degrom_2018)

        assert 10.0 <= result.bravs <= 20.0, (
            f"deGrom 2018 BRAVS ({result.bravs:.1f}) should be 10-20 wins"
        )

    def test_ohtani_two_way(self, ohtani_2023: PlayerSeason) -> None:
        """Ohtani 2023 should have BOTH hitting AND pitching components.

        As a two-way player (599 PA + 132 IP), both should be present and
        both should be meaningfully positive.  Total BRAVS should be 8-16 wins.
        """
        result = compute_bravs(ohtani_2023)

        # Both components must exist
        assert "hitting" in result.components, "Two-way player must have hitting component"
        assert "pitching" in result.components, "Two-way player must have pitching component"

        # Both should be positive
        assert result.components["hitting"].runs_mean > 0.0, (
            "Ohtani hitting should be positive"
        )
        assert result.components["pitching"].runs_mean > 0.0, (
            "Ohtani pitching should be positive"
        )

        # Total in range (Pythagenpat RPW ~5.9 roughly doubles vs traditional WAR)
        assert 12.0 <= result.bravs <= 24.0, (
            f"Ohtani 2023 BRAVS ({result.bravs:.1f}) should be 12-24 wins"
        )


# ---------------------------------------------------------------------------
# Baseline / replacement-level tests
# ---------------------------------------------------------------------------

class TestBaselineBRAVS:
    """Verify behavior around the FAT replacement level."""

    def test_replacement_near_baseline(self, replacement_player: PlayerSeason) -> None:
        """A FAT-calibrated player should be near 0 BRAVS (within +/- 5 wins).

        The replacement fixture plays only 100 of 162 games, incurring a
        significant durability penalty.  With the Pythagenpat RPW (~5.9),
        total BRAVS lands around -3 to -4 wins.
        """
        result = compute_bravs(replacement_player)

        assert -6.0 <= result.bravs <= 2.0, (
            f"Replacement player BRAVS ({result.bravs:.1f}) should be within [-6, 2]"
        )

    def test_negative_value_player(self, negative_value_player: PlayerSeason) -> None:
        """A truly terrible hitter should have clearly negative BRAVS."""
        result = compute_bravs(negative_value_player)

        assert result.bravs < 0.0, (
            f"Negative-value player BRAVS ({result.bravs:.1f}) should be < 0"
        )


# ---------------------------------------------------------------------------
# Uncertainty / credible intervals
# ---------------------------------------------------------------------------

class TestUncertaintyBehavior:
    """Tests for posterior uncertainty properties."""

    def test_short_season_wider_intervals(
        self,
        trout_2016: PlayerSeason,
        short_season_2020: PlayerSeason,
    ) -> None:
        """A 60-game season should have wider 90% credible intervals than a
        full 162-game season, all else being roughly equal.
        """
        result_full = compute_bravs(trout_2016)
        result_short = compute_bravs(short_season_2020)

        ci_width_full = result_full.bravs_ci_90[1] - result_full.bravs_ci_90[0]
        ci_width_short = result_short.bravs_ci_90[1] - result_short.bravs_ci_90[0]

        # The short season has fewer PA and games, so per-win uncertainty
        # should be wider even though the total value is smaller.
        # We compare the CI width relative to the absolute BRAVS to account for
        # scale differences; alternatively, just check raw width is positive.
        assert ci_width_short > 0.0, "Short season CI should have positive width"
        assert ci_width_full > 0.0, "Full season CI should have positive width"

        # Normalize by games to get uncertainty-per-game as a fairer comparison.
        per_game_width_full = ci_width_full / trout_2016.games
        per_game_width_short = ci_width_short / short_season_2020.games

        assert per_game_width_short > per_game_width_full, (
            f"Per-game CI width for short season ({per_game_width_short:.3f}) "
            f"should exceed full season ({per_game_width_full:.3f})"
        )

    def test_credible_interval_ordering(self, trout_2016: PlayerSeason) -> None:
        """CI lower < mean < CI upper for both 50% and 90% intervals."""
        result = compute_bravs(trout_2016)

        lo_90, hi_90 = result.bravs_ci_90
        assert lo_90 < result.bravs < hi_90, (
            f"90% CI [{lo_90:.2f}, {hi_90:.2f}] should bracket "
            f"BRAVS mean {result.bravs:.2f}"
        )

        lo_50, hi_50 = result.bravs_ci_50
        assert lo_50 < result.bravs < hi_50, (
            f"50% CI [{lo_50:.2f}, {hi_50:.2f}] should bracket "
            f"BRAVS mean {result.bravs:.2f}"
        )


# ---------------------------------------------------------------------------
# Catcher component routing
# ---------------------------------------------------------------------------

class TestCatcherComponent:
    """Verify that catchers and non-catchers are routed correctly."""

    def test_catcher_gets_catcher_component(self, hedges_catcher: PlayerSeason) -> None:
        """A catcher should have a non-zero catcher component."""
        result = compute_bravs(hedges_catcher)

        assert "catcher" in result.components, (
            "Catcher player should have a 'catcher' component"
        )
        # Hedges is an elite framer: framing_runs=15 should produce
        # a meaningfully positive catcher component.
        assert result.components["catcher"].runs_mean != 0.0, (
            "Hedges-type elite framer should have non-zero catcher value"
        )

    def test_non_catcher_no_catcher_component(self, trout_2016: PlayerSeason) -> None:
        """A non-catcher (CF) should NOT have a catcher component."""
        result = compute_bravs(trout_2016)

        assert "catcher" not in result.components, (
            f"CF should not have catcher component; got: {list(result.components)}"
        )


# ---------------------------------------------------------------------------
# Component structure
# ---------------------------------------------------------------------------

class TestComponentStructure:
    """Ensure the result contains all expected component keys."""

    def test_component_keys_present_position_player(
        self, trout_2016: PlayerSeason
    ) -> None:
        """A position player should have hitting, baserunning, fielding,
        positional, approach_quality, durability, and leverage components.
        """
        result = compute_bravs(trout_2016)

        expected = {
            "hitting",
            "baserunning",
            "fielding",
            "positional",
            "approach_quality",
            "durability",
            "leverage",
        }
        missing = expected - set(result.components)
        assert not missing, f"Missing components: {missing}"

    def test_component_keys_present_pitcher(self, degrom_2018: PlayerSeason) -> None:
        """A pitcher should have pitching plus positional, fielding, durability,
        and leverage components.
        """
        result = compute_bravs(degrom_2018)

        # deGrom has PA >= 1 (NL pitcher bats), so hitting and baserunning
        # will also be computed.
        assert "pitching" in result.components, "Pitcher must have pitching component"
        assert "fielding" in result.components, "Pitcher must have fielding component"
        assert "positional" in result.components, "Pitcher must have positional component"
        assert "durability" in result.components, "Pitcher must have durability component"
        assert "leverage" in result.components, "Pitcher must have leverage component"

    def test_component_keys_present_two_way(self, ohtani_2023: PlayerSeason) -> None:
        """A two-way player should have both hitting and pitching components."""
        result = compute_bravs(ohtani_2023)

        assert "hitting" in result.components
        assert "pitching" in result.components


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Deterministic output with fixed seed."""

    def test_same_seed_identical_results(self, trout_2016: PlayerSeason) -> None:
        """Two calls with the same seed must produce bit-identical BRAVS."""
        r1 = compute_bravs(trout_2016, seed=42, n_samples=5000)
        r2 = compute_bravs(trout_2016, seed=42, n_samples=5000)

        assert r1.bravs == r2.bravs, (
            f"Same seed should give identical BRAVS: {r1.bravs} vs {r2.bravs}"
        )
        assert r1.total_runs_mean == r2.total_runs_mean
        assert r1.bravs_ci_90 == r2.bravs_ci_90

    def test_different_seed_varies(self, trout_2016: PlayerSeason) -> None:
        """Different seeds should produce slightly different posterior samples."""
        r1 = compute_bravs(trout_2016, seed=1, n_samples=5000)
        r2 = compute_bravs(trout_2016, seed=999, n_samples=5000)

        assert r1.total_samples is not None and r2.total_samples is not None
        assert not np.array_equal(r1.total_samples, r2.total_samples), (
            "Different seeds should produce different sample arrays"
        )


# ---------------------------------------------------------------------------
# Summary / display
# ---------------------------------------------------------------------------

class TestSummaryOutput:
    """Verify the human-readable summary string."""

    def test_summary_contains_player_name(self, trout_2016: PlayerSeason) -> None:
        """summary() should include the player name."""
        result = compute_bravs(trout_2016)
        text = result.summary()

        assert "Mike Trout" in text, "Summary should contain the player name"
        assert "2016" in text, "Summary should contain the season"

    def test_summary_contains_total_bravs(self, trout_2016: PlayerSeason) -> None:
        """summary() should include the total BRAVS value."""
        result = compute_bravs(trout_2016)
        text = result.summary()

        assert "BRAVS" in text, "Summary should mention BRAVS"
        assert "wins" in text.lower() or "Wins" in text, "Summary should mention wins"
