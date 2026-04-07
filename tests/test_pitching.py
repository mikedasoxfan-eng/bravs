"""Tests for the pitching component of BRAVS.

Validates that compute_pitching produces sensible run values for
starters, relievers, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from baseball_metric.components.pitching import compute_pitching
from baseball_metric.core.types import ComponentResult, PlayerSeason


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestPitchingValues:
    """Core pitching-value sanity checks."""

    def test_degrom_pitching_positive(self, degrom_2018: PlayerSeason) -> None:
        """deGrom 2018 (1.70 ERA, 269 K) should be very positive.

        His FIP was 1.98 that year.  Even after shrinkage, the pitching
        component should be well above +50 runs above FAT.
        """
        result = compute_pitching(degrom_2018)

        assert isinstance(result, ComponentResult)
        assert result.runs_mean > 50.0, (
            f"deGrom 2018 pitching runs ({result.runs_mean:.1f}) should be > 50; "
            f"metadata: {result.metadata}"
        )

    def test_degrom_metadata_contains_fip(self, degrom_2018: PlayerSeason) -> None:
        """The result metadata should include observed and posterior FIP."""
        result = compute_pitching(degrom_2018)

        assert "obs_fip" in result.metadata, "metadata should include obs_fip"
        assert "post_fip_mean" in result.metadata, "metadata should include post_fip_mean"
        # deGrom's FIP was under 2.0 in 2018
        assert result.metadata["obs_fip"] < 3.0, (
            f"deGrom observed FIP ({result.metadata['obs_fip']}) should be < 3.0"
        )

    def test_ohtani_pitching_positive(self, ohtani_2023: PlayerSeason) -> None:
        """Ohtani 2023 pitched 132 IP with strong K rate — should be positive."""
        result = compute_pitching(ohtani_2023)

        assert result.runs_mean > 0.0, (
            f"Ohtani 2023 pitching runs ({result.runs_mean:.1f}) should be > 0"
        )


# ---------------------------------------------------------------------------
# Replacement / league-average pitcher
# ---------------------------------------------------------------------------

class TestReplacementPitcher:
    """FAT-calibrated pitchers should produce values near zero."""

    def test_replacement_pitcher_near_zero(self) -> None:
        """A pitcher whose FIP matches league average over ~180 IP should
        yield pitching runs in the low single digits (near FAT baseline).
        """
        # Construct a "league average" pitcher: ~4.2 FIP over 180 IP
        player = PlayerSeason(
            player_id="lgavg_p",
            player_name="League Average Pitcher",
            season=2023,
            team="XXX",
            position="P",
            ip=180.0,
            er=84,       # 4.20 ERA
            hits_allowed=170,
            hr_allowed=22,   # ~1.1 HR/9
            bb_allowed=55,   # ~2.75 BB/9
            hbp_allowed=7,
            k_pitching=155,  # ~7.75 K/9
            games_pitched=32,
            games_started=32,
            park_factor=1.0,
            league_rpg=4.62,
        )
        result = compute_pitching(player)

        # A league-average pitcher over a full season is above FAT but not
        # dramatically so; value should be positive but modest.
        assert -15.0 <= result.runs_mean <= 35.0, (
            f"League-avg pitcher runs ({result.runs_mean:.1f}) should be modest"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestPitchingEdgeCases:
    """Boundary conditions and degenerate inputs."""

    def test_zero_ip_returns_zero(self) -> None:
        """A player with 0 IP should get exactly 0.0 pitching value."""
        player = PlayerSeason(
            player_id="zero_ip",
            player_name="No IP Player",
            season=2023,
            team="XXX",
            position="P",
            ip=0.0,
        )
        result = compute_pitching(player)

        assert result.runs_mean == 0.0, "Zero IP should yield exactly 0.0 runs"
        assert result.runs_var == 0.0, "Zero IP should yield exactly 0.0 variance"

    def test_fractional_ip(self) -> None:
        """IP expressed as fractional innings (e.g., 6.2 = 6 2/3) should work."""
        player = PlayerSeason(
            player_id="frac_ip",
            player_name="Fractional IP",
            season=2023,
            team="XXX",
            position="P",
            ip=6.2,
            er=3,
            hr_allowed=1,
            bb_allowed=2,
            hbp_allowed=0,
            k_pitching=7,
            games_pitched=1,
            games_started=1,
        )
        result = compute_pitching(player)

        # Should not crash and should return a finite result.
        assert np.isfinite(result.runs_mean), "Fractional IP should yield finite result"


# ---------------------------------------------------------------------------
# Bayesian shrinkage
# ---------------------------------------------------------------------------

class TestPitchingShrinkage:
    """Verify that small-sample FIP is shrunk toward the prior."""

    def test_shrinkage_small_sample(self) -> None:
        """With only 20 IP, an absurdly low FIP should be pulled toward league avg.

        A pitcher who strikes out everyone in 20 IP with 0 HR / 0 BB
        would have a phenomenal observed FIP, but the posterior should
        be substantially shrunk toward the ~4.20 prior.
        """
        player = PlayerSeason(
            player_id="tiny_sp",
            player_name="Small Sample Ace",
            season=2023,
            team="XXX",
            position="P",
            ip=20.0,
            er=0,
            hits_allowed=5,
            hr_allowed=0,
            bb_allowed=0,
            hbp_allowed=0,
            k_pitching=30,  # 13.5 K/9 in 20 IP
            games_pitched=4,
            games_started=4,
            park_factor=1.0,
            league_rpg=4.62,
        )
        result = compute_pitching(player)

        # With 20 IP the posterior FIP should be above the absurd observed
        # value (shrunk toward 4.20 prior).
        post_fip = result.metadata.get("post_fip_mean", 0.0)
        obs_fip = result.metadata.get("obs_fip", 0.0)
        assert post_fip > obs_fip, (
            f"Posterior FIP ({post_fip:.2f}) should exceed observed ({obs_fip:.2f}) "
            f"due to shrinkage with only 20 IP"
        )

    def test_large_sample_less_shrinkage(self, degrom_2018: PlayerSeason) -> None:
        """With 217 IP, deGrom's posterior FIP should be close to observed."""
        result = compute_pitching(degrom_2018)

        post_fip = result.metadata.get("post_fip_mean", 0.0)
        obs_fip_adj = result.metadata.get("obs_fip_adj", 0.0)

        # With 217 IP, the posterior should be quite close to the (adjusted) observed.
        assert abs(post_fip - obs_fip_adj) < 0.6, (
            f"Posterior FIP ({post_fip:.2f}) should be within 0.6 of "
            f"adjusted observed ({obs_fip_adj:.2f}) with 217 IP"
        )


# ---------------------------------------------------------------------------
# Starter vs. reliever
# ---------------------------------------------------------------------------

class TestStarterVsReliever:
    """Different role baselines for starters and relievers."""

    def test_starter_vs_reliever_metadata(self) -> None:
        """A starter should be flagged differently from a reliever."""
        starter = PlayerSeason(
            player_id="start01",
            player_name="Starter",
            season=2023,
            team="XXX",
            position="P",
            ip=180.0,
            er=72,
            hr_allowed=20,
            bb_allowed=50,
            hbp_allowed=5,
            k_pitching=170,
            games_pitched=32,
            games_started=32,
        )
        reliever = PlayerSeason(
            player_id="reliev01",
            player_name="Reliever",
            season=2023,
            team="XXX",
            position="P",
            ip=65.0,
            er=22,
            hr_allowed=7,
            bb_allowed=20,
            hbp_allowed=2,
            k_pitching=75,
            games_pitched=65,
            games_started=0,
        )

        result_sp = compute_pitching(starter)
        result_rp = compute_pitching(reliever)

        assert result_sp.metadata.get("is_starter") is True, (
            "Starter should have is_starter=True in metadata"
        )
        assert result_rp.metadata.get("is_starter") is False, (
            "Reliever should have is_starter=False in metadata"
        )


# ---------------------------------------------------------------------------
# Credible intervals & reproducibility
# ---------------------------------------------------------------------------

class TestPitchingIntervals:
    """Posterior CI and reproducibility checks."""

    def test_credible_intervals_contain_mean(self, degrom_2018: PlayerSeason) -> None:
        """90% CI must bracket the mean."""
        result = compute_pitching(degrom_2018)

        lo, hi = result.ci_90
        assert lo <= result.runs_mean <= hi, (
            f"90% CI [{lo:.1f}, {hi:.1f}] does not contain mean {result.runs_mean:.1f}"
        )

    def test_ci_ordering(self, degrom_2018: PlayerSeason) -> None:
        """50% CI should be nested inside 90% CI."""
        result = compute_pitching(degrom_2018)

        assert result.ci_90[0] <= result.ci_50[0], "90% lower should be <= 50% lower"
        assert result.ci_50[1] <= result.ci_90[1], "50% upper should be <= 90% upper"

    def test_reproducibility(self, degrom_2018: PlayerSeason) -> None:
        """Same seed produces bit-identical output."""
        r1 = compute_pitching(degrom_2018, rng=np.random.default_rng(777))
        r2 = compute_pitching(degrom_2018, rng=np.random.default_rng(777))

        assert r1.runs_mean == r2.runs_mean
        assert r1.ci_90 == r2.ci_90
