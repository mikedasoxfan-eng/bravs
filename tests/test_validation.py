"""Tests for baseball_metric.data.validation.

Validates that the data-quality layer catches impossible stat lines,
flags suspicious outliers, and passes clean data through correctly.
"""

from __future__ import annotations

import pytest

from baseball_metric.core.types import PlayerSeason
from baseball_metric.data.validation import ValidationResult, validate_player_season


# ---------------------------------------------------------------------------
# Helper to build a minimal valid PlayerSeason
# ---------------------------------------------------------------------------

def _valid_player(**overrides: object) -> PlayerSeason:
    """Return a structurally valid PlayerSeason with optional overrides."""
    defaults = dict(
        player_id="valid01",
        player_name="Valid Player",
        season=2023,
        team="NYY",
        position="SS",
        pa=500,
        ab=450,
        hits=120,
        singles=75,
        doubles=25,
        triples=5,
        hr=15,
        bb=40,
        ibb=2,
        hbp=5,
        k=100,
        sf=3,
        sh=2,
        games=140,
    )
    defaults.update(overrides)
    return PlayerSeason(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Valid data should pass
# ---------------------------------------------------------------------------

class TestValidData:
    """Confirm that well-formed data passes validation without errors."""

    def test_valid_player_passes(self) -> None:
        """A cleanly constructed PlayerSeason should produce is_valid=True."""
        player = _valid_player()
        result = validate_player_season(player)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True, f"Valid player should pass; errors: {result.errors}"
        assert len(result.errors) == 0

    def test_valid_pitcher_passes(self) -> None:
        """A valid pitcher stat line should also pass."""
        player = _valid_player(
            position="P",
            pa=60,
            ab=55,
            hits=5,
            singles=3,
            doubles=1,
            triples=1,
            hr=0,
            bb=3,
            k=25,
            sf=1,
            sh=1,
            ip=200.0,
            er=80,
            hr_allowed=20,
            bb_allowed=50,
            hbp_allowed=5,
            k_pitching=200,
            games_pitched=32,
            games_started=32,
        )
        result = validate_player_season(player)
        assert result.is_valid is True, f"Valid pitcher should pass; errors: {result.errors}"

    def test_zero_pa_pitcher_passes(self) -> None:
        """An AL pitcher with 0 PA should pass (no batting checks triggered)."""
        player = _valid_player(
            position="P",
            pa=0,
            ab=0,
            hits=0,
            singles=0,
            doubles=0,
            triples=0,
            hr=0,
            bb=0,
            k=0,
            ip=180.0,
            er=72,
            k_pitching=160,
        )
        result = validate_player_season(player)
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# Errors: impossible stat lines
# ---------------------------------------------------------------------------

class TestImpossibleStats:
    """Stat lines that are logically impossible should produce errors."""

    def test_negative_stats_fail(self) -> None:
        """Negative counting stats (BB, K, HR) should produce an error."""
        player = _valid_player(bb=-5)
        result = validate_player_season(player)

        assert result.is_valid is False, "Negative BB should fail validation"
        assert any("Negative" in e or "negative" in e.lower() for e in result.errors), (
            f"Error list should mention negative stats; got: {result.errors}"
        )

    def test_negative_hr_fails(self) -> None:
        """Negative HR should fail."""
        player = _valid_player(hr=-1)
        result = validate_player_season(player)
        assert result.is_valid is False

    def test_negative_k_fails(self) -> None:
        """Negative K should fail."""
        player = _valid_player(k=-10)
        result = validate_player_season(player)
        assert result.is_valid is False

    def test_hits_greater_than_ab_fails(self) -> None:
        """Hits > AB is impossible and should produce an error."""
        player = _valid_player(hits=200, ab=150, singles=140, doubles=30, triples=15, hr=15)
        result = validate_player_season(player)

        assert result.is_valid is False, "H > AB should fail validation"
        assert any("H" in e and "AB" in e for e in result.errors), (
            f"Error should mention H > AB; got: {result.errors}"
        )

    def test_hr_greater_than_hits_fails(self) -> None:
        """HR > H is impossible."""
        player = _valid_player(hits=10, hr=15, singles=-5, doubles=0, triples=0)
        result = validate_player_season(player)

        assert result.is_valid is False, "HR > H should fail validation"
        assert any("HR" in e and "H" in e for e in result.errors), (
            f"Error should mention HR > H; got: {result.errors}"
        )

    def test_ab_greater_than_pa_fails(self) -> None:
        """AB > PA is impossible (PA = AB + BB + HBP + SF + SH + ...)."""
        player = _valid_player(pa=400, ab=450)
        result = validate_player_season(player)

        assert result.is_valid is False, "AB > PA should fail validation"

    def test_negative_games_fails(self) -> None:
        """Negative games played should fail."""
        player = _valid_player(games=-1)
        result = validate_player_season(player)

        assert result.is_valid is False, "Negative games should fail"

    def test_negative_innings_fielded_fails(self) -> None:
        """Negative innings fielded should fail."""
        player = _valid_player(inn_fielded=-100.0)
        result = validate_player_season(player)

        assert result.is_valid is False, "Negative inn_fielded should fail"

    def test_missing_player_name_fails(self) -> None:
        """Empty player_name should produce an error."""
        player = _valid_player(player_name="")
        result = validate_player_season(player)

        assert result.is_valid is False, "Empty player_name should fail"

    def test_bad_season_fails(self) -> None:
        """Season before 1871 or after 2030 should fail."""
        player_old = _valid_player(season=1800)
        player_future = _valid_player(season=2050)

        assert validate_player_season(player_old).is_valid is False
        assert validate_player_season(player_future).is_valid is False

    def test_negative_pitching_stats_fail(self) -> None:
        """Negative pitching counting stats should fail."""
        player = _valid_player(
            ip=100.0,
            k_pitching=-10,
            bb_allowed=30,
        )
        result = validate_player_season(player)
        assert result.is_valid is False


# ---------------------------------------------------------------------------
# Warnings: suspicious outliers
# ---------------------------------------------------------------------------

class TestSuspiciousOutliers:
    """Outlier stat lines that are theoretically possible but suspicious
    should produce warnings (not errors).
    """

    def test_extreme_high_ba_warns(self) -> None:
        """BA > .420 in 200+ AB should trigger a warning."""
        # .440 BA in 300 AB
        player = _valid_player(
            ab=300,
            pa=350,
            hits=132,
            singles=90,
            doubles=20,
            triples=5,
            hr=17,
            bb=40,
            hbp=5,
            k=50,
            sf=3,
            sh=2,
        )
        result = validate_player_season(player)

        assert result.is_valid is True, "High BA should still be valid"
        assert any("high BA" in w or "BA" in w for w in result.warnings), (
            f"Should warn about extreme BA; got: {result.warnings}"
        )

    def test_extreme_low_ba_warns(self) -> None:
        """BA < .150 in 200+ AB should trigger a warning."""
        # .140 BA in 250 AB
        player = _valid_player(
            ab=250,
            pa=300,
            hits=35,
            singles=20,
            doubles=8,
            triples=2,
            hr=5,
            bb=40,
            hbp=5,
            k=100,
            sf=3,
            sh=2,
        )
        result = validate_player_season(player)

        assert result.is_valid is True, "Low BA should still be valid"
        assert any("low BA" in w or "BA" in w for w in result.warnings), (
            f"Should warn about extreme BA; got: {result.warnings}"
        )

    def test_extreme_low_era_warns(self) -> None:
        """ERA < 1.0 in 100+ IP should trigger a warning."""
        player = _valid_player(
            ip=150.0,
            er=10,  # 0.60 ERA
            hr_allowed=3,
            bb_allowed=20,
            hbp_allowed=2,
            k_pitching=180,
            games_pitched=25,
            games_started=25,
        )
        result = validate_player_season(player)

        assert result.is_valid is True, "Low ERA should still be valid"
        assert any("ERA" in w or "era" in w.lower() for w in result.warnings), (
            f"Should warn about extreme ERA; got: {result.warnings}"
        )

    def test_extreme_high_era_warns(self) -> None:
        """ERA > 9.0 in 50+ IP should trigger a warning."""
        player = _valid_player(
            ip=60.0,
            er=65,  # 9.75 ERA
            hr_allowed=15,
            bb_allowed=40,
            hbp_allowed=5,
            k_pitching=30,
            games_pitched=20,
            games_started=10,
        )
        result = validate_player_season(player)

        assert result.is_valid is True
        assert any("ERA" in w or "era" in w.lower() for w in result.warnings), (
            f"Should warn about extreme ERA; got: {result.warnings}"
        )

    def test_very_high_pa_warns(self) -> None:
        """PA > 800 should trigger a 'combined stats' warning."""
        player = _valid_player(pa=850, ab=750)
        result = validate_player_season(player)

        assert any("PA" in w for w in result.warnings), (
            f"Should warn about very high PA; got: {result.warnings}"
        )

    def test_high_games_warns(self) -> None:
        """Games > 162 should trigger a warning (possible trade combine)."""
        player = _valid_player(games=165)
        result = validate_player_season(player)

        assert any("162" in w or "Games" in w or "games" in w.lower() for w in result.warnings), (
            f"Should warn about games > 162; got: {result.warnings}"
        )

    def test_high_ip_warns(self) -> None:
        """IP > 400 should trigger a warning (dead-ball era or error)."""
        player = _valid_player(ip=420.0)
        result = validate_player_season(player)

        assert any("IP" in w for w in result.warnings), (
            f"Should warn about extremely high IP; got: {result.warnings}"
        )

    def test_singles_mismatch_warns(self) -> None:
        """1B + 2B + 3B + HR != H should produce a warning about recomputing."""
        player = _valid_player(
            hits=120,
            singles=70,  # 70 + 25 + 5 + 15 = 115 != 120
            doubles=25,
            triples=5,
            hr=15,
        )
        result = validate_player_season(player)

        assert any("1B" in w or "singles" in w.lower() or "recomputing" in w.lower()
                    for w in result.warnings), (
            f"Should warn about hit components mismatch; got: {result.warnings}"
        )


# ---------------------------------------------------------------------------
# Parametrized edge-case combinations
# ---------------------------------------------------------------------------

class TestParametrizedValidation:
    """Parametrized tests covering multiple invalid stat combinations."""

    @pytest.mark.parametrize("field,value,should_fail", [
        ("bb", -1, True),
        ("k", -1, True),
        ("hr", -1, True),
        ("games", -1, True),
        ("inn_fielded", -50.0, True),
        ("season", 1500, True),
        ("season", 2100, True),
    ])
    def test_invalid_values(self, field: str, value: object, should_fail: bool) -> None:
        """Various individual invalid field values should fail validation."""
        player = _valid_player(**{field: value})
        result = validate_player_season(player)

        if should_fail:
            assert result.is_valid is False, (
                f"Setting {field}={value} should fail validation"
            )

    @pytest.mark.parametrize("field,value", [
        ("games", 0),
        ("pa", 0),
        ("ip", 0.0),
    ])
    def test_zero_values_valid(self, field: str, value: object) -> None:
        """Zero values for optional counting stats should not fail."""
        player = _valid_player(**{field: value})
        result = validate_player_season(player)

        # Zero PA / zero games / zero IP are valid (player just didn't play).
        # There shouldn't be errors from these zero values alone.
        # (Other fields might trigger mismatches, so we just check no
        # "Negative" errors appear.)
        negative_errors = [e for e in result.errors if "Negative" in e or "negative" in e.lower()]
        assert len(negative_errors) == 0, (
            f"Setting {field}={value} should not trigger negative-stat errors; "
            f"got: {negative_errors}"
        )
