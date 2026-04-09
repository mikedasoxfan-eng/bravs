"""Durability/availability component of BRAVS.

Credits players who are available for more games than expected,
and penalizes players who miss significant time. The value of
availability is measured as the marginal games × FAT-level
value per game — the team must replace missing players with
freely available talent.
"""

from __future__ import annotations

import numpy as np

from baseball_metric.core.types import ComponentResult, PlayerSeason
from baseball_metric.utils.constants import (
    EXPECTED_APPEARANCES_RELIEVER,
    EXPECTED_GAMES_POSITION,
    EXPECTED_STARTS_PITCHER,
    GAMES_PER_SEASON,
    MARGINAL_GAME_VALUE_PITCHER,
    MARGINAL_GAME_VALUE_POSITION,
)
from baseball_metric.utils.math_helpers import credible_interval


def _prorate_expected(full_season_expected: int, season_games: int) -> int:
    """Prorate expected games/starts for shortened seasons.

    In a 60-game season (2020), a position player's expected games
    should be ~57, not 155. This prevents penalizing players for
    games that simply didn't exist.
    """
    if season_games >= 150:
        return full_season_expected
    # Scale expected proportionally to actual season length, with 95% factor
    # (a healthy player should play ~95% of available games)
    scale = (season_games * 0.95) / GAMES_PER_SEASON
    return max(int(full_season_expected * scale), 1)


def compute_durability(
    player: PlayerSeason,
    rng: np.random.Generator | None = None,
    n_samples: int = 10000,
) -> ComponentResult:
    """Compute the durability/availability component of BRAVS.

    Durability value = (games_played - games_expected) × marginal_game_value

    For position players: games_expected ≈ 155 (prorated for short seasons)
    For starting pitchers: games_expected ≈ 32 starts (prorated)
    For relievers: games_expected ≈ 65 appearances (prorated)

    Expected games are prorated based on actual season length to avoid
    penalizing players for games that didn't exist (e.g., 2020's 60-game
    season).

    Args:
        player: PlayerSeason with games data.
        rng: Random generator for reproducibility.
        n_samples: Number of posterior samples.

    Returns:
        ComponentResult with durability runs above expected.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if player.is_pitcher:
        is_starter = player.games_started > player.games_pitched * 0.5
        if is_starter:
            expected = _prorate_expected(EXPECTED_STARTS_PITCHER, player.season_games)
            actual = player.games_started
            marginal = MARGINAL_GAME_VALUE_PITCHER * 2.0  # starts are worth more than appearances
        else:
            expected = _prorate_expected(EXPECTED_APPEARANCES_RELIEVER, player.season_games)
            actual = player.games_pitched
            marginal = MARGINAL_GAME_VALUE_PITCHER
    else:
        expected = _prorate_expected(EXPECTED_GAMES_POSITION, player.season_games)
        actual = player.games
        marginal = MARGINAL_GAME_VALUE_POSITION

    # Durability runs (in wins, convert to runs later in aggregation)
    games_delta = actual - expected
    # Convert marginal game value (in wins) to runs: wins × RPW
    # We use an approximate RPW here; the final conversion happens in model.py
    approx_rpw = 9.8
    durability_runs = games_delta * marginal * approx_rpw

    # Small variance: durability is observed, not estimated
    durability_var = 2.0  # ~1.4 runs SD, reflecting uncertainty in marginal value

    samples = rng.normal(durability_runs, np.sqrt(durability_var), size=n_samples)

    ci50 = credible_interval(samples, 0.50)
    ci90 = credible_interval(samples, 0.90)

    return ComponentResult(
        name="durability",
        runs_mean=durability_runs,
        runs_var=durability_var,
        ci_50=ci50,
        ci_90=ci90,
        samples=samples,
        metadata={
            "games_actual": actual,
            "games_expected": expected,
            "games_delta": games_delta,
            "marginal_value_per_game_wins": round(marginal, 4),
        },
    )
