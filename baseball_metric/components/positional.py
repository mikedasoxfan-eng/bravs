"""Positional adjustment component of BRAVS.

Credits players at premium defensive positions for the offensive
opportunity cost of playing that position. Supports multi-position
players by weighting the adjustment by games at each position.
"""

from __future__ import annotations

import numpy as np

from baseball_metric.core.types import ComponentResult, PlayerSeason
from baseball_metric.utils.constants import GAMES_PER_SEASON, POS_ADJ


def compute_positional(
    player: PlayerSeason,
    rng: np.random.Generator | None = None,
    n_samples: int = 10000,
) -> ComponentResult:
    """Compute the positional adjustment component of BRAVS.

    For single-position players, applies the standard Tango-scale
    adjustment prorated by games. For multi-position players (when
    positions_played dict is provided), weights the adjustment by
    games at each position.

    Example: 80 games at SS (+7.5/162) + 50 games at 2B (+2.5/162)
    = 80/162 * 7.5 + 50/162 * 2.5 = 3.70 + 0.77 = 4.47 runs

    Args:
        player: PlayerSeason with position and games played.
        rng: Random generator for reproducibility.
        n_samples: Number of posterior samples.

    Returns:
        ComponentResult with positional adjustment in runs.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if player.positions_played and len(player.positions_played) > 1:
        # Multi-position: weight by games at each position
        adj_runs = 0.0
        total_games = sum(player.positions_played.values())
        pos_breakdown: dict[str, float] = {}
        for pos, games_at in player.positions_played.items():
            pos_adj = POS_ADJ.get(pos, 0.0)
            contribution = pos_adj * (games_at / GAMES_PER_SEASON)
            adj_runs += contribution
            pos_breakdown[pos] = round(contribution, 1)
    else:
        # Single position (or no multi-position data)
        pos = player.position
        adj_per_162 = POS_ADJ.get(pos, 0.0)
        games_fraction = player.games / GAMES_PER_SEASON if player.games > 0 else 0.0
        adj_runs = adj_per_162 * games_fraction
        total_games = player.games
        pos_breakdown = {pos: round(adj_runs, 1)}

    # Small uncertainty reflecting position classification ambiguity
    adj_var = 1.0

    samples = rng.normal(adj_runs, np.sqrt(adj_var), size=n_samples)

    from baseball_metric.utils.math_helpers import credible_interval

    ci50 = credible_interval(samples, 0.50)
    ci90 = credible_interval(samples, 0.90)

    return ComponentResult(
        name="positional",
        runs_mean=adj_runs,
        runs_var=adj_var,
        ci_50=ci50,
        ci_90=ci90,
        samples=samples,
        metadata={
            "positions": pos_breakdown,
            "total_games": total_games,
            "is_multi_position": len(pos_breakdown) > 1,
        },
    )
