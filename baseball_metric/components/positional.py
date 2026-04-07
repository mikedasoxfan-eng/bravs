"""Positional adjustment component of BRAVS.

Credits players at premium defensive positions for the offensive
opportunity cost of playing that position. A shortstop who hits
league-average is more valuable than a first baseman who hits
league-average because shortstops are scarce.
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

    The positional adjustment is a deterministic credit/penalty based
    on the defensive spectrum. It has near-zero uncertainty since it
    reflects a known structural feature of baseball, not a measurement.

    Prorated by games played relative to a full 162-game season.

    Args:
        player: PlayerSeason with position and games played.
        rng: Random generator for reproducibility.
        n_samples: Number of posterior samples.

    Returns:
        ComponentResult with positional adjustment in runs.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pos = player.position
    adj_per_162 = POS_ADJ.get(pos, 0.0)

    # Prorate by games played
    games_fraction = player.games / GAMES_PER_SEASON if player.games > 0 else 0.0
    adj_runs = adj_per_162 * games_fraction

    # Small uncertainty reflecting multi-position players and position ambiguity
    adj_var = 1.0  # ~1 run of uncertainty in positional classification

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
            "position": pos,
            "adj_per_162": adj_per_162,
            "games_fraction": round(games_fraction, 3),
        },
    )
