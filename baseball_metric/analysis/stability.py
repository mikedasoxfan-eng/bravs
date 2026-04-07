"""Reliability and stability analysis for BRAVS.

Provides tools to evaluate how reproducible and stable the BRAVS metric
is across random game splits and across consecutive seasons.  High
reliability means the metric is measuring persistent player skill rather
than noise.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import BRAVSResult, PlayerSeason


def _split_season_in_half(
    player: PlayerSeason,
    rng: np.random.Generator,
) -> tuple[PlayerSeason, PlayerSeason]:
    """Randomly split a PlayerSeason's counting stats into two halves.

    Each counting stat is allocated to half-A or half-B on a per-game
    basis by drawing each game uniformly at random.  Rate stats and
    context fields are copied unchanged.

    Args:
        player: A fully populated ``PlayerSeason``.
        rng: NumPy random generator for reproducibility.

    Returns:
        Two ``PlayerSeason`` objects representing each half.
    """
    import copy

    half_a = copy.deepcopy(player)
    half_b = copy.deepcopy(player)

    total_games = max(player.games, 1)

    # For each game, assign it to half A or B with equal probability.
    assignments = rng.integers(0, 2, size=total_games)
    frac_a = float(assignments.sum()) / total_games
    frac_b = 1.0 - frac_a

    # Scale counting stats proportionally.
    counting_int_fields = [
        "pa", "ab", "hits", "singles", "doubles", "triples", "hr",
        "bb", "ibb", "hbp", "k", "sf", "sh", "sb", "cs", "gidp",
        "games", "er", "hits_allowed", "hr_allowed", "bb_allowed",
        "hbp_allowed", "k_pitching", "games_pitched", "games_started",
        "saves", "holds", "catcher_pitches", "extra_bases_taken",
        "extra_base_opportunities", "outs_on_bases", "pitches_seen",
    ]
    counting_float_fields = [
        "ip", "inn_fielded",
    ]

    for field_name in counting_int_fields:
        total = getattr(player, field_name, 0)
        val_a = int(round(total * frac_a))
        val_b = total - val_a
        setattr(half_a, field_name, val_a)
        setattr(half_b, field_name, val_b)

    for field_name in counting_float_fields:
        total = getattr(player, field_name, 0.0)
        val_a = total * frac_a
        val_b = total * frac_b
        setattr(half_a, field_name, val_a)
        setattr(half_b, field_name, val_b)

    # Fielding rate stats -- scale proportionally.
    for metric in ("uzr", "drs", "oaa", "framing_runs", "blocking_runs", "throwing_runs"):
        val = getattr(player, metric, None)
        if val is not None:
            setattr(half_a, metric, val * frac_a)
            setattr(half_b, metric, val * frac_b)

    return half_a, half_b


def split_half_reliability(
    player_seasons: Sequence[PlayerSeason],
    n_splits: int = 1000,
    seed: int = 42,
    n_samples: int = 2000,
) -> NDArray[np.floating[object]]:
    """Estimate split-half reliability of BRAVS via random game splits.

    For each of *n_splits* iterations, every season in *player_seasons* is
    randomly split into two halves.  BRAVS is computed for each half and
    the Pearson correlation between the two half-vectors is recorded.

    Args:
        player_seasons: List of player-seasons to include.  Should contain
            enough players (ideally 50+) for a meaningful correlation.
        n_splits: Number of random split iterations.
        seed: Master random seed.
        n_samples: Posterior samples per BRAVS computation.

    Returns:
        1-D array of *n_splits* Pearson *r* values representing the
        distribution of split-half correlations.
    """
    rng = np.random.default_rng(seed)
    correlations = np.empty(n_splits, dtype=np.float64)

    for i in range(n_splits):
        bravs_a: list[float] = []
        bravs_b: list[float] = []
        iter_seed = int(rng.integers(0, 2**31))

        for ps in player_seasons:
            half_a, half_b = _split_season_in_half(ps, np.random.default_rng(iter_seed))
            result_a = compute_bravs(half_a, n_samples=n_samples, seed=iter_seed)
            result_b = compute_bravs(half_b, n_samples=n_samples, seed=iter_seed + 1)
            bravs_a.append(result_a.bravs)
            bravs_b.append(result_b.bravs)

        if len(bravs_a) >= 3:
            r, _ = stats.pearsonr(bravs_a, bravs_b)
            correlations[i] = r
        else:
            correlations[i] = np.nan

    return correlations


def year_over_year_correlation(
    player_seasons_y1: Sequence[PlayerSeason],
    player_seasons_y2: Sequence[PlayerSeason],
    n_samples: int = 4000,
    seed: int = 42,
) -> tuple[float, float, list[tuple[str, float, float]]]:
    """Compute the year-over-year BRAVS correlation for matched players.

    Players are matched on ``player_id``.  Only players present in both
    year-lists are included.

    Args:
        player_seasons_y1: Player-seasons for Year N.
        player_seasons_y2: Player-seasons for Year N+1.
        n_samples: Posterior samples per BRAVS computation.
        seed: Random seed.

    Returns:
        A tuple of ``(pearson_r, p_value, paired_values)`` where
        *paired_values* is a list of ``(player_id, bravs_y1, bravs_y2)``.
    """
    y1_map: dict[str, PlayerSeason] = {ps.player_id: ps for ps in player_seasons_y1}
    y2_map: dict[str, PlayerSeason] = {ps.player_id: ps for ps in player_seasons_y2}

    common_ids = sorted(set(y1_map.keys()) & set(y2_map.keys()))
    if len(common_ids) < 3:
        raise ValueError(
            f"Need at least 3 matched players, found {len(common_ids)}."
        )

    bravs_y1: list[float] = []
    bravs_y2: list[float] = []
    paired: list[tuple[str, float, float]] = []

    for pid in common_ids:
        r1 = compute_bravs(y1_map[pid], n_samples=n_samples, seed=seed)
        r2 = compute_bravs(y2_map[pid], n_samples=n_samples, seed=seed)
        bravs_y1.append(r1.bravs)
        bravs_y2.append(r2.bravs)
        paired.append((pid, r1.bravs, r2.bravs))

    r, p = stats.pearsonr(bravs_y1, bravs_y2)
    return float(r), float(p), paired


def reliability_coefficient(
    correlations: NDArray[np.floating[object]] | Sequence[float],
) -> dict[str, float]:
    """Compute the Spearman-Brown corrected reliability from split-half correlations.

    The Spearman-Brown prophecy formula corrects a split-half correlation
    to estimate the reliability of the full-length measure:

        r_full = 2 * r_half / (1 + r_half)

    Args:
        correlations: Array of split-half Pearson *r* values (e.g., from
            :func:`split_half_reliability`).

    Returns:
        Dict with keys:
        - ``mean_split_half_r``: mean of raw split-half correlations
        - ``median_split_half_r``: median of raw split-half correlations
        - ``spearman_brown_reliability``: corrected reliability estimate
        - ``ci_95_lower``: 2.5th percentile of corrected reliability
        - ``ci_95_upper``: 97.5th percentile of corrected reliability
    """
    corrs = np.asarray(correlations, dtype=np.float64)
    corrs = corrs[~np.isnan(corrs)]

    if len(corrs) == 0:
        return {
            "mean_split_half_r": np.nan,
            "median_split_half_r": np.nan,
            "spearman_brown_reliability": np.nan,
            "ci_95_lower": np.nan,
            "ci_95_upper": np.nan,
        }

    mean_r = float(np.mean(corrs))
    median_r = float(np.median(corrs))

    # Apply Spearman-Brown to each split to get a distribution of
    # corrected reliabilities.
    sb_corrs = 2.0 * corrs / (1.0 + np.abs(corrs))

    return {
        "mean_split_half_r": mean_r,
        "median_split_half_r": median_r,
        "spearman_brown_reliability": float(np.mean(sb_corrs)),
        "ci_95_lower": float(np.percentile(sb_corrs, 2.5)),
        "ci_95_upper": float(np.percentile(sb_corrs, 97.5)),
    }
