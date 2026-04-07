"""Historical validation and backtesting for BRAVS.

Computes BRAVS across large sets of player-seasons, compares the results
against established WAR implementations (fWAR from FanGraphs, bWAR from
Baseball-Reference), and produces leaderboards with credible intervals.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import BRAVSResult, PlayerSeason


def compute_historical_bravs(
    player_seasons: Sequence[PlayerSeason],
    n_samples: int = 10000,
    seed: int = 42,
) -> list[BRAVSResult]:
    """Compute BRAVS for every player-season in the input list.

    Args:
        player_seasons: Iterable of fully populated ``PlayerSeason`` objects.
        n_samples: Posterior samples per computation.
        seed: Base random seed; each player gets ``seed + index`` for
            reproducibility while avoiding identical streams.

    Returns:
        A list of ``BRAVSResult`` objects in the same order as the input.
    """
    results: list[BRAVSResult] = []
    for idx, ps in enumerate(player_seasons):
        result = compute_bravs(ps, n_samples=n_samples, seed=seed + idx)
        results.append(result)
    return results


def compare_to_war(
    bravs_results: Sequence[BRAVSResult],
    war_values: pd.DataFrame,
) -> pd.DataFrame:
    """Compare BRAVS to fWAR and bWAR for the same player-seasons.

    Args:
        bravs_results: List of ``BRAVSResult`` objects.
        war_values: DataFrame with columns ``player_id``, ``season``,
            ``fWAR``, and ``bWAR``.  Additional columns are preserved.

    Returns:
        A merged DataFrame with columns:
        - ``player_id``, ``player_name``, ``season``, ``team``, ``position``
        - ``BRAVS``, ``BRAVS_ci90_lo``, ``BRAVS_ci90_hi``
        - ``fWAR``, ``bWAR``
        - ``BRAVS_minus_fWAR``, ``BRAVS_minus_bWAR``
        - One column per component (runs)
    """
    rows: list[dict[str, object]] = []
    for r in bravs_results:
        ci90 = r.bravs_ci_90
        row: dict[str, object] = {
            "player_id": r.player.player_id,
            "player_name": r.player.player_name,
            "season": r.player.season,
            "team": r.player.team,
            "position": r.player.position,
            "BRAVS": round(r.bravs, 2),
            "BRAVS_ci90_lo": round(ci90[0], 2),
            "BRAVS_ci90_hi": round(ci90[1], 2),
            "total_runs": round(r.total_runs_mean, 1),
            "rpw": round(r.rpw, 2),
        }
        # Component breakdown in runs.
        for comp_name, comp in r.components.items():
            row[f"comp_{comp_name}_runs"] = round(comp.runs_mean, 1)
        rows.append(row)

    bravs_df = pd.DataFrame(rows)

    # Merge with external WAR values.
    merged = bravs_df.merge(
        war_values[["player_id", "season", "fWAR", "bWAR"]].drop_duplicates(),
        on=["player_id", "season"],
        how="left",
    )

    merged["BRAVS_minus_fWAR"] = merged["BRAVS"] - merged["fWAR"]
    merged["BRAVS_minus_bWAR"] = merged["BRAVS"] - merged["bWAR"]

    return merged.sort_values("BRAVS", ascending=False).reset_index(drop=True)


def generate_leaderboard(
    bravs_results: Sequence[BRAVSResult],
    top_n: int = 25,
) -> pd.DataFrame:
    """Produce a leaderboard DataFrame sorted by BRAVS.

    Args:
        bravs_results: List of ``BRAVSResult`` objects.
        top_n: Number of top players to include.  Pass ``0`` or a value
            larger than the list length to include everyone.

    Returns:
        DataFrame with columns:
        - ``rank``, ``player_name``, ``season``, ``team``, ``position``
        - ``BRAVS``, ``BRAVS_ci50_lo``, ``BRAVS_ci50_hi``
        - ``BRAVS_ci90_lo``, ``BRAVS_ci90_hi``
        - One column per component (wins)
    """
    rows: list[dict[str, object]] = []
    for r in bravs_results:
        ci50 = r.bravs_ci_50
        ci90 = r.bravs_ci_90
        row: dict[str, object] = {
            "player_name": r.player.player_name,
            "player_id": r.player.player_id,
            "season": r.player.season,
            "team": r.player.team,
            "position": r.player.position,
            "BRAVS": round(r.bravs, 2),
            "BRAVS_ci50_lo": round(ci50[0], 2),
            "BRAVS_ci50_hi": round(ci50[1], 2),
            "BRAVS_ci90_lo": round(ci90[0], 2),
            "BRAVS_ci90_hi": round(ci90[1], 2),
        }
        # Component breakdown in wins.
        for comp_name, comp in r.components.items():
            row[f"comp_{comp_name}_wins"] = round(comp.wins(r.rpw), 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("BRAVS", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"

    if top_n > 0:
        df = df.head(top_n)

    return df
