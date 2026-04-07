"""Head-to-head comparison of BRAVS against WAR implementations.

Provides correlation analysis, identification of divergent players, and
component-level variance decomposition to understand where and why BRAVS
diverges from fWAR and bWAR.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from baseball_metric.core.types import BRAVSResult, ComponentResult


def bravs_vs_war_analysis(
    bravs_results: Sequence[BRAVSResult],
    war_data: pd.DataFrame,
) -> dict[str, object]:
    """Comprehensive comparison of BRAVS to WAR.

    Computes correlations (Pearson and Spearman), mean absolute error,
    root mean squared error, and systematic bias for both fWAR and bWAR.

    Args:
        bravs_results: List of ``BRAVSResult`` objects.
        war_data: DataFrame with columns ``player_id``, ``season``,
            ``fWAR``, ``bWAR``.

    Returns:
        Dict with keys:
        - ``n_matched``: number of matched player-seasons
        - ``fWAR_pearson_r``, ``fWAR_pearson_p``: Pearson correlation with fWAR
        - ``fWAR_spearman_r``, ``fWAR_spearman_p``: Spearman rank correlation
        - ``fWAR_mae``: mean absolute error vs fWAR
        - ``fWAR_rmse``: root mean squared error vs fWAR
        - ``fWAR_mean_bias``: mean(BRAVS - fWAR), positive = BRAVS higher
        - ``bWAR_pearson_r``, ``bWAR_pearson_p``, etc.: same for bWAR
        - ``merged_df``: the merged DataFrame used for analysis
    """
    # Build lookup from BRAVS results.
    bravs_lookup: dict[tuple[str, int], float] = {}
    for r in bravs_results:
        bravs_lookup[(r.player.player_id, r.player.season)] = r.bravs

    # Merge on player_id + season.
    war_copy = war_data.copy()
    war_copy["_key"] = list(zip(war_copy["player_id"], war_copy["season"]))
    war_copy["BRAVS"] = war_copy["_key"].map(bravs_lookup)
    merged = war_copy.dropna(subset=["BRAVS"]).copy()
    merged.drop(columns=["_key"], inplace=True)

    n = len(merged)
    output: dict[str, object] = {"n_matched": n, "merged_df": merged}

    if n < 3:
        return output

    bravs_arr = merged["BRAVS"].to_numpy(dtype=np.float64)

    for war_col in ("fWAR", "bWAR"):
        if war_col not in merged.columns:
            continue
        war_arr = merged[war_col].to_numpy(dtype=np.float64)
        valid = ~(np.isnan(bravs_arr) | np.isnan(war_arr))
        b = bravs_arr[valid]
        w = war_arr[valid]

        if len(b) < 3:
            continue

        pr, pp = stats.pearsonr(b, w)
        sr, sp = stats.spearmanr(b, w)
        diff = b - w
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        mean_bias = float(np.mean(diff))

        output[f"{war_col}_pearson_r"] = round(float(pr), 4)
        output[f"{war_col}_pearson_p"] = float(pp)
        output[f"{war_col}_spearman_r"] = round(float(sr), 4)
        output[f"{war_col}_spearman_p"] = float(sp)
        output[f"{war_col}_mae"] = round(mae, 3)
        output[f"{war_col}_rmse"] = round(rmse, 3)
        output[f"{war_col}_mean_bias"] = round(mean_bias, 3)

    return output


def find_divergent_players(
    bravs_results: Sequence[BRAVSResult],
    war_data: pd.DataFrame,
    threshold: float = 2.0,
    war_column: str = "fWAR",
) -> pd.DataFrame:
    """Find players where BRAVS and WAR diverge by more than a threshold.

    Args:
        bravs_results: List of ``BRAVSResult`` objects.
        war_data: DataFrame with ``player_id``, ``season``, and the WAR
            column specified by *war_column*.
        threshold: Minimum absolute difference in wins to flag a player.
        war_column: Which WAR column to compare against (``"fWAR"`` or
            ``"bWAR"``).

    Returns:
        DataFrame of divergent player-seasons with columns:
        - ``player_name``, ``player_id``, ``season``, ``position``
        - ``BRAVS``, WAR column, ``difference``
        - One column per BRAVS component (runs) to help diagnose divergence
        Sorted by absolute difference, descending.
    """
    # Build index from WAR data.
    war_index: dict[tuple[str, int], float] = {}
    for _, row in war_data.iterrows():
        key = (row["player_id"], int(row["season"]))
        if war_column in row and pd.notna(row[war_column]):
            war_index[key] = float(row[war_column])

    rows: list[dict[str, object]] = []
    for r in bravs_results:
        key = (r.player.player_id, r.player.season)
        if key not in war_index:
            continue
        war_val = war_index[key]
        diff = r.bravs - war_val
        if abs(diff) < threshold:
            continue

        row: dict[str, object] = {
            "player_name": r.player.player_name,
            "player_id": r.player.player_id,
            "season": r.player.season,
            "position": r.player.position,
            "BRAVS": round(r.bravs, 2),
            war_column: round(war_val, 2),
            "difference": round(diff, 2),
            "abs_difference": round(abs(diff), 2),
        }
        for comp_name, comp in r.components.items():
            row[f"comp_{comp_name}_runs"] = round(comp.runs_mean, 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("abs_difference", ascending=False).reset_index(drop=True)
        df.drop(columns=["abs_difference"], inplace=True)
    return df


def component_contribution_analysis(
    bravs_results: Sequence[BRAVSResult],
) -> pd.DataFrame:
    """Analyze which BRAVS components contribute the most variance.

    For each component, computes the variance of that component's runs-
    above-FAT contribution across all player-seasons, plus the correlation
    of that component with total BRAVS.  This reveals which components
    drive the most differentiation among players.

    Args:
        bravs_results: List of ``BRAVSResult`` objects.

    Returns:
        DataFrame with columns:
        - ``component``: component name
        - ``mean_runs``: average runs contribution across players
        - ``sd_runs``: standard deviation of runs contribution
        - ``variance_share``: fraction of total BRAVS variance explained
            (approximated as component variance / sum of component variances)
        - ``corr_with_total``: Pearson correlation between component runs
            and total BRAVS
        Sorted by ``variance_share`` descending.
    """
    if not bravs_results:
        return pd.DataFrame(
            columns=["component", "mean_runs", "sd_runs", "variance_share", "corr_with_total"]
        )

    # Collect all component names across all results.
    all_components: set[str] = set()
    for r in bravs_results:
        all_components.update(r.components.keys())

    total_bravs = np.array([r.bravs for r in bravs_results], dtype=np.float64)

    rows: list[dict[str, object]] = []
    component_variances: dict[str, float] = {}

    for comp_name in sorted(all_components):
        values = np.array(
            [
                r.components[comp_name].runs_mean if comp_name in r.components else 0.0
                for r in bravs_results
            ],
            dtype=np.float64,
        )
        comp_var = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
        component_variances[comp_name] = comp_var

        if len(values) >= 3 and np.std(values) > 0:
            corr, _ = stats.pearsonr(values, total_bravs)
        else:
            corr = np.nan

        rows.append({
            "component": comp_name,
            "mean_runs": round(float(np.mean(values)), 2),
            "sd_runs": round(float(np.std(values, ddof=1)), 2) if len(values) > 1 else 0.0,
            "variance_raw": comp_var,
            "corr_with_total": round(float(corr), 4) if not np.isnan(corr) else np.nan,
        })

    total_var = sum(component_variances.values())
    for row in rows:
        row["variance_share"] = (
            round(float(row["variance_raw"]) / total_var, 4) if total_var > 0 else 0.0
        )

    df = pd.DataFrame(rows)
    df = df[["component", "mean_runs", "sd_runs", "variance_share", "corr_with_total"]]
    df = df.sort_values("variance_share", ascending=False).reset_index(drop=True)
    return df
