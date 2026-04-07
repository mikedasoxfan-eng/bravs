"""Systematic bias detection for the BRAVS framework.

Checks whether BRAVS exhibits systematic biases along dimensions such as
fielding position, era, league, handedness, team quality, or park factor.
Biases indicate that the metric is over- or under-valuing certain player
groups relative to an external benchmark (typically fWAR).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from baseball_metric.core.types import BRAVSResult


def _build_analysis_frame(
    bravs_results: Sequence[BRAVSResult],
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine BRAVS results with optional external metadata into one frame.

    Args:
        bravs_results: List of ``BRAVSResult`` objects.
        metadata: Optional DataFrame with columns ``player_id``, ``season``,
            and any of: ``handedness``, ``team_wins``, ``park_factor``,
            ``fWAR``, ``bWAR``, ``league``, ``era_bucket``.

    Returns:
        Merged DataFrame with one row per player-season.
    """
    rows: list[dict[str, Any]] = []
    for r in bravs_results:
        row: dict[str, Any] = {
            "player_id": r.player.player_id,
            "player_name": r.player.player_name,
            "season": r.player.season,
            "position": r.player.position,
            "league": r.player.league,
            "park_factor": r.player.park_factor,
            "BRAVS": r.bravs,
            "total_runs": r.total_runs_mean,
        }
        for comp_name, comp in r.components.items():
            row[f"comp_{comp_name}_runs"] = comp.runs_mean
        rows.append(row)

    df = pd.DataFrame(rows)

    if metadata is not None:
        df = df.merge(
            metadata,
            on=["player_id", "season"],
            how="left",
            suffixes=("", "_meta"),
        )

    return df


def detect_bias(
    bravs_results: Sequence[BRAVSResult],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Regress BRAVS residuals on player/context features to detect bias.

    The residual is defined as ``BRAVS - fWAR`` (if fWAR is available in
    *metadata*) or simply BRAVS centered to zero mean.  Positive
    coefficients indicate that BRAVS systematically over-values players
    in that group relative to the benchmark.

    Continuous predictors are standardized (z-scored) before regression so
    that coefficients are on a comparable scale (effect sizes in wins per
    SD change in the predictor).

    Categorical predictors (position, league, handedness) are one-hot
    encoded with the most common category dropped as the reference level.

    Args:
        bravs_results: List of ``BRAVSResult`` objects.
        metadata: DataFrame with columns ``player_id``, ``season``, and
            any subset of: ``fWAR``, ``handedness`` (``"L"``/``"R"``/``"S"``),
            ``team_wins``, ``park_factor``, ``era_bucket`` (e.g., decade
            string like ``"2010s"``), ``league`` (``"AL"``/``"NL"``).

    Returns:
        DataFrame with one row per predictor and columns:
        - ``predictor``: name of the predictor
        - ``coefficient``: OLS regression coefficient
        - ``std_error``: standard error of the coefficient
        - ``t_statistic``: t-statistic
        - ``p_value``: two-sided p-value
        - ``effect_size_wins``: coefficient (same as ``coefficient`` for
          standardized predictors; represents wins of bias per 1-SD change)
    """
    df = _build_analysis_frame(bravs_results, metadata)

    # Compute residual.
    if "fWAR" in df.columns and df["fWAR"].notna().sum() > 10:
        df["residual"] = df["BRAVS"] - df["fWAR"]
    else:
        df["residual"] = df["BRAVS"] - df["BRAVS"].mean()

    # --- Build predictor matrix ---
    predictor_cols: list[str] = []

    # Continuous predictors.
    continuous = ["park_factor", "team_wins"]
    for col in continuous:
        if col in df.columns and df[col].notna().sum() > 10:
            mean = df[col].mean()
            sd = df[col].std()
            if sd > 0:
                df[f"{col}_z"] = (df[col] - mean) / sd
                predictor_cols.append(f"{col}_z")

    # Season as continuous (linear era trend).
    if "season" in df.columns:
        season_mean = df["season"].mean()
        season_sd = df["season"].std()
        if season_sd > 0:
            df["season_z"] = (df["season"] - season_mean) / season_sd
            predictor_cols.append("season_z")

    # Categorical predictors -- one-hot encode.
    categorical = ["position", "league", "handedness", "era_bucket"]
    for col in categorical:
        if col not in df.columns or df[col].notna().sum() < 10:
            continue
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
        df = pd.concat([df, dummies], axis=1)
        predictor_cols.extend(dummies.columns.tolist())

    if not predictor_cols:
        return pd.DataFrame(
            columns=["predictor", "coefficient", "std_error", "t_statistic",
                      "p_value", "effect_size_wins"]
        )

    # Drop rows with missing residual or predictors.
    analysis_cols = ["residual"] + predictor_cols
    df_clean = df[analysis_cols].dropna()

    if len(df_clean) < len(predictor_cols) + 2:
        return pd.DataFrame(
            columns=["predictor", "coefficient", "std_error", "t_statistic",
                      "p_value", "effect_size_wins"]
        )

    # --- OLS regression via numpy ---
    y = df_clean["residual"].to_numpy(dtype=np.float64)
    X = df_clean[predictor_cols].to_numpy(dtype=np.float64)
    # Add intercept.
    X = np.column_stack([np.ones(len(X)), X])
    all_names = ["intercept"] + predictor_cols

    # Solve via least squares.
    beta, residuals_ss, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    n, p = X.shape
    dof = n - p
    if dof <= 0:
        dof = 1
    mse = float(np.sum(resid**2)) / dof
    cov_beta = mse * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov_beta))

    result_rows: list[dict[str, Any]] = []
    for i, name in enumerate(all_names):
        if name == "intercept":
            continue  # Skip intercept from output.
        t_stat = beta[i] / se[i] if se[i] > 0 else 0.0
        p_val = float(2.0 * stats.t.sf(abs(t_stat), dof))
        result_rows.append({
            "predictor": name,
            "coefficient": round(float(beta[i]), 5),
            "std_error": round(float(se[i]), 5),
            "t_statistic": round(float(t_stat), 3),
            "p_value": round(p_val, 6),
            "effect_size_wins": round(float(beta[i]), 4),
        })

    result_df = pd.DataFrame(result_rows)
    result_df = result_df.sort_values("p_value").reset_index(drop=True)
    return result_df


def positional_bias_check(
    bravs_results: Sequence[BRAVSResult],
) -> pd.DataFrame:
    """Specific check for systematic positional bias in BRAVS.

    Groups player-seasons by primary position and computes summary
    statistics of the BRAVS distribution at each position.  Also runs
    a one-way ANOVA to test whether mean BRAVS differs significantly
    across positions (beyond what positional adjustments should account
    for).

    Args:
        bravs_results: List of ``BRAVSResult`` objects.

    Returns:
        DataFrame with one row per position and columns:
        - ``position``: the fielding position
        - ``n``: count of player-seasons
        - ``mean_bravs``: mean BRAVS at that position
        - ``median_bravs``: median BRAVS
        - ``sd_bravs``: standard deviation of BRAVS
        - ``mean_positional_adj_runs``: mean positional adjustment component
        - ``anova_f``: F-statistic from one-way ANOVA across all positions
        - ``anova_p``: p-value from the ANOVA test
    """
    records: list[dict[str, Any]] = []
    for r in bravs_results:
        pos_runs = (
            r.components["positional"].runs_mean
            if "positional" in r.components
            else 0.0
        )
        records.append({
            "position": r.player.position,
            "BRAVS": r.bravs,
            "positional_runs": pos_runs,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(
            columns=["position", "n", "mean_bravs", "median_bravs", "sd_bravs",
                      "mean_positional_adj_runs", "anova_f", "anova_p"]
        )

    # Per-position summary.
    grouped = df.groupby("position")
    summary = grouped.agg(
        n=("BRAVS", "count"),
        mean_bravs=("BRAVS", "mean"),
        median_bravs=("BRAVS", "median"),
        sd_bravs=("BRAVS", "std"),
        mean_positional_adj_runs=("positional_runs", "mean"),
    ).reset_index()

    # One-way ANOVA across positions (requires >= 2 groups with >= 2 obs).
    groups = [g["BRAVS"].to_numpy() for _, g in grouped if len(g) >= 2]
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        summary["anova_f"] = round(float(f_stat), 3)
        summary["anova_p"] = round(float(p_val), 6)
    else:
        summary["anova_f"] = np.nan
        summary["anova_p"] = np.nan

    # Round for readability.
    for col in ("mean_bravs", "median_bravs", "sd_bravs", "mean_positional_adj_runs"):
        summary[col] = summary[col].round(3)

    summary = summary.sort_values("mean_bravs", ascending=False).reset_index(drop=True)
    return summary
