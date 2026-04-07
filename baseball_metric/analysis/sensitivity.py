"""Sensitivity analysis for the BRAVS framework.

Systematically perturbs model parameters to measure how sensitive the
total BRAVS output is to each input assumption.  This helps identify
which calibration choices matter most and where additional empirical
work would reduce uncertainty.
"""

from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np

import baseball_metric.utils.constants as constants
from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import BRAVSResult, PlayerSeason

# Default perturbation levels expressed as multiplicative factors.
DEFAULT_PERTURBATIONS: list[float] = [-0.50, -0.25, -0.10, 0.10, 0.25, 0.50]

# Parameters to perturb, grouped by category.
# Each entry is (constant_name, human_readable_label).
WOBA_WEIGHT_PARAMS: list[tuple[str, str]] = [
    ("WOBA_WEIGHT_BB", "wOBA weight: BB"),
    ("WOBA_WEIGHT_HBP", "wOBA weight: HBP"),
    ("WOBA_WEIGHT_1B", "wOBA weight: 1B"),
    ("WOBA_WEIGHT_2B", "wOBA weight: 2B"),
    ("WOBA_WEIGHT_3B", "wOBA weight: 3B"),
    ("WOBA_WEIGHT_HR", "wOBA weight: HR"),
    ("WOBA_SCALE", "wOBA scale factor"),
]

FIP_PARAMS: list[tuple[str, str]] = [
    ("FIP_HR_COEFF", "FIP coefficient: HR"),
    ("FIP_BB_COEFF", "FIP coefficient: BB"),
    ("FIP_K_COEFF", "FIP coefficient: K"),
]

POSITIONAL_PARAMS: list[tuple[str, str]] = [
    # Handled specially -- the POS_ADJ dict is perturbed as a unit.
]

FAT_PARAMS: list[tuple[str, str]] = [
    ("FAT_BATTING_RUNS_PER_600PA", "FAT baseline: batting"),
    ("FAT_PITCHING_RUNS_PER_200IP", "FAT baseline: pitching"),
]

LEVERAGE_PARAMS: list[tuple[str, str]] = [
    ("LEVERAGE_DAMPING_EXPONENT", "Leverage damping exponent"),
]

FIELDING_WEIGHT_PARAMS: list[tuple[str, str]] = [
    # Handled specially -- the DEFAULT_FIELDING_WEIGHTS dict is perturbed.
]

PRIOR_SD_PARAMS: list[tuple[str, str]] = [
    ("PRIOR_WOBA_SD", "Prior SD: wOBA"),
    ("PRIOR_FIP_SD", "Prior SD: FIP"),
    ("PRIOR_FIELDING_SD", "Prior SD: fielding"),
    ("PRIOR_BASERUNNING_SD", "Prior SD: baserunning"),
    ("PRIOR_FRAMING_SD", "Prior SD: catcher framing"),
    ("PRIOR_AQI_SD", "Prior SD: approach quality"),
]

ALL_SCALAR_PARAMS: list[tuple[str, str]] = (
    WOBA_WEIGHT_PARAMS
    + FIP_PARAMS
    + FAT_PARAMS
    + LEVERAGE_PARAMS
    + PRIOR_SD_PARAMS
)


def _perturb_scalar(
    const_name: str,
    factor: float,
) -> float:
    """Apply a multiplicative perturbation and return the original value."""
    original: float = getattr(constants, const_name)
    setattr(constants, const_name, original * (1.0 + factor))
    return original


def _restore_scalar(const_name: str, original: float) -> None:
    """Restore a scalar constant to its original value."""
    setattr(constants, const_name, original)


def _perturb_pos_adj(factor: float) -> dict[str, float]:
    """Perturb all positional adjustments by *factor* and return originals."""
    originals = dict(constants.POS_ADJ)
    constants.POS_ADJ = {
        pos: val * (1.0 + factor) for pos, val in originals.items()
    }
    return originals


def _restore_pos_adj(originals: dict[str, float]) -> None:
    constants.POS_ADJ = originals


def _perturb_fielding_weights(factor: float) -> dict[str, float]:
    """Perturb all fielding ensemble weights by *factor* and renormalize."""
    originals = dict(constants.DEFAULT_FIELDING_WEIGHTS)
    raw = {k: v * (1.0 + factor) for k, v in originals.items()}
    total = sum(raw.values())
    constants.DEFAULT_FIELDING_WEIGHTS = {
        k: v / total for k, v in raw.items()
    }
    return originals


def _restore_fielding_weights(originals: dict[str, float]) -> None:
    constants.DEFAULT_FIELDING_WEIGHTS = originals


def run_sensitivity_analysis(
    player: PlayerSeason,
    parameters: Sequence[tuple[str, str]] | None = None,
    perturbations: Sequence[float] | None = None,
    n_samples: int = 4000,
    seed: int = 42,
) -> dict[str, dict[float, float]]:
    """Run a one-at-a-time sensitivity analysis on BRAVS.

    For each parameter, the constant is temporarily perturbed by each
    level in *perturbations*, a fresh BRAVS is computed, and the change
    in total BRAVS (wins) relative to the baseline is recorded.

    Args:
        player: A fully populated ``PlayerSeason``.
        parameters: List of ``(constant_name, label)`` tuples to perturb.
            Defaults to ``ALL_SCALAR_PARAMS`` plus positional and fielding
            weight groups.
        perturbations: Multiplicative perturbation levels.  Defaults to
            ``[-0.50, -0.25, -0.10, +0.10, +0.25, +0.50]``.
        n_samples: Posterior samples per BRAVS computation (lower = faster).
        seed: Random seed for reproducibility.

    Returns:
        A nested dict: ``{parameter_label: {perturbation_level: bravs_delta}}``.
    """
    if parameters is None:
        parameters = ALL_SCALAR_PARAMS
    if perturbations is None:
        perturbations = DEFAULT_PERTURBATIONS

    # Compute the unperturbed baseline.
    baseline: BRAVSResult = compute_bravs(player, n_samples=n_samples, seed=seed)
    baseline_bravs: float = baseline.bravs

    results: dict[str, dict[float, float]] = {}

    # --- Scalar parameters ---
    for const_name, label in parameters:
        results[label] = {}
        for pct in perturbations:
            original = _perturb_scalar(const_name, pct)
            try:
                perturbed: BRAVSResult = compute_bravs(
                    player, n_samples=n_samples, seed=seed
                )
                results[label][pct] = perturbed.bravs - baseline_bravs
            finally:
                _restore_scalar(const_name, original)

    # --- Positional adjustments (as a group) ---
    results["Positional adjustments (all)"] = {}
    for pct in perturbations:
        originals = _perturb_pos_adj(pct)
        try:
            perturbed = compute_bravs(player, n_samples=n_samples, seed=seed)
            results["Positional adjustments (all)"][pct] = (
                perturbed.bravs - baseline_bravs
            )
        finally:
            _restore_pos_adj(originals)

    # --- Fielding ensemble weights ---
    results["Fielding ensemble weights"] = {}
    for pct in perturbations:
        originals_fw = _perturb_fielding_weights(pct)
        try:
            perturbed = compute_bravs(player, n_samples=n_samples, seed=seed)
            results["Fielding ensemble weights"][pct] = (
                perturbed.bravs - baseline_bravs
            )
        finally:
            _restore_fielding_weights(originals_fw)

    return results


def summarize_sensitivity(
    sensitivity_results: dict[str, dict[float, float]],
) -> dict[str, float]:
    """Summarize sensitivity results as max absolute BRAVS change per parameter.

    Args:
        sensitivity_results: Output of :func:`run_sensitivity_analysis`.

    Returns:
        Dict mapping parameter label to the maximum absolute BRAVS change
        observed across all perturbation levels, sorted descending.
    """
    summary: dict[str, float] = {}
    for label, deltas in sensitivity_results.items():
        summary[label] = max(abs(d) for d in deltas.values()) if deltas else 0.0
    return dict(sorted(summary.items(), key=lambda kv: kv[1], reverse=True))
