"""Uncertainty and posterior distribution visualizations."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from baseball_metric.core.types import BRAVSResult


def plot_posterior(result: BRAVSResult, output_dir: str = "output/") -> str:
    """KDE density plot of the total BRAVS posterior with shaded 90 % CI.

    The posterior mean is marked with a vertical dashed line.

    Parameters
    ----------
    result : BRAVSResult
        Must have ``total_samples`` populated.
    output_dir : str
        Directory where PNG and SVG files are saved.

    Returns
    -------
    str
        Base file path (without extension) of the saved figures.

    Raises
    ------
    ValueError
        If ``total_samples`` is None.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    if result.total_samples is None:
        raise ValueError(
            f"No posterior samples available for {result.player.player_name} "
            f"({result.player.season}).  Cannot plot posterior."
        )

    wins_samples = result.total_samples / result.rpw
    ci90 = result.bravs_ci_90
    mean_val = result.bravs

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(wins_samples, ax=ax, fill=False, color="#2c3e50", linewidth=2, label="Posterior")

    # Shade 90 % CI region
    kde_x = np.linspace(float(wins_samples.min()), float(wins_samples.max()), 500)
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(wins_samples)
    kde_y = kde(kde_x)

    mask = (kde_x >= ci90[0]) & (kde_x <= ci90[1])
    ax.fill_between(kde_x[mask], kde_y[mask], alpha=0.3, color="#3498db", label="90% CI")

    ax.axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")

    ax.set_xlabel("BRAVS (Wins)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Posterior Distribution — {result.player.player_name} ({result.player.season})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)

    fig.tight_layout()
    safe_name = result.player.player_name.replace(" ", "_").replace("/", "-")
    base = os.path.join(output_dir, f"posterior_{safe_name}_{result.player.season}")
    fig.savefig(f"{base}.png", dpi=150)
    fig.savefig(f"{base}.svg")
    plt.close(fig)
    return base


def plot_component_uncertainties(result: BRAVSResult, output_dir: str = "output/") -> str:
    """Violin plots showing the posterior distribution of each component.

    Only components that have posterior samples are displayed.

    Parameters
    ----------
    result : BRAVSResult
        Computed BRAVS result with component samples.
    output_dir : str
        Directory where PNG and SVG files are saved.

    Returns
    -------
    str
        Base file path (without extension) of the saved figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    # Collect components that have samples
    comp_data: list[tuple[str, np.ndarray]] = []
    for name, comp in sorted(result.components.items()):
        if comp.samples is not None and len(comp.samples) > 0:
            comp_data.append((name, np.asarray(comp.samples, dtype=float)))

    if not comp_data:
        # Fall back to point estimates with error bars
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [name for name, _ in sorted(result.components.items(), key=lambda x: abs(x[1].runs_mean))]
        means = [result.components[n].runs_mean for n in names]
        ci_lo = [result.components[n].runs_mean - result.components[n].ci_90[0] for n in names]
        ci_hi = [result.components[n].ci_90[1] - result.components[n].runs_mean for n in names]
        colors = ["#2ecc71" if m >= 0 else "#e74c3c" for m in means]
        ax.barh(names, means, color=colors, edgecolor="white", linewidth=0.5)
        ax.errorbar(means, names, xerr=[ci_lo, ci_hi], fmt="none", ecolor="grey", capsize=3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Runs Above FAT", fontsize=12)
        ax.set_title(
            f"Component Uncertainty (point est.) — {result.player.player_name} ({result.player.season})",
            fontsize=13,
            fontweight="bold",
        )
    else:
        import pandas as pd

        rows = []
        for name, samples in comp_data:
            for s in samples:
                rows.append({"Component": name, "Runs": float(s)})
        df = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(10, max(6, len(comp_data) * 0.6)))
        sns.violinplot(
            data=df,
            y="Component",
            x="Runs",
            ax=ax,
            inner="quartile",
            palette="coolwarm",
            orient="h",
            density_norm="width",
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Runs Above FAT", fontsize=12)
        ax.set_ylabel("")
        ax.set_title(
            f"Component Posterior Distributions — {result.player.player_name} ({result.player.season})",
            fontsize=13,
            fontweight="bold",
        )

    fig.tight_layout()
    safe_name = result.player.player_name.replace(" ", "_").replace("/", "-")
    base = os.path.join(output_dir, f"component_uncertainty_{safe_name}_{result.player.season}")
    fig.savefig(f"{base}.png", dpi=150)
    fig.savefig(f"{base}.svg")
    plt.close(fig)
    return base


def plot_comparison_posteriors(
    results: list[BRAVSResult],
    output_dir: str = "output/",
) -> str:
    """Overlaid posterior density curves for comparing multiple players.

    Each player's total BRAVS posterior is drawn as a separate KDE on the
    same axes, using a distinct color from the seaborn palette.

    Parameters
    ----------
    results : list[BRAVSResult]
        Each should have ``total_samples`` populated.
    output_dir : str
        Directory where PNG and SVG files are saved.

    Returns
    -------
    str
        Base file path (without extension) of the saved figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    palette = sns.color_palette("husl", n_colors=len(results))

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, result in enumerate(results):
        label = f"{result.player.player_name} ({result.player.season})"
        if result.total_samples is not None:
            wins_samples = result.total_samples / result.rpw
            sns.kdeplot(
                wins_samples,
                ax=ax,
                fill=True,
                alpha=0.25,
                color=palette[idx],
                linewidth=2,
                label=label,
            )
        else:
            # Fall back: draw a vertical line at the point estimate
            ax.axvline(
                result.bravs,
                color=palette[idx],
                linewidth=2,
                linestyle="--",
                label=f"{label} (point est.)",
            )

    ax.set_xlabel("BRAVS (Wins)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Posterior Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="best")

    fig.tight_layout()
    base = os.path.join(output_dir, "posterior_comparison")
    fig.savefig(f"{base}.png", dpi=150)
    fig.savefig(f"{base}.svg")
    plt.close(fig)
    return base
