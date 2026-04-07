"""Leaderboard visualization for BRAVS rankings."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from baseball_metric.core.types import BRAVSResult


def plot_leaderboard(
    results: list[BRAVSResult],
    top_n: int = 25,
    output_dir: str = "output/",
) -> str:
    """Horizontal bar chart of the top-N players ranked by BRAVS.

    Each bar includes error bars showing the 90 % credible interval.
    Bars are labeled with the player name and season.

    Parameters
    ----------
    results : list[BRAVSResult]
        Full list of BRAVS results.  Will be sorted internally.
    top_n : int
        Number of players to display.
    output_dir : str
        Directory where PNG and SVG files are saved.

    Returns
    -------
    str
        Base file path (without extension) of the saved figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    # Sort descending by BRAVS and take top N
    ranked = sorted(results, key=lambda r: r.bravs, reverse=True)[:top_n]
    # Reverse so the highest value appears at the top of the horizontal chart
    ranked = list(reversed(ranked))

    labels = [f"{r.player.player_name} ({r.player.season})" for r in ranked]
    bravs_vals = np.array([r.bravs for r in ranked])

    ci_lo = np.array([r.bravs - r.bravs_ci_90[0] for r in ranked])
    ci_hi = np.array([r.bravs_ci_90[1] - r.bravs for r in ranked])

    # Color gradient by value
    palette = sns.color_palette("viridis", n_colors=len(ranked))

    fig, ax = plt.subplots(figsize=(10, max(6, len(ranked) * 0.35)))
    bars = ax.barh(
        labels,
        bravs_vals,
        color=palette,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.errorbar(
        bravs_vals,
        labels,
        xerr=[ci_lo, ci_hi],
        fmt="none",
        ecolor="grey",
        elinewidth=1,
        capsize=3,
    )

    # Value annotation at the end of each bar
    for bar, val in zip(bars, bravs_vals):
        ax.text(
            val + 0.15 if val >= 0 else val - 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
            color="black",
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("BRAVS (Wins Above FAT)", fontsize=12)
    ax.set_title(f"BRAVS Leaderboard — Top {len(ranked)}", fontsize=14, fontweight="bold")

    fig.tight_layout()
    base = os.path.join(output_dir, f"leaderboard_top{len(ranked)}")
    fig.savefig(f"{base}.png", dpi=150)
    fig.savefig(f"{base}.svg")
    plt.close(fig)
    return base
