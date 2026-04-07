"""Player card visualizations showing BRAVS component decomposition."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import seaborn as sns

if TYPE_CHECKING:
    from baseball_metric.core.types import BRAVSResult


def plot_player_card(result: BRAVSResult, output_dir: str = "output/") -> str:
    """Horizontal bar chart of component decomposition for a single player.

    Components are sorted by absolute value.  Positive bars are green,
    negative bars are red.  The title includes the player name, season,
    and total BRAVS with its 90 % credible interval.  A text box in the
    lower-right corner shows key metadata (RPW, leverage multiplier,
    total runs).

    Parameters
    ----------
    result : BRAVSResult
        Computed BRAVS result for a single player-season.
    output_dir : str
        Directory where PNG and SVG files are saved.

    Returns
    -------
    str
        Base file path (without extension) of the saved figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    components = result.components
    sorted_comps = sorted(components.values(), key=lambda c: abs(c.runs_mean))

    names = [c.name for c in sorted_comps]
    values = [c.runs_mean for c in sorted_comps]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.45)))
    ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)

    # Error bars from 90 % CI on each component (in runs)
    ci_lo = [c.runs_mean - c.ci_90[0] for c in sorted_comps]
    ci_hi = [c.ci_90[1] - c.runs_mean for c in sorted_comps]
    ax.errorbar(
        values,
        names,
        xerr=[ci_lo, ci_hi],
        fmt="none",
        ecolor="grey",
        elinewidth=1,
        capsize=3,
    )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Runs Above FAT", fontsize=12)
    ax.set_ylabel("")

    ci90 = result.bravs_ci_90
    ax.set_title(
        f"{result.player.player_name} ({result.player.season})  —  "
        f"BRAVS {result.bravs:+.1f} wins  "
        f"[90% CI: {ci90[0]:+.1f}, {ci90[1]:+.1f}]",
        fontsize=13,
        fontweight="bold",
    )

    # Metadata text box
    meta_text = (
        f"Position: {result.player.position}\n"
        f"Total Runs: {result.total_runs_mean:+.1f}\n"
        f"RPW: {result.rpw:.2f}\n"
        f"Leverage: {result.leverage_multiplier:.3f}"
    )
    props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5)
    ax.text(
        0.98,
        0.02,
        meta_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )

    fig.tight_layout()
    safe_name = result.player.player_name.replace(" ", "_").replace("/", "-")
    base = os.path.join(output_dir, f"player_card_{safe_name}_{result.player.season}")
    fig.savefig(f"{base}.png", dpi=150)
    fig.savefig(f"{base}.svg")
    plt.close(fig)
    return base


def plot_multiple_cards(
    results: list[BRAVSResult],
    output_dir: str = "output/",
) -> str:
    """Grid of player cards for several players.

    Arranges mini horizontal bar charts in a grid (up to 3 columns).

    Parameters
    ----------
    results : list[BRAVSResult]
        List of BRAVS results to visualize.
    output_dir : str
        Directory where PNG and SVG files are saved.

    Returns
    -------
    str
        Base file path (without extension) of the saved figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    n = len(results)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for idx, result in enumerate(results):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        components = result.components
        sorted_comps = sorted(components.values(), key=lambda comp: abs(comp.runs_mean))
        names = [comp.name for comp in sorted_comps]
        values = [comp.runs_mean for comp in sorted_comps]
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

        ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"{result.player.player_name} ({result.player.season})\n"
            f"BRAVS {result.bravs:+.1f} wins",
            fontsize=10,
            fontweight="bold",
        )
        ax.tick_params(labelsize=8)

    # Hide empty subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    base = os.path.join(output_dir, "player_cards_grid")
    fig.savefig(f"{base}.png", dpi=150)
    fig.savefig(f"{base}.svg")
    plt.close(fig)
    return base
