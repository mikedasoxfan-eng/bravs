"""BRAVS vs WAR comparison and divergence visualizations."""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_bravs_vs_war(
    bravs_values: Sequence[float],
    war_values: Sequence[float],
    player_names: Sequence[str],
    output_dir: str = "output/",
) -> str:
    """Scatter plot comparing BRAVS to WAR with a 45-degree reference line.

    Points where ``|BRAVS - WAR| > 2`` are annotated with the player name.

    Parameters
    ----------
    bravs_values : sequence of float
        BRAVS wins for each player.
    war_values : sequence of float
        Corresponding WAR values.
    player_names : sequence of str
        Player names used for annotating outliers.
    output_dir : str
        Directory where PNG and SVG files are saved.

    Returns
    -------
    str
        Base file path (without extension) of the saved figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    bravs = np.asarray(bravs_values, dtype=float)
    war = np.asarray(war_values, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(war, bravs, alpha=0.6, s=40, color="#3498db", edgecolors="white", linewidth=0.5)

    # 45-degree reference
    lo = min(war.min(), bravs.min()) - 1
    hi = max(war.max(), bravs.max()) + 1
    ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1, label="BRAVS = WAR")

    # Annotate outliers
    diffs = np.abs(bravs - war)
    for i in np.where(diffs > 2)[0]:
        ax.annotate(
            player_names[i],
            (war[i], bravs[i]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color="#e74c3c",
            fontweight="bold",
        )

    ax.set_xlabel("WAR", fontsize=12)
    ax.set_ylabel("BRAVS (Wins)", fontsize=12)
    ax.set_title("BRAVS vs WAR Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    fig.tight_layout()
    base = os.path.join(output_dir, "bravs_vs_war")
    fig.savefig(f"{base}.png", dpi=150)
    fig.savefig(f"{base}.svg")
    plt.close(fig)
    return base


def plot_divergence_analysis(
    divergences: list[dict],
    output_dir: str = "output/",
) -> str:
    """Bar chart of the biggest BRAVS-WAR divergences.

    Parameters
    ----------
    divergences : list of dict
        Each dict should contain at minimum:
        - ``"player_name"`` (str)
        - ``"bravs"`` (float)
        - ``"war"`` (float)
        The function computes ``divergence = bravs - war`` and sorts by
        absolute divergence descending.
    output_dir : str
        Directory where PNG and SVG files are saved.

    Returns
    -------
    str
        Base file path (without extension) of the saved figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    # Compute divergence and sort
    for d in divergences:
        d.setdefault("divergence", d["bravs"] - d["war"])
    sorted_div = sorted(divergences, key=lambda d: abs(d["divergence"]), reverse=True)

    names = [d["player_name"] for d in sorted_div]
    divs = [d["divergence"] for d in sorted_div]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in divs]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))
    ax.barh(names, divs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)

    # Value labels
    for i, (val, name) in enumerate(zip(divs, names)):
        ax.text(
            val + (0.1 if val >= 0 else -0.1),
            i,
            f"{val:+.1f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
        )

    ax.set_xlabel("BRAVS - WAR (Wins)", fontsize=12)
    ax.set_title("Biggest BRAVS vs WAR Divergences", fontsize=14, fontweight="bold")

    fig.tight_layout()
    base = os.path.join(output_dir, "divergence_analysis")
    fig.savefig(f"{base}.png", dpi=150)
    fig.savefig(f"{base}.svg")
    plt.close(fig)
    return base
