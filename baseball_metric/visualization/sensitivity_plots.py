"""Sensitivity analysis heatmap for BRAVS parameter perturbations."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_sensitivity_heatmap(
    sensitivity_data: dict,
    output_dir: str = "output/",
) -> str:
    """Heatmap showing how BRAVS changes under parameter perturbations.

    Parameters
    ----------
    sensitivity_data : dict
        Expected structure::

            {
                "parameters": ["param_a", "param_b", ...],
                "perturbations": [-0.2, -0.1, 0.0, 0.1, 0.2],
                "values": [              # 2-D: rows=params, cols=perturbations
                    [delta_bravs, ...],   # param_a across perturbations
                    [delta_bravs, ...],   # param_b ...
                ],
            }

        ``values[i][j]`` is the change in BRAVS (wins) when parameter *i*
        is perturbed by level *j*.
    output_dir : str
        Directory where PNG and SVG files are saved.

    Returns
    -------
    str
        Base file path (without extension) of the saved figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    parameters: list[str] = sensitivity_data["parameters"]
    perturbations: list[float] = sensitivity_data["perturbations"]
    values = np.asarray(sensitivity_data["values"], dtype=float)

    # Format perturbation labels
    pert_labels = [f"{p:+.0%}" if abs(p) >= 0.01 else "baseline" for p in perturbations]

    fig, ax = plt.subplots(figsize=(max(8, len(perturbations) * 1.2), max(6, len(parameters) * 0.6)))

    vmax = float(np.abs(values).max()) or 1.0
    sns.heatmap(
        values,
        ax=ax,
        xticklabels=pert_labels,
        yticklabels=parameters,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "BRAVS Change (Wins)"},
    )

    ax.set_xlabel("Perturbation Level", fontsize=12)
    ax.set_ylabel("Parameter", fontsize=12)
    ax.set_title("Sensitivity Analysis — BRAVS Response to Parameter Perturbation",
                  fontsize=13, fontweight="bold")

    fig.tight_layout()
    base = os.path.join(output_dir, "sensitivity_heatmap")
    fig.savefig(f"{base}.png", dpi=150)
    fig.savefig(f"{base}.svg")
    plt.close(fig)
    return base
