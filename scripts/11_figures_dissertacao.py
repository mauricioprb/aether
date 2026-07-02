"""Generate dissertation-quality figures from the 4-model whitelist.

Usage: uv run python scripts/11_figures_dissertacao.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from analysis.comparison import (
    WHITELIST,
    load_whitelist,
    plot_cumulative_error,
    plot_mae_bar,
    plot_parity_grid,
    plot_residual_overlay,
)
from plot_style import L, apply_abnt_style, save_fig

apply_abnt_style()

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def fig1_parity_four() -> None:
    """4-panel parity plot: ETR handcrafted | SchNet | MACE Stage A | ETR + MACE emb."""
    runs = load_whitelist()
    if not runs:
        print("Skipping fig1: no runs available")
        return
    fig = plot_parity_grid(runs)
    save_fig(fig, "fig1_parity_four_panels", FIGURES)
    print("Salvo fig1_parity_four_panels")
    plt.close(fig)


def fig2_mae_bar() -> None:
    """Horizontal bar chart - MAE (meV) for all 4 models."""
    runs = load_whitelist()
    if not runs:
        print("Skipping fig2: no runs available")
        return
    fig = plot_mae_bar(runs)
    save_fig(fig, "fig2_mae_bar_meV", FIGURES)
    print("Salvo fig2_mae_bar_meV")
    plt.close(fig)


def fig3_cumulative_error() -> None:
    """Cumulative error curves - all 4 models overlaid."""
    runs = load_whitelist()
    if not runs:
        print("Skipping fig3: no runs available")
        return
    fig = plot_cumulative_error(runs)
    save_fig(fig, "fig3_cumulative_error", FIGURES)
    print("Salvo fig3_cumulative_error")
    plt.close(fig)


def fig4_residual_overlay() -> None:
    """Residual histogram - all 4 models overlaid."""
    runs = load_whitelist()
    if not runs:
        print("Skipping fig4: no runs available")
        return
    fig = plot_residual_overlay(runs)
    save_fig(fig, "fig4_residual_overlay", FIGURES)
    print("Salvo fig4_residual_overlay")
    plt.close(fig)


def fig5_dg_distribution() -> None:
    """Histogram of ΔG_H values in the dataset (independent of model runs)."""
    from ase.io import Trajectory
    traj_path = Path("data/processed/her_dataset.traj")
    if not traj_path.exists():
        print(f"Skipping fig5: {traj_path} missing")
        return
    traj = Trajectory(str(traj_path))
    dgs = [a.info["delta_G_H"] for a in traj]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dgs, bins=50, color="#6b8ba4", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="0.3", ls="--", lw=1.5)
    ax.set_xlabel(L["dg"])
    ax.set_ylabel(L["contagem"])
    ax.set_title(f"Distribuição de $\\Delta G_{{\\mathrm{{H}}}}$ no conjunto de dados\n"
                 f"N = {len(dgs)} estruturas")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "fig5_dg_distribution", FIGURES)
    print("Salvo fig5_dg_distribution")
    plt.close(fig)


def main():
    print(f"Generating dissertation figures (whitelist: {[w[0] for w in WHITELIST]})")
    fig1_parity_four()
    fig2_mae_bar()
    fig3_cumulative_error()
    fig4_residual_overlay()
    fig5_dg_distribution()
    print("Done! All figures in results/figures/")


if __name__ == "__main__":
    main()
