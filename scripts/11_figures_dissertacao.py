"""Generate dissertation-quality figures for her-gnn.
Usage: uv run python scripts/11_figures_dissertacao.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
RUNS = RESULTS / "runs"

PALETTE = {
    "etr_handcrafted": "#dd8452",   # laranja
    "mace_stageA": "#4c72b0",        # azul
    "mace_frozen_etr": "#55a868",    # verde
    "schnet": "#c44e52",             # vermelho
    "paper": "#888888",              # cinza
}


def load_predictions(run_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load y_true, y_pred from a run's predictions.parquet."""
    pq_path = RUNS / run_name / "predictions.parquet"
    if pq_path.exists():
        df = pd.read_parquet(pq_path)
        return df["y_true"].values, df["y_pred"].values
    # Fallback: try parities from metrics.json
    raise FileNotFoundError(f"No predictions found for {run_name}")


def load_metrics(run_name: str) -> dict:
    with open(RUNS / run_name / "metrics.json") as f:
        return json.load(f)


# ── Figure 1: Main parity plot (3 panels) ────────────────────────────────────
def fig1_parity_three():
    """Parity plots: ETR handcrafted | MACE Stage A (GNN) | ETR + MACE frozen."""
    runs = [
        ("20260523_220553_etr_baseline", "ETR + 10 features\n(R² = 0.934)", "etr_handcrafted"),
        ("20260524_034820_mace_ft_stageC", None, None),  # placeholder
    ]

    # We need to generate Stage A predictions properly
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: ETR handcrafted
    yt, yp = load_predictions("20260523_220553_etr_baseline")
    _parity(axes[0], yt, yp, "ETR + 10 features handcrafted", PALETTE["etr_handcrafted"])

    # Panel 2: MACE Stage A - evaluate from checkpoint
    yt, yp = _eval_stage_a()
    _parity(axes[1], yt, yp, "MACE-MP-0 fine-tune (Stage A)\nGNN pura", PALETTE["mace_stageA"])

    # Panel 3: ETR + MACE frozen embeddings
    yt, yp = load_predictions("20260524_010118_etr_emb_all")
    _parity(axes[2], yt, yp, "ETR + MACE embeddings 512-dim\n(frozen)", PALETTE["mace_frozen_etr"])

    fig.tight_layout()
    path = FIGURES / "fig1_parity_three_panels.png"
    fig.savefig(path)
    print(f"Saved {path}")


def _eval_stage_a() -> tuple[np.ndarray, np.ndarray]:
    """Quick inference of Stage A checkpoint on test set."""
    import torch
    from torch_geometric.loader import DataLoader
    from data.mace_dataset import MACEDataset, three_way_split_mace
    from models.mace_finetune import LitMACEFineTune

    ckpt = "logs/checkpoints/mace_ft_stageA/epoch=64-step=17160.ckpt"
    model = LitMACEFineTune.load_from_checkpoint(ckpt, mace_model="medium", mace_device="cuda")
    model.eval().cuda()

    splits = three_way_split_mace()
    ds = MACEDataset(ids=splits["test"], z_table=model.z_table, r_max=model.r_max)
    loader = DataLoader(ds, batch_size=16)

    yt, yp = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.cuda()
            pred = model(batch)
            yt.extend(batch.y.view(-1).cpu().tolist())
            yp.extend(pred.cpu().tolist())
    return np.array(yt), np.array(yp)


def _parity(ax, y_true, y_pred, title, color):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    ax.scatter(y_true, y_pred, s=12, alpha=0.5, color=color, edgecolors="none")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=1, ls="--")
    txt = f"R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top",
            bbox={"boxstyle": "round", "fc": "white", "ec": "0.8"})
    ax.set_xlabel(r"DFT $\Delta G_{\mathrm{H}^*}$ (eV)")
    ax.set_ylabel(r"Predito $\Delta G_{\mathrm{H}^*}$ (eV)")
    ax.set_title(title)
    ax.set_aspect("equal", "box")
    ax.spines[["top", "right"]].set_visible(False)


# ── Figure 2: Model comparison bar chart ──────────────────────────────────────
def fig2_comparison_bars():
    """Horizontal bar chart comparing all models by R²."""
    models = [
        ("ETR + MACE 512-dim", 0.961, PALETTE["mace_frozen_etr"]),
        ("MACE Stage A (GNN)", 0.947, PALETTE["mace_stageA"]),
        ("MACE Stage C (full)", 0.940, PALETTE["mace_stageA"]),
        ("ETR handcrafted", 0.934, PALETTE["etr_handcrafted"]),
        ("Wang et al. (2025)", 0.922, PALETTE["paper"]),
        ("SchNet (do zero)", 0.908, PALETTE["schnet"]),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = [m[0] for m in models][::-1]
    values = [m[1] for m in models][::-1]
    colors = [m[2] for m in models][::-1]

    bars = ax.barh(names, values, color=colors, height=0.6)
    ax.axvline(0.922, color="gray", ls="--", lw=1, alpha=0.7)
    ax.text(0.922, len(models) - 0.5, "  Wang et al. (0.922)", fontsize=9, color="gray", va="bottom")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("R² (test)")
    ax.set_xlim(0.89, 0.975)
    ax.set_title("Predição de ΔG_H* - comparação de modelos")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = FIGURES / "fig2_model_comparison.png"
    fig.savefig(path)
    print(f"Saved {path}")


# ── Figure 3: Feature reduction sweep ─────────────────────────────────────────
def fig3_feature_reduction():
    """R² vs number of features for SHAP and PCA strategies."""
    # Data from the sweeps
    shap_data = [(512, 0.961), (100, 0.961), (50, 0.958), (20, 0.952), (10, 0.940)]
    pca_data = [(77, 0.960), (38, 0.957), (26, 0.956)]
    baselines = {"ETR handcrafted (10 feat)": 0.934, "SchNet": 0.908}

    fig, ax = plt.subplots(figsize=(8, 5))

    xs_shap = [d[0] for d in shap_data]
    ys_shap = [d[1] for d in shap_data]
    ax.plot(xs_shap, ys_shap, "o-", color=PALETTE["mace_stageA"], label="Top-K SHAP", markersize=8)

    xs_pca = [d[0] for d in pca_data]
    ys_pca = [d[1] for d in pca_data]
    ax.plot(xs_pca, ys_pca, "s--", color=PALETTE["etr_handcrafted"], label="PCA", markersize=8)

    # Baselines
    ax.axhline(0.934, color="gray", ls=":", lw=1, alpha=0.7)
    ax.text(400, 0.934, "ETR handcrafted (0.934)", fontsize=8, color="gray", va="bottom")
    ax.axhline(0.922, color="lightgray", ls=":", lw=1, alpha=0.5)
    ax.text(400, 0.922, "Wang et al. (0.922)", fontsize=8, color="lightgray", va="bottom")

    ax.set_xlabel("Número de features")
    ax.set_ylabel("R² (test)")
    ax.set_title("Redução de dimensionalidade - embeddings MACE")
    ax.set_xscale("log")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0.90, 0.97)

    fig.tight_layout()
    path = FIGURES / "fig3_feature_reduction.png"
    fig.savefig(path)
    print(f"Saved {path}")


# ── Figure 4: Error histogram ─────────────────────────────────────────────────
def fig4_error_histogram():
    """Distribution of prediction errors for Stage A."""
    yt, yp = _eval_stage_a()
    errors = yp - yt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=60, color=PALETTE["mace_stageA"], alpha=0.8, edgecolor="white")
    ax.axvline(0, color="0.3", ls="--", lw=1)
    ax.set_xlabel(r"Erro de predição $\Delta G_{\mathrm{H}^*}$ (eV)")
    ax.set_ylabel("Frequência")
    ax.set_title(f"Distribuição de erros - MACE Stage A\nMAE = {np.abs(errors).mean():.3f} eV")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = FIGURES / "fig4_error_histogram.png"
    fig.savefig(path)
    print(f"Saved {path}")


# ── Figure 5: ΔG_H distribution ───────────────────────────────────────────────
def fig5_dg_distribution():
    """Histogram of ΔG_H values in the dataset."""
    from ase.io import Trajectory
    traj = Trajectory("data/processed/her_dataset.traj")
    dgs = [a.info["delta_G_H"] for a in traj]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dgs, bins=50, color="#6b8ba4", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="0.3", ls="--", lw=1.5)
    ax.set_xlabel(r"$\Delta G_{\mathrm{H}^*}$ (eV)")
    ax.set_ylabel("Frequência")
    ax.set_title(f"Distribuição de ΔG_H* no dataset\nN = {len(dgs)} estruturas")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = FIGURES / "fig5_dg_distribution.png"
    fig.savefig(path)
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Generating dissertation figures...")
    fig1_parity_three()
    fig2_comparison_bars()
    fig3_feature_reduction()
    fig4_error_histogram()
    fig5_dg_distribution()
    print("Done! All figures in results/figures/")


if __name__ == "__main__":
    main()
