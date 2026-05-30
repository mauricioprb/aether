"""Key figures: dG histogram, parity, SHAP importance.

Each figure is saved as 300-dpi PNG and PDF under ``data/figures``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline import BaselineResult

logger = logging.getLogger(__name__)

FIG_DIR = Path("data/figures")
_PALETTE = {"train": "#4c72b0", "test": "#dd8452", "bar": "#55a868"}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "savefig.bbox": "tight",
    })


def save_fig(fig: plt.Figure, name: str, fig_dir: Path = FIG_DIR) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"{name}.{ext}", dpi=300)
    logger.info("saved %s.{png,pdf}", name)


def plot_dG_hist(dG: np.ndarray, name: str = "fig3b_dG_hist") -> plt.Figure:
    """Distribution of delta_G_H over [-2, 2] eV."""
    _style()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(dG, bins=60, range=(-2, 2), color=_PALETTE["train"], alpha=0.85,
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel(r"$\Delta G_{\mathrm{H}}$ (eV)")
    ax.set_ylabel("count")
    ax.set_title(f"HER adsorption energies (n={len(dG)})")
    ax.axvline(0, color="0.4", lw=1, ls="--")
    save_fig(fig, name)
    return fig


def plot_parity(result: BaselineResult, name: str = "fig4f_parity") -> plt.Figure:
    """Predicted vs DFT delta_G_H for train and test, with metrics."""
    from training.evaluate import _metrics_text_box, metrics_from_preds

    _style()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(result.y_train, result.y_train_pred, s=10, alpha=0.4,
               color=_PALETTE["train"], label="train", edgecolors="none")
    ax.scatter(result.y_test, result.y_test_pred, s=14, alpha=0.7,
               color=_PALETTE["test"], label="test", edgecolors="none")

    lo = min(result.y_train.min(), result.y_test.min())
    hi = max(result.y_train.max(), result.y_test.max())
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=1, ls="--")

    m = metrics_from_preds(result.y_test, result.y_test_pred)
    ax.text(0.05, 0.95, _metrics_text_box(m), transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox={"boxstyle": "round", "fc": "white", "ec": "0.8"})

    ax.set_xlabel(r"DFT $\Delta G_{\mathrm{H}}$ (eV)")
    ax.set_ylabel(r"predicted $\Delta G_{\mathrm{H}}$ (eV)")
    ax.set_title("Extra Trees parity (train + test)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_aspect("equal", "box")
    save_fig(fig, name)
    return fig


def plot_shap_bar(result: BaselineResult, name: str = "fig6d_shap",
                  title: str = "Feature importance (mean |SHAP|)",
                  max_samples: int = 2000) -> plt.Figure:
    """Mean(|SHAP value|) bar chart of feature importance."""
    import shap

    _style()
    X = result.X_test
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)
    explainer = shap.TreeExplainer(result.model)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)

    order = np.argsort(mean_abs)
    feats = np.array(result.features)[order]
    vals = mean_abs[order]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(feats, vals, color=_PALETTE["bar"])
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(title)
    save_fig(fig, name)
    return fig


def importance_frame(result: BaselineResult) -> pd.DataFrame:
    """Impurity-based feature importances as a sorted DataFrame."""
    imp = result.model.feature_importances_
    return (pd.DataFrame({"feature": result.features, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))
