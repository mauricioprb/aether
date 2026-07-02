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
from plot_style import SABATIER_DE_OPT_EV, L, apply_abnt_style, save_fig

logger = logging.getLogger(__name__)

FIG_DIR = Path("data/figures")
_PALETTE = {"train": "#4c72b0", "test": "#dd8452", "bar": "#55a868"}


def plot_dG_hist(dG: np.ndarray, name: str = "fig3b_dG_hist") -> plt.Figure:
    """Distribuição do alvo ΔE_H em [-2, 2] eV, com o ótimo de Sabatier."""
    apply_abnt_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(dG, bins=60, range=(-2, 2), color=_PALETTE["train"], alpha=0.85,
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel(L["dg"])
    ax.set_ylabel(L["contagem"])
    ax.set_title(f"Energias de adsorção da HER (n={len(dG)})")
    ax.axvline(SABATIER_DE_OPT_EV, color="0.2", lw=1.2, ls="--",
               label=L["otimo_sabatier"])
    ax.legend(frameon=False, fontsize=9)
    save_fig(fig, name, FIG_DIR)
    return fig


def plot_parity(result: BaselineResult, name: str = "fig4f_parity") -> plt.Figure:
    """ΔG_H predito vs. DFT para treino e teste, com métricas."""
    from training.evaluate import _metrics_text_box, metrics_from_preds

    apply_abnt_style()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(result.y_train, result.y_train_pred, s=10, alpha=0.4,
               color=_PALETTE["train"], label=L["treino"], edgecolors="none")
    ax.scatter(result.y_test, result.y_test_pred, s=14, alpha=0.7,
               color=_PALETTE["test"], label=L["teste"], edgecolors="none")

    lo = min(result.y_train.min(), result.y_test.min())
    hi = max(result.y_train.max(), result.y_test.max())
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=1, ls="--")

    m = metrics_from_preds(result.y_test, result.y_test_pred)
    ax.text(0.05, 0.95, _metrics_text_box(m), transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox={"boxstyle": "round", "fc": "white", "ec": "0.8"})

    ax.set_xlabel(L["dg_dft"])
    ax.set_ylabel(L["dg_pred"])
    ax.set_title("Paridade Extra Trees (treino + teste)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_aspect("equal", "box")
    save_fig(fig, name, FIG_DIR)
    return fig


def plot_shap_bar(result: BaselineResult, name: str = "fig6d_shap",
                  title: str = "Importância das variáveis (média $|$SHAP$|$)",
                  max_samples: int = 2000) -> plt.Figure:
    """Gráfico de barras da importância por média(|valor SHAP|)."""
    import shap

    apply_abnt_style()
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
    ax.set_xlabel(L["importancia_shap"])
    ax.set_title(title)
    save_fig(fig, name, FIG_DIR)
    return fig


def importance_frame(result: BaselineResult) -> pd.DataFrame:
    """Impurity-based feature importances as a sorted DataFrame."""
    imp = result.model.feature_importances_
    return (pd.DataFrame({"feature": result.features, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))
