from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics_from_preds(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def parity_figure(y_true, y_pred, title: str = "parity", color: str = "#dd8452") -> plt.Figure:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = metrics_from_preds(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=14, alpha=0.6, color=color, edgecolors="none")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=1, ls="--")
    txt = (f"$R^2$ = {m['r2']:.3f}\nMAE = {m['mae']:.3f}\n"
           f"MSE = {m['mse']:.3f}\nRMSE = {m['rmse']:.3f}")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top",
            bbox={"boxstyle": "round", "fc": "white", "ec": "0.8"})
    ax.set_xlabel(r"DFT $\Delta G_{\mathrm{H}}$ (eV)")
    ax.set_ylabel(r"predicted $\Delta G_{\mathrm{H}}$ (eV)")
    ax.set_title(title)
    ax.set_aspect("equal", "box")
    ax.spines[["top", "right"]].set_visible(False)
    return fig
