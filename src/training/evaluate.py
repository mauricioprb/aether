from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CHEM_ACCURACY_EV = 0.043  # ~1 kcal/mol; standard chemical accuracy threshold


def metrics_from_preds(y_true, y_pred) -> dict[str, float]:
    """Unified regression metric pack used by every model in the project.

    Keys:
      r2, mae, rmse                 — base regression metrics
      mdae, max_err                 — robust + worst-case
      pearson_r, spearman_rho       — correlation (Spearman = ranking quality)
      smape                         — symmetric MAPE (safe when target crosses zero)
      mae_meV                       — MAE in meV (literature comparison)
      frac_chem_acc                 — fraction of preds with |error| < 0.043 eV
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    abs_err = np.abs(err)
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))

    # Symmetric MAPE — stable for energies that cross zero. Bounded in [0, 2].
    smape_denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-12)
    smape = float(np.mean(2.0 * abs_err / smape_denom))

    pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    spearman_rho = float(
        pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    )

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "mdae": float(np.median(abs_err)),
        "max_err": float(abs_err.max()),
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "smape": smape,
        "mae_meV": mae * 1000.0,
        "frac_chem_acc": float((abs_err < CHEM_ACCURACY_EV).mean()),
    }


def _metrics_text_box(m: dict[str, float]) -> str:
    return (
        f"$R^2$ = {m['r2']:.3f}\n"
        f"MAE = {m['mae']:.3f} eV ({m['mae_meV']:.0f} meV)\n"
        f"RMSE = {m['rmse']:.3f} eV\n"
        f"MDAE = {m['mdae']:.3f} eV\n"
        f"$\\rho_s$ = {m['spearman_rho']:.3f}\n"
        f"% < 43 meV = {100 * m['frac_chem_acc']:.1f}%"
    )


def parity_figure(y_true, y_pred, title: str = "parity", color: str = "#4c72b0") -> plt.Figure:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = metrics_from_preds(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=14, alpha=0.55, color=color, edgecolors="none")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=1, ls="--")
    ax.text(0.04, 0.96, _metrics_text_box(m), transform=ax.transAxes, va="top",
            fontsize=9, bbox={"boxstyle": "round", "fc": "white", "ec": "0.8"})
    ax.set_xlabel(r"DFT $\Delta G_{\mathrm{H}}$ (eV)")
    ax.set_ylabel(r"predicted $\Delta G_{\mathrm{H}}$ (eV)")
    ax.set_title(title)
    ax.set_aspect("equal", "box")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def residual_hist_figure(y_true, y_pred, title: str = "residual histogram",
                          color: str = "#4c72b0") -> plt.Figure:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    err = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(err, bins=60, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="0.3", lw=1, ls="--")
    ax.axvline(CHEM_ACCURACY_EV, color="0.5", lw=1, ls=":", label="chemical accuracy")
    ax.axvline(-CHEM_ACCURACY_EV, color="0.5", lw=1, ls=":")
    ax.set_xlabel("residual (eV) = predicted - DFT")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def residual_vs_pred_figure(y_true, y_pred, title: str = "residual vs predicted",
                             color: str = "#4c72b0") -> plt.Figure:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    err = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, err, s=12, alpha=0.5, color=color, edgecolors="none")
    ax.axhline(0, color="0.3", lw=1, ls="--")
    ax.set_xlabel(r"predicted $\Delta G_{\mathrm{H}}$ (eV)")
    ax.set_ylabel("residual (eV)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def cumulative_error_figure(y_true, y_pred, title: str = "cumulative error",
                             color: str = "#4c72b0") -> plt.Figure:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    abs_err = np.sort(np.abs(y_pred - y_true))
    frac = np.arange(1, len(abs_err) + 1) / len(abs_err)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(abs_err, frac, color=color, lw=2)
    ax.axvline(CHEM_ACCURACY_EV, color="0.5", lw=1, ls=":", label="chemical accuracy (43 meV)")
    ax.set_xlabel("|error| threshold (eV)")
    ax.set_ylabel("fraction of test samples")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    return fig


def standard_figures(y_true, y_pred, model_label: str,
                      color: str = "#4c72b0") -> dict[str, plt.Figure]:
    """Generate the 4 diagnostic figures every model should produce."""
    return {
        "parity.png": parity_figure(y_true, y_pred, title=f"{model_label} - parity", color=color),
        "residual_hist.png": residual_hist_figure(y_true, y_pred,
                                                  title=f"{model_label} - residuals", color=color),
        "residual_vs_pred.png": residual_vs_pred_figure(y_true, y_pred,
                                                        title=f"{model_label} - residual vs pred",
                                                        color=color),
        "cumulative_error.png": cumulative_error_figure(y_true, y_pred,
                                                         title=f"{model_label} - cumulative error",
                                                         color=color),
    }
