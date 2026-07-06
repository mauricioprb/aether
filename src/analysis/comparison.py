"""Cross-model comparison helpers (whitelist of dissertation models).

Used by ``scripts/06_compare.py`` and ``scripts/11_figures_dissertacao.py`` so
both scripts speak about the same set of models with the same plots.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import L, MODEL_COLORS, apply_abnt_style, usar_virgula

logger = logging.getLogger(__name__)

CHEM_ACCURACY_EV = 0.043
RESULTS = Path("results")

# (display_name, exact_run_name, color, multiseed_group_or_None)
# For multi-seed models, pick the seed whose R²_test is closest to the
# multi-seed mean (representative of typical performance, not best/worst).
# multiseed_group is the prefix used by aggregate_groups to compute mean ± std.
#   SchNet 5 seeds mean = 0.9105 -> seed=2 (R²=0.8991) most representative
#   Stage A 5 seeds mean = 0.9564 -> seed=3 (R²=0.9562) most representative
WHITELIST: list[tuple[str, str, str, str | None]] = [
    ("ETR + 10 descritores",  "etr_baseline",            MODEL_COLORS["etr_baseline"], None),
    ("SchNet (do zero)",      "schnet_v2_seed2",         MODEL_COLORS["schnet"],       "schnet_v2"),
    ("MACE Estágio A (GNN)",  "mace_ft_stageA_v2_seed3", MODEL_COLORS["mace_head"],    "mace_ft_stageA_v2"),
    ("ETR + emb. MACE 512",   "etr_emb_all",             MODEL_COLORS["etr_emb"],      None),
]

METRIC_COLS: list[tuple[str, str]] = [
    ("r2_test", "R²"),
    ("mae_test", "MAE (eV)"),
    ("rmse_test", "RMSE (eV)"),
    ("mdae_test", "MDAE (eV)"),
    ("max_err_test", "max err (eV)"),
    ("pearson_r_test", "Pearson r"),
    ("spearman_rho_test", "Spearman ρ"),
    ("mae_meV_test", "MAE (meV)"),
    ("frac_chem_acc_test", "% < 43 meV"),
    ("n_params", "# params"),
    ("elapsed_sec", "tempo (s)"),
]


@dataclass
class ModelRun:
    display: str
    color: str
    entry: dict
    preds: pd.DataFrame
    multiseed_stats: dict | None = None  # {n, r2_mean, r2_std, mae_mean, mae_std}

    @property
    def y_true(self) -> np.ndarray:
        return self.preds["y_true"].to_numpy()

    @property
    def y_pred(self) -> np.ndarray:
        return self.preds["y_pred"].to_numpy()


def latest_match(summary: list[dict], exact_name: str) -> dict | None:
    """Return the most recent run whose ``name`` matches ``exact_name`` exactly."""
    runs = [e for e in summary if e.get("name") == exact_name]
    return max(runs, key=lambda e: e["timestamp"]) if runs else None


def load_run(entry: dict, display: str, color: str,
              multiseed_stats: dict | None = None) -> ModelRun | None:
    pq = Path(entry["run_dir"]) / "predictions.parquet"
    if not pq.exists():
        logger.warning("missing predictions.parquet for %s", display)
        return None
    df = pd.read_parquet(pq)
    test = df[df["split"] == "test"] if "split" in df.columns else df
    return ModelRun(display=display, color=color, entry=entry,
                    preds=test.reset_index(drop=True),
                    multiseed_stats=multiseed_stats)


def _compute_multiseed_stats(summary: list[dict], group_prefix: str) -> dict | None:
    """Compute mean/std of R²/MAE for runs whose name starts with ``group_prefix_seed``."""
    rx = re.compile(rf"^{re.escape(group_prefix)}_seed\d+$")
    siblings = [e for e in summary if rx.match(e.get("name", ""))]
    if not siblings:
        return None
    r2s = [e["r2_test"] for e in siblings if isinstance(e.get("r2_test"), (int, float))]
    maes = [e["mae_test"] for e in siblings if isinstance(e.get("mae_test"), (int, float))]
    if not r2s:
        return None
    return {
        "n": len(r2s),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
        "mae_mean": float(np.mean(maes)) if maes else None,
        "mae_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
    }


def load_whitelist(summary_path: Path = RESULTS / "summary.json") -> list[ModelRun]:
    summary = json.loads(summary_path.read_text())
    runs: list[ModelRun] = []
    for display, exact_name, color, group in WHITELIST:
        entry = latest_match(summary, exact_name)
        if entry is None:
            logger.warning("no summary entry with name == %r", exact_name)
            continue
        ms = _compute_multiseed_stats(summary, group) if group else None
        loaded = load_run(entry, display, color, multiseed_stats=ms)
        if loaded is not None:
            runs.append(loaded)
    return runs


def _fmt(key: str, v) -> str:
    if v is None:
        return "-"
    if not isinstance(v, (int, float)):
        return str(v)
    if key == "frac_chem_acc_test":
        return f"{100 * v:.1f}"
    if key == "mae_meV_test":
        return f"{v:.0f}"
    if key == "n_params":
        return f"{int(v):,}"
    if key == "elapsed_sec":
        return f"{v:.1f}"
    return f"{v:.4f}"


def comparison_table(runs: list[ModelRun]) -> str:
    """Markdown table. For multi-seed models, override r2/mae/rmse cells with
    mean ± std so the table reflects multi-seed truth rather than a single seed."""
    header = "| Modelo | " + " | ".join(label for _, label in METRIC_COLS) + " |\n"
    sep = "|" + "|".join(["---"] * (len(METRIC_COLS) + 1)) + "|\n"
    body = []
    for r in runs:
        cells = [r.display]
        ms = r.multiseed_stats
        for key, _ in METRIC_COLS:
            if ms and key == "r2_test":
                cells.append(f"{ms['r2_mean']:.4f} ± {ms['r2_std']:.4f}")
            elif ms and key == "mae_test":
                cells.append(f"{ms['mae_mean']:.4f} ± {ms['mae_std']:.4f}")
            else:
                cells.append(_fmt(key, r.entry.get(key)))
        body.append("| " + " | ".join(cells) + " |")
    return header + sep + "\n".join(body) + "\n"


def plot_parity_grid(runs: list[ModelRun]) -> plt.Figure:
    apply_abnt_style()
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, runs, strict=True):
        ax.scatter(r.y_true, r.y_pred, s=12, alpha=0.5, color=r.color, edgecolors="none")
        lo = min(r.y_true.min(), r.y_pred.min())
        hi = max(r.y_true.max(), r.y_pred.max())
        ax.plot([lo, hi], [lo, hi], color="0.3", lw=1, ls="--")
        r2 = r.entry.get("r2_test", float("nan"))
        mae = r.entry.get("mae_test", float("nan"))
        def _pt(v: float, nd: int = 3) -> str:
            return f"{v:.{nd}f}".replace(".", ",")

        lines = [f"R² = {_pt(r2)}", f"MAE = {_pt(mae)} eV"]
        if r.multiseed_stats:
            ms = r.multiseed_stats
            lines.append(f"n={ms['n']} seeds:")
            lines.append(f"  R² = {_pt(ms['r2_mean'])} ± {_pt(ms['r2_std'])}")
            lines.append(f"  MAE = {_pt(ms['mae_mean'])} ± {_pt(ms['mae_std'])} eV")
        ax.text(0.05, 0.95, "\n".join(lines),
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"boxstyle": "round", "fc": "white", "ec": "0.8"})
        ax.set_title(r.display)
        ax.set_xlabel(L["dg_dft"])
        ax.set_ylabel(L["dg_pred"])
        ax.set_aspect("equal", "box")
        ax.spines[["top", "right"]].set_visible(False)
        usar_virgula(ax)
    fig.tight_layout()
    return fig


def plot_cumulative_error(runs: list[ModelRun]) -> plt.Figure:
    apply_abnt_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in runs:
        abs_err = np.sort(np.abs(r.y_pred - r.y_true))
        frac = np.arange(1, len(abs_err) + 1) / len(abs_err)
        ax.plot(abs_err, frac, lw=2, color=r.color, label=r.display)
    ax.axvline(CHEM_ACCURACY_EV, color="0.4", lw=1, ls=":",
               label=L["acuracia_quimica"])
    ax.set_xlabel(L["limiar_erro"])
    ax.set_ylabel(L["frac_teste"])
    ax.set_ylim(0, 1.02)
    usar_virgula(ax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    return fig


def plot_residual_overlay(runs: list[ModelRun]) -> plt.Figure:
    apply_abnt_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in runs:
        err = r.y_pred - r.y_true
        ax.hist(err, bins=60, alpha=0.45, color=r.color, label=r.display,
                edgecolor="white", linewidth=0.2)
    ax.axvline(0, color="0.3", lw=1, ls="--")
    ax.set_xlabel(L["residuo"])
    ax.set_ylabel(L["contagem"])
    ax.set_title("Distribuição de erros")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_mae_bar(runs: list[ModelRun]) -> plt.Figure:
    """Horizontal bar of MAE (meV). Multi-seed models use mean ± std with error bars."""
    apply_abnt_style()
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    runs_rev = list(runs)[::-1]
    names = [r.display for r in runs_rev]
    colors = [r.color for r in runs_rev]
    mae_means = []
    mae_stds = []
    labels = []
    for r in runs_rev:
        if r.multiseed_stats:
            ms = r.multiseed_stats
            m = ms["mae_mean"] * 1000.0
            s = ms["mae_std"] * 1000.0
            mae_means.append(m)
            mae_stds.append(s)
            labels.append(f"{m:.0f} ± {s:.0f} meV (n={ms['n']})")
        else:
            v = r.entry.get("mae_meV_test") or 1000 * r.entry.get("mae_test", 0)
            mae_means.append(v)
            mae_stds.append(0.0)
            labels.append(f"{v:.0f} meV")
    bars = ax.barh(names, mae_means, color=colors, height=0.55,
                    xerr=mae_stds, error_kw={"ecolor": "0.3", "capsize": 4})
    ax.axvline(43, color="0.4", ls=":", lw=1, label=L["acuracia_quimica"])
    for bar, lbl, std in zip(bars, labels, mae_stds, strict=True):
        ax.text(bar.get_width() + std + 2, bar.get_y() + bar.get_height() / 2,
                lbl, va="center", fontsize=9)
    ax.set_xlabel(L["mae_meV"])
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
