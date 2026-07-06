"""Figuras SHAP da dissertação em padrão ABNT (PT-BR + Times), com persistência.

Regenera as duas figuras de explicabilidade do Capítulo 4 da dissertação:

  - ``fig6_shap_descritores``: importância dos 10 descritores físico-químicos
    no ETR de linha de base (teal, identidade do modelo na paleta);
  - ``fig7_shap_embeddings``: top-20 dimensões latentes do ETR sobre
    embeddings do MACE-MP-0 (embH em verde, embN em grafite, com legenda).

Além das figuras (PDF vetorial + PNG 300 dpi), persiste os ranqueamentos
completos em CSV (``results/shap_ranking_*.csv``), de modo que os valores
fiquem auditáveis sem recomputação. Eixos com vírgula decimal (PT-BR) e sem
título interno: a legenda ABNT fica no LaTeX, acima da figura.

Pré-requisitos:
  - descritores: ``data/metadata.sqlite`` (``make data-download data-build``);
  - embeddings:  ``data/mace_features/*_emb.npz`` (``make mace-embeddings``).

O ETR treina em CPU; nenhuma etapa exige GPU.

Uso:
    uv run python scripts/19_shap_figures.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from plot_style import L, MODEL_COLORS, PALETA, apply_abnt_style, save_fig, usar_virgula

logger = logging.getLogger("shap-figures")

SQLITE_PATH = Path("data/metadata.sqlite")
FIG_DIR = Path("results/figures")
CSV_DIR = Path("results")
TOP_K = 20

COR_DESCRITORES = MODEL_COLORS["etr_baseline"]  # teal da paleta da dissertação
COR_EMB_H = MODEL_COLORS["etr_emb"]            # verde; vetor do H adsorvido
COR_EMB_N = PALETA["grafite"]                  # média dos átomos centrais (neutro)

def _barh(labels: list[str], values: np.ndarray, colors, xlabel: str) -> plt.Figure:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(6.0, 0.32 * n + 1.1))
    ax.barh(range(n), values, color=colors)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel(xlabel)
    usar_virgula(ax, "x")
    fig.tight_layout()
    return fig


def fig_descritores() -> None:
    """SHAP dos 10 descritores, reproduzindo o protocolo do etr_baseline."""
    import shap

    from baseline import run_baseline
    from data.splits import load_or_create_splits
    from storage import load_features_frame

    df = load_features_frame(SQLITE_PATH)
    splits = load_or_create_splits(sqlite_path=SQLITE_PATH)
    result = run_baseline(df, splits=splits)
    logger.info("ETR baseline reproduzido: R2 teste=%.4f", result.metrics_test["r2"])

    explainer = shap.TreeExplainer(result.model)
    values = explainer.shap_values(result.X_test)
    ranking = (pd.Series(np.abs(values).mean(axis=0), index=result.features)
               .sort_values(ascending=False).rename("mean_abs_shap"))
    ranking.to_csv(CSV_DIR / "shap_ranking_descritores.csv")
    logger.info("ranqueamento persistido em %s", CSV_DIR / "shap_ranking_descritores.csv")

    asc = ranking.iloc[::-1]
    fig = _barh(list(asc.index), asc.to_numpy(), COR_DESCRITORES, L["importancia_shap"])
    save_fig(fig, "fig6_shap_descritores", FIG_DIR)
    plt.close(fig)


def fig_embeddings() -> None:
    """SHAP das dimensões latentes, reproduzindo o protocolo do etr_emb_all."""
    from analysis.feature_eda import load_xy
    from analysis.feature_importance import fit_etr, shap_importance

    X_train, y_train, X_test, _, names, _, _ = load_xy(suffix="_emb")
    logger.info("embeddings carregados: train=%d test=%d dim=%d",
                len(X_train), len(X_test), len(names))

    grid = {"n_estimators": [300], "max_depth": [None, 20], "min_samples_leaf": [1]}
    model = fit_etr(X_train, y_train, grid)
    ranking, _ = shap_importance(model, X_test, names)
    ranking.to_csv(CSV_DIR / "shap_ranking_embeddings.csv")
    logger.info("ranqueamento persistido em %s", CSV_DIR / "shap_ranking_embeddings.csv")

    top = ranking.head(TOP_K).iloc[::-1]
    cores = [COR_EMB_H if str(n).startswith("embH") else COR_EMB_N for n in top.index]
    fig = _barh(list(top.index), top.to_numpy(), cores, L["importancia_shap"])
    fig.axes[0].legend(handles=[
        Patch(color=COR_EMB_H, label="vetor do H adsorvido (embH)"),
        Patch(color=COR_EMB_N, label="média dos átomos centrais (embN)"),
    ], loc="lower right", frameon=False)
    save_fig(fig, "fig7_shap_embeddings", FIG_DIR)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    apply_abnt_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        fig_descritores()
    except FileNotFoundError as e:
        logger.warning("descritores pulados (%s); rode `make data-download data-build`", e)
    try:
        fig_embeddings()
    except FileNotFoundError as e:
        logger.warning("embeddings pulados (%s); rode `make mace-embeddings`", e)
    logger.info("pronto. Figuras em %s", FIG_DIR)


if __name__ == "__main__":
    main()
