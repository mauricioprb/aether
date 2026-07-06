"""Regenera TODAS as figuras em português + Times New Roman (ABNT), a partir
dos ``predictions.parquet`` já salvos. Não precisa de GPU nem de re-treino.

Produz:
  - results/figures/fig1..fig5  (comparação dos 4 modelos + distribuição de ΔG)
  - results/runs/{run}/figures/*.pdf + *.png  (as 4 figuras diagnósticas por run,
    reescritas em PT-BR/Times para cada run que tenha predições de teste)

Cada figura sai em PDF vetorial (preferido pelo LaTeX) e PNG 300 dpi.

Uso:
    uv run python scripts/18_render_figures.py            # tudo
    uv run python scripts/18_render_figures.py --per-run  # só as por-run
    uv run python scripts/18_render_figures.py --compare  # só as de comparação
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import MODEL_COLORS, PALETA, apply_abnt_style, save_fig
from training.evaluate import standard_figures

logger = logging.getLogger("render-figures")


def _color_for(run_name: str) -> str:
    """Cor de identidade do modelo (paleta da dissertação) a partir do nome do run."""
    if "etr_baseline" in run_name:
        return MODEL_COLORS["etr_baseline"]
    if "schnet" in run_name:
        return MODEL_COLORS["schnet"]
    if "stageA" in run_name or "mace_ft" in run_name:
        return MODEL_COLORS["mace_head"]
    if "etr_emb" in run_name or "embeddings" in run_name:
        return MODEL_COLORS["etr_emb"]
    return PALETA["azul"]

RUNS_DIR = Path("results/runs")


def render_compare() -> None:
    """Executa o gerador oficial das figuras de dissertação (fig1..fig5)."""
    import runpy

    logger.info("gerando figuras de comparação (fig1..fig5)")
    runpy.run_path("scripts/11_figures_dissertacao.py", run_name="__main__")


def render_per_run() -> None:
    """Reescreve as 4 figuras diagnósticas de cada run em PT-BR/Times (PDF+PNG)."""
    apply_abnt_style()
    parquets = sorted(glob.glob(str(RUNS_DIR / "*" / "predictions.parquet")))
    if not parquets:
        logger.warning("nenhum predictions.parquet em %s", RUNS_DIR)
        return
    for pq in parquets:
        run_dir = Path(pq).parent
        df = pd.read_parquet(pq)
        test = df[df["split"] == "test"]
        if test.empty:
            continue
        label = "_".join(run_dir.name.split("_")[2:]) or run_dir.name
        figs = standard_figures(test["y_true"].to_numpy(), test["y_pred"].to_numpy(),
                                model_label=label, color=_color_for(run_dir.name))
        for name, fig in figs.items():
            stem = Path(name).stem  # parity, residual_hist, ...
            save_fig(fig, stem, run_dir / "figures")
            plt.close(fig)
        logger.info("run %s: 4 figuras reescritas", run_dir.name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-run", action="store_true", help="só as figuras por-run")
    ap.add_argument("--compare", action="store_true", help="só as de comparação")
    args = ap.parse_args()

    do_all = not (args.per_run or args.compare)
    if args.compare or do_all:
        render_compare()
    if args.per_run or do_all:
        render_per_run()
    logger.info("pronto. Figuras em results/figures/ e results/runs/*/figures/")


if __name__ == "__main__":
    main()
