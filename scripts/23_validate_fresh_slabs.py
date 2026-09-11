"""Validacao externa de geometria: lajes (111) construidas do zero, relaxadas
com MACE-MP-0, preditas pelo modelo e comparadas com o DFT depositado.

Por que isto existe. O R2 reportado mede o modelo sobre a MESMA distribuicao em
que treinou: mesmas publicacoes (6837 das 7238 linhas vem de um unico trabalho),
mesmo funcional, mesmas geometrias relaxadas por DFT. Acertar ali prova
consistencia interna, nao que o modelo aprendeu quimica. Aqui a geometria nao
vem do banco: e gerada por constante de rede tabelada e relaxada por um
potencial universal, sem DFT em ponto nenhum. Se a predicao continuar de pe, o
que o modelo aprendeu generaliza para fora do dataset.

O controle importa tanto quanto o teste: cada laje e predita antes e depois do
relaxamento, entao o efeito da geometria fica isolado do efeito do material.

Uso:
    uv run python scripts/23_validate_fresh_slabs.py
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.build import add_adsorbate, fcc111
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, reference_states
from ase.optimize import BFGS
from scipy.stats import spearmanr

from geometry import min_distance_ratio
from plot_style import PALETA, apply_abnt_style, save_fig
from screening import (
    SABATIER_CORRECTION_EV,
    load_canonical_test_ids,
    predict_atoms,
)

logger = logging.getLogger("fresh-slabs")

SQLITE_PATH = Path("data/metadata.sqlite")
OUT_DIR = Path("results/validation")
FIG_DIR = Path("results/figures")

# Apenas metais cujo estado fundamental e fcc: so neles a (111) e de fato a
# faceta compacta, e montar fcc(111) para um bcc ou hcp seria comparar com uma
# superficie que nao existe. Mo, W, Ti e companhia ficam de fora de proposito.
FCC_METALS = ["Ag", "Al", "Au", "Cu", "Ir", "Ni", "Pb", "Pd", "Pt", "Rh"]

# 2x2 com 1 H = 0,25 ML, que e a cobertura dominante no recorte do dataset.
# 4 camadas com as 2 de baixo fixas e a pratica padrao para adsorcao em slab.
SIZE = (2, 2, 4)
N_LAYERS_FIXED = 2
VACUUM = 10.0
FMAX = 0.05
MAX_STEPS = 200


def _relax_calculator():
    """MACE-MP-0 em float64.

    Separado do calculador de embedding de proposito: o embedding TEM de sair em
    float32 porque foi assim que as features de treino foram extraidas, mas
    otimizar geometria em float32 e impreciso demais (o proprio MACE avisa).
    """
    from mace.calculators import mace_mp

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return mace_mp(model="medium", device=device, default_dtype="float64")


def build_slab(element: str):
    """Laje (111) ideal com 1 H no sitio fcc. Constante de rede tabelada no ASE,
    nao digitada a mao."""
    a = reference_states[atomic_numbers[element]]["a"]
    slab = fcc111(element, size=SIZE, a=a, vacuum=VACUUM)
    add_adsorbate(slab, "H", height=1.0, position="fcc")
    slab.set_constraint(FixAtoms(mask=[at.tag > N_LAYERS_FIXED for at in slab]))
    return slab, float(a)


def dft_reference(conn: sqlite3.Connection, element: str, test_ids: frozenset[str]) -> dict:
    """Referencia DFT do proprio Catalysis Hub para o metal puro em (111).

    Reporta mediana, dispersao e n porque o banco tem varias entradas por metal,
    de publicacoes diferentes: um unico valor esconderia essa variacao.
    """
    df = pd.read_sql_query(
        "SELECT id, chemical_formula, delta_G_H FROM structures WHERE facet = '111'", conn
    )
    metals = df.chemical_formula.map(lambda f: set(re.findall(r"[A-Z][a-z]?", f)) - {"H"})
    rows = df[metals.map(lambda m: m == {element})]
    if rows.empty:
        return {}
    in_test = rows.id.isin(test_ids)
    return {
        "dft_dE_median": float(rows.delta_G_H.median()),
        "dft_dE_std": float(rows.delta_G_H.std(ddof=0)) if len(rows) > 1 else 0.0,
        "n_ref": int(len(rows)),
        "n_ref_test": int(in_test.sum()),
    }


def run(model: str = "ensemble") -> pd.DataFrame:
    calc = _relax_calculator()
    test_ids = load_canonical_test_ids()
    conn = sqlite3.connect(SQLITE_PATH)
    records = []

    for el in FCC_METALS:
        ref = dft_reference(conn, el, test_ids)
        if not ref:
            logger.warning("%s: sem referencia (111) no dataset - pulando", el)
            continue

        slab, a = build_slab(el)
        # Controle: a mesma laje antes de relaxar. Isola o efeito da geometria.
        dE_ideal = float(predict_atoms([slab.copy()], model)[0])

        slab.calc = calc
        t0 = time.perf_counter()
        opt = BFGS(slab, logfile=None)
        opt.run(fmax=FMAX, steps=MAX_STEPS)
        elapsed = time.perf_counter() - t0
        fmax_final = float(np.sqrt((slab.get_forces() ** 2).sum(axis=1)).max())

        relaxed = slab.copy()
        relaxed.calc = None
        relaxed.set_constraint()
        dE_relaxed = float(predict_atoms([relaxed], model)[0])

        records.append({
            "element": el,
            "a_ang": a,
            "n_atoms": len(slab),
            "steps": opt.get_number_of_steps(),
            "relax_sec": round(elapsed, 1),
            "fmax_final": round(fmax_final, 4),
            "converged": fmax_final <= FMAX,
            "min_dist_ratio": round(min_distance_ratio(relaxed), 3),
            "dE_pred_ideal": dE_ideal,
            "dE_pred_relaxed": dE_relaxed,
            **ref,
        })
        logger.info("%-3s relaxado em %4.1fs (%2d passos) | previsto %+.3f | DFT %+.3f",
                     el, elapsed, opt.get_number_of_steps(), dE_relaxed, ref["dft_dE_median"])

    conn.close()
    df = pd.DataFrame(records)
    df["erro_relaxed"] = df.dE_pred_relaxed - df.dft_dE_median
    df["erro_ideal"] = df.dE_pred_ideal - df.dft_dE_median
    # dG_H = dE_H + 0,24 eV: e no espaco de dG que o criterio de Sabatier vale.
    for c in ("dE_pred_relaxed", "dE_pred_ideal", "dft_dE_median"):
        df[c.replace("dE", "dG")] = df[c] + SABATIER_CORRECTION_EV
    return df


def figure(df: pd.DataFrame) -> None:
    apply_abnt_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))

    lo = float(min(df.dft_dE_median.min(), df.dE_pred_relaxed.min())) - 0.1
    hi = float(max(df.dft_dE_median.max(), df.dE_pred_relaxed.max())) + 0.1
    ax1.plot([lo, hi], [lo, hi], color=PALETA["grafite"], lw=1, ls="--", zorder=1)
    ax1.errorbar(df.dft_dE_median, df.dE_pred_relaxed, xerr=df.dft_dE_std,
                  fmt="o", ms=6, color=PALETA["indigo"], ecolor=PALETA["grafite"],
                  elinewidth=0.8, capsize=2, zorder=3)
    for _, r in df.iterrows():
        ax1.annotate(r.element, (r.dft_dE_median, r.dE_pred_relaxed),
                      textcoords="offset points", xytext=(6, -3), fontsize=9)
    mae = float(df.erro_relaxed.abs().mean())
    rho = float(spearmanr(df.dft_dE_median, df.dE_pred_relaxed).statistic)
    ax1.set_xlabel("ΔE$_H$ DFT depositado (eV)")
    ax1.set_ylabel("ΔE$_H$ previsto, laje relaxada (eV)")
    ax1.set_title(f"MAE = {mae*1000:.0f} meV   ρ = {rho:.3f}")
    ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)

    order = df.sort_values("dG_pred_relaxed")
    x = np.arange(len(order))
    ax2.axhline(0, color=PALETA["grafite"], lw=1, ls="--")
    ax2.plot(x, order.dG_pred_relaxed, "o-", color=PALETA["indigo"],
              label="previsto (laje relaxada)")
    ax2.plot(x, order.dft_dG_median, "s--", color=PALETA["teal"], alpha=0.85,
              label="DFT depositado")
    ax2.set_xticks(x, order.element)
    ax2.set_ylabel("ΔG$_H$ (eV)")
    ax2.set_xlabel("metais ordenados pela predição")
    ax2.legend(frameon=False)
    ax2.set_title("Ordenação de Sabatier")

    fig.tight_layout()
    save_fig(fig, "fig_validacao_lajes_externas", FIG_DIR)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="ensemble", choices=["etr_emb", "stagea", "ensemble"])
    args = p.parse_args()

    df = run(args.model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fresh_slabs.csv"
    df.to_csv(out, index=False)

    mae_r = df.erro_relaxed.abs().mean()
    mae_i = df.erro_ideal.abs().mean()
    rho = spearmanr(df.dft_dE_median, df.dE_pred_relaxed).statistic
    sign = (np.sign(df.dG_pred_relaxed) == np.sign(df.dft_dG_median)).mean()

    print(f"\n{'':<4}{'previsto':>10}{'DFT':>9}{'erro':>9}{'ideal':>9}  n_ref")
    for _, r in df.sort_values("dft_dE_median").iterrows():
        print(f"{r.element:<4}{r.dE_pred_relaxed:>+10.3f}{r.dft_dE_median:>+9.3f}"
               f"{r.erro_relaxed:>+9.3f}{r.erro_ideal:>+9.3f}  {r.n_ref:>4}")
    print(f"\nMAE laje relaxada : {mae_r*1000:6.1f} meV")
    print(f"MAE laje ideal    : {mae_i*1000:6.1f} meV  (controle, sem relaxar)")
    print(f"Spearman          : {rho:6.3f}")
    print(f"Acerto de sinal   : {sign:6.1%}")
    print(f"\ncsv: {out}")

    figure(df)


if __name__ == "__main__":
    main()
