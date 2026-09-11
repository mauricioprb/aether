"""Analogos acessiveis de um catalisador de referencia, por dois criterios.

A ideia: dado um material caro que funciona (Pt(111) na HER), procurar no
recorte curado quais materiais de composicao acessivel se parecem com ele.

"Parecer" tem duas leituras, e elas NAO sao equivalentes:

  1. dG_H proximo  - o criterio de Sabatier. Fraco sozinho: dois materiais
     chegam ao mesmo dG_H por mecanismos diferentes.
  2. embedding proximo - os 512 numeros do MACE descrevem o ambiente local em
     volta do H adsorvido. Proximidade aqui diz que o hidrogenio "ve" um
     ambiente eletronico parecido, nao so que dois escalares coincidem.

Medindo os dois no dataset, a correlacao entre eles e de apenas ~0,31: o
embedding carrega informacao que o escalar nao carrega. Por isso o ranking sai
dos dois, e o relatorio mostra as duas colunas separadas em vez de fundir num
score unico que esconderia a diferenca.

Proveniencia: a acessibilidade de cada elemento vem de data/element_context.csv,
que exige fonte citavel. Sem fonte preenchida o script PARA -- e deliberado:
afirmar que um elemento e "acessivel" ou "abundante no Brasil" e afirmacao
economica, nao resultado do modelo, e nao pode entrar num artigo sem referencia.

Uso:
    uv run python scripts/24_cheap_analogs.py --reference Pt --facet 111
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import PALETA, apply_abnt_style, save_fig
from screening import SABATIER_CORRECTION_EV, load_all_embeddings

logger = logging.getLogger("cheap-analogs")

SQLITE_PATH = Path("data/metadata.sqlite")
CONTEXT_PATH = Path("data/element_context.csv")
OUT_DIR = Path("results/validation")
FIG_DIR = Path("results/figures")


def metals_of(formula: str) -> set[str]:
    return set(re.findall(r"[A-Z][a-z]?", formula)) - {"H"}


def accessible_elements() -> set[str]:
    """Elementos marcados como acessiveis E com fonte citavel.

    O corte por fonte nao e burocracia: e o que impede o resultado de virar uma
    lista de materiais "baratos" apoiada em nada.
    """
    if not CONTEXT_PATH.exists():
        sys.exit(f"FALTA {CONTEXT_PATH}. Rode a preparacao da tabela de contexto.")
    ctx = pd.read_csv(CONTEXT_PATH).fillna("")
    marcados = ctx[ctx.brasil_relevante.astype(str).str.lower().isin({"sim", "true", "1"})]
    if marcados.empty:
        sys.exit(
            f"Nenhum elemento marcado em {CONTEXT_PATH}.\n"
            "Preencha 'brasil_relevante' (sim/nao) e 'fonte'/'url'/'ano' a partir de\n"
            "fonte real (USGS Mineral Commodity Summaries, Anuario Mineral ANM).\n"
            "O script nao inventa esses dados de proposito."
        )
    sem_fonte = marcados[marcados.fonte.astype(str).str.strip() == ""]
    if not sem_fonte.empty:
        sys.exit(
            "Elementos marcados como relevantes mas SEM fonte: "
            f"{', '.join(sem_fonte.element)}.\n"
            "Toda afirmacao de abundancia precisa de referencia citavel."
        )
    logger.info("elementos acessiveis com fonte: %s", ", ".join(marcados.element))
    return set(marcados.element)


def load_frame(conn: sqlite3.Connection, emb: dict) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT id, chemical_formula, facet, site_type, coverage, delta_G_H, pub_id "
        "FROM structures", conn,
    )
    df = df[df.id.isin(emb)].copy()
    df["dG"] = df.delta_G_H + SABATIER_CORRECTION_EV
    df["metals"] = df.chemical_formula.map(metals_of)
    return df


def run(reference: str, facet: str, top: int) -> pd.DataFrame:
    emb = load_all_embeddings()
    conn = sqlite3.connect(SQLITE_PATH)
    df = load_frame(conn, emb)
    conn.close()

    ref = df[(df.metals.map(lambda m: m == {reference})) & (df.facet == facet)]
    if ref.empty:
        sys.exit(f"sem estruturas de {reference} puro na faceta {facet}")
    Xr = np.vstack([emb[i] for i in ref.id]).mean(axis=0)
    dG_ref = float(ref.dG.mean())
    logger.info("referencia: %d estruturas de %s(%s), dG medio %+.3f eV",
                 len(ref), reference, facet, dG_ref)

    acessiveis = accessible_elements()
    cand = df[df.metals.map(lambda m: bool(m) and m <= acessiveis)].copy()
    if cand.empty:
        sys.exit("nenhuma candidata com os elementos marcados como acessiveis")

    Xc = np.vstack([emb[i] for i in cand.id])
    # Cosseno: compara a direcao do ambiente local, nao a magnitude do embedding.
    # O valor absoluto nao significa nada sozinho; o que vale e o ordenamento.
    cand["similaridade"] = (Xc @ Xr) / (np.linalg.norm(Xc, axis=1) * np.linalg.norm(Xr))
    cand["dif_dG"] = (cand.dG - dG_ref).abs()

    # Rank combinado por posicao em cada criterio, nao por soma dos valores: as
    # duas escalas nao sao comparaveis e somar eV com cosseno nao significa nada.
    cand["rank_dG"] = cand.dif_dG.rank()
    cand["rank_emb"] = cand.similaridade.rank(ascending=False)
    cand["rank_combinado"] = cand.rank_dG + cand.rank_emb

    rho = float(np.corrcoef(cand.similaridade, -cand.dif_dG)[0, 1])
    logger.info("correlacao entre os dois criterios: %.3f "
                 "(baixa = o embedding carrega o que o escalar nao carrega)", rho)

    out = cand.nsmallest(top, "rank_combinado").copy()
    out["referencia"] = f"{reference}({facet})"
    out["dG_referencia"] = dG_ref
    return out.sort_values("rank_combinado")


def figure(cand: pd.DataFrame, reference: str) -> None:
    apply_abnt_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color=PALETA["grafite"], lw=1, ls="--")
    sc = ax.scatter(cand.similaridade, cand.dG, s=42, c=cand.rank_combinado,
                     cmap="viridis_r", edgecolor="white", linewidth=0.5)
    for _, r in cand.iterrows():
        ax.annotate(r.chemical_formula, (r.similaridade, r.dG),
                     textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel(f"similaridade de embedding com {reference} (cosseno)")
    ax.set_ylabel("ΔG$_H$ previsto (eV)")
    ax.set_title(f"Análogos acessíveis de {reference}")
    fig.colorbar(sc, ax=ax, label="posição no ranking combinado")
    fig.tight_layout()
    save_fig(fig, "fig_analogos_acessiveis", FIG_DIR)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", default="Pt")
    p.add_argument("--facet", default="111")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()

    cand = run(args.reference, args.facet, args.top)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"analogos_{args.reference}.csv"
    cols = ["chemical_formula", "facet", "site_type", "dG", "dif_dG",
            "similaridade", "rank_dG", "rank_emb", "rank_combinado", "pub_id", "id"]
    cand[cols].to_csv(out, index=False)

    print(cand[["chemical_formula", "facet", "site_type", "dG", "dif_dG",
                 "similaridade", "pub_id"]].to_string(index=False,
                 float_format=lambda v: f"{v:+.4f}"))
    print(f"\ncsv: {out}")
    figure(cand, args.reference)


if __name__ == "__main__":
    main()
