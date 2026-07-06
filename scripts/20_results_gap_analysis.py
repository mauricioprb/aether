"""Análises pendentes do Capítulo 4 da dissertação, a partir de artefatos persistidos.

Produz três blocos, todos rastreáveis (nenhum modelo é retreinado):

1. TRIAGEM (Seção 4.6): ranqueia o conjunto de teste canônico pelo critério de
   Sabatier (s = |dE_pred + 0,24|) usando as predições salvas dos três
   preditores da metodologia: ETR + embeddings, média das 5 execuções da cabeça
   (Stage A) e a média dos dois ("ensemble"). Persiste
   ``results/screen_test_ranking.csv`` e imprime top-k, precisão@k e
   enriquecimento (janela |dG_dft| <= 0,1 eV), além do recorte sem metais
   nobres (motivação da justificativa: substituir Pt).

2. ESTRATOS (Seção 4.1): repartição da base canônica (5860) por fonte/funcional
   (bimetálicas BEEF-vdW vs. nitretos PBE), tipo de sítio e faceta, com
   estatísticas do rótulo por estrato. Persiste ``results/strata_table.csv``.

3. CAUDA (Seção 4.5): diagnóstico geométrico das estruturas com erro > 0,5 eV
   nos quatro preditores (tail_audit.csv): distância mínima H-metal, altura do
   H em relação à camada superior e rótulos divergentes de gêmeos
   (mesma fórmula/faceta/sítio). Persiste ``results/tail_diagnosis.csv``.

Uso:
    uv run python scripts/20_results_gap_analysis.py
"""

from __future__ import annotations

import glob
import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("gap-analysis")

RESULTS = Path("results")
SQLITE = Path("data/metadata.sqlite")
TRAJ = Path("data/processed/her_dataset.traj")
SPLITS = Path("data/splits.json")
DG_CORR = 0.24          # dG_H = dE_H + 0,24 eV (Nørskov 2005)
JANELA_OTIMO = 0.10     # |dG_dft| <= 0,10 eV conta como "próximo do ótimo"
NOBRES = {"Pt", "Pd", "Rh", "Ir", "Ru", "Os", "Au", "Ag", "Re"}


def _meta() -> pd.DataFrame:
    con = sqlite3.connect(SQLITE)
    try:
        return pd.read_sql_query(
            "SELECT id, composition, chemical_formula, facet, site_type, "
            "delta_G_H, dft_functional, pub_id, n_atoms, traj_index "
            "FROM structures", con)
    finally:
        con.close()


def _test_preds() -> pd.DataFrame:
    """Predições de teste persistidas: etr_emb, média stagea (5 sementes)."""
    emb = pd.read_parquet(glob.glob("results/runs/*_etr_emb_all/predictions.parquet")[0])
    emb = emb[emb.split == "test"][["sid", "y_true", "y_pred"]].rename(
        columns={"y_pred": "pred_etr_emb"})
    frames = []
    for pq in sorted(glob.glob("results/runs/*_mace_ft_stageA_v2_seed*/predictions.parquet")):
        df = pd.read_parquet(pq)
        df = df[df.split == "test"][["sid", "y_pred"]]
        frames.append(df.set_index("sid")["y_pred"])
    stagea = pd.concat(frames, axis=1).mean(axis=1).rename("pred_stagea").reset_index()
    out = emb.merge(stagea, on="sid", validate="1:1")
    out["pred_ensemble"] = (out.pred_etr_emb + out.pred_stagea) / 2
    return out


def _elements(composition: str) -> set[str]:
    from ase.formula import Formula
    return set(Formula(composition).count())


def triagem() -> None:
    meta = _meta()
    df = _test_preds().merge(meta, left_on="sid", right_on="id", validate="1:1")
    for c in ("pred_etr_emb", "pred_stagea", "pred_ensemble", "y_true"):
        df[f"dG_{c}"] = df[c] + DG_CORR
    df["s_ensemble"] = df["dG_pred_ensemble"].abs()
    df["dG_dft"] = df["dG_y_true"]
    df = df.sort_values("s_ensemble").reset_index(drop=True)
    df["nobre"] = df["composition"].map(lambda c: bool(_elements(c) & NOBRES))

    cols = ["sid", "chemical_formula", "facet", "site_type", "dft_functional",
            "dG_pred_etr_emb", "dG_pred_stagea", "dG_pred_ensemble",
            "dG_dft", "s_ensemble", "nobre"]
    df[cols].to_csv(RESULTS / "screen_test_ranking.csv", index=False)

    base = (df["dG_dft"].abs() <= JANELA_OTIMO).mean()
    print("\n=== TRIAGEM (teste canônico, n=%d) ===" % len(df))
    print(f"fração base |dG_dft|<= {JANELA_OTIMO}: {base:.3f}")
    for k in (10, 20, 50):
        top = df.head(k)
        prec = (top["dG_dft"].abs() <= JANELA_OTIMO).mean()
        print(f"precisão@{k}: {prec:.2f}  (enriquecimento {prec / base:.1f}x)")
    print("\nTop-10 geral (ensemble):")
    print(df.head(10)[["chemical_formula", "facet", "site_type",
                       "dG_pred_ensemble", "dG_dft"]].to_string(index=False))
    sem = df[~df.nobre]
    print(f"\nTop-10 SEM metais nobres (candidatos: {len(sem)}):")
    print(sem.head(10)[["chemical_formula", "facet", "site_type",
                        "dG_pred_ensemble", "dG_dft"]].to_string(index=False))
    p10 = (sem.head(10)["dG_dft"].abs() <= JANELA_OTIMO).mean()
    print(f"precisão@10 (sem nobres): {p10:.2f}")


def estratos() -> None:
    splits = json.loads(SPLITS.read_text())
    canon = set(splits["train"]) | set(splits["test"])
    meta = _meta()
    meta = meta[meta.id.astype(str).isin({str(i) for i in canon})].copy()
    fam = {"BEEF-vdW": "Bimetálicas (BEEF-vdW)", "PBE": "Nitretos (PBE)"}
    meta["familia"] = meta.dft_functional.map(fam).fillna(meta.dft_functional)

    linhas = []
    for nome, g in meta.groupby("familia"):
        linhas.append({
            "familia": nome, "n": len(g), "pct": 100 * len(g) / len(meta),
            "top": (g.site_type == "top").mean() * 100,
            "bridge": (g.site_type == "bridge").mean() * 100,
            "hollow": (g.site_type == "hollow").mean() * 100,
            "n_facetas": g.facet.nunique(),
            "dE_medio": g.delta_G_H.mean(), "dE_dp": g.delta_G_H.std(),
            "dE_min": g.delta_G_H.min(), "dE_max": g.delta_G_H.max(),
        })
    tab = pd.DataFrame(linhas)
    tab.to_csv(RESULTS / "strata_table.csv", index=False)
    print("\n=== ESTRATOS (base canônica, n=%d) ===" % len(meta))
    print(tab.to_string(index=False))
    print("\nfacetas mais frequentes:")
    print(meta.facet.value_counts().head(6).to_string())
    print("\ncomposições distintas:", meta.composition.nunique())


def cauda() -> None:
    from ase.io import Trajectory

    audit = pd.read_csv(RESULTS / "tail_audit.csv")
    piores = audit[audit["n_models_gt_0.5eV"] == 4].copy()
    meta = _meta().set_index("id")
    traj = Trajectory(str(TRAJ))
    full = _meta()

    diag = []
    for _, r in piores.iterrows():
        m = meta.loc[r.sid]
        atoms = traj[int(m.traj_index)]
        sym = atoms.get_chemical_symbols()
        h_idx = [i for i, s in enumerate(sym) if s == "H"]
        met = [i for i, s in enumerate(sym) if s != "H"]
        dmin, altura = np.nan, np.nan
        if h_idx:
            h = h_idx[0]
            d = atoms.get_distances(h, met, mic=True)
            dmin = float(d.min())
            z_h = atoms.positions[h, 2]
            z_top = np.percentile(atoms.positions[met, 2], 90)
            altura = float(z_h - z_top)
        # gêmeos: mesma fórmula/faceta/sítio com rótulos divergentes
        tw = full[(full.chemical_formula == m.chemical_formula)
                  & (full.facet == m.facet) & (full.site_type == m.site_type)]
        spread = float(tw.delta_G_H.max() - tw.delta_G_H.min()) if len(tw) > 1 else 0.0
        diag.append({
            "sid": r.sid, "formula": m.chemical_formula, "faceta": m.facet,
            "sitio": m.site_type, "funcional": m.dft_functional,
            "dE_dft": float(m.delta_G_H), "erro_medio": float(r.mean_abs_err),
            "dmin_H_metal": dmin, "altura_H": altura,
            "n_gemeos": int(len(tw)), "amplitude_gemeos": spread,
        })
    out = pd.DataFrame(diag)
    out.to_csv(RESULTS / "tail_diagnosis.csv", index=False)
    print("\n=== CAUDA: erro unânime > 0,5 eV nos 4 preditores ===")
    print(out.to_string(index=False))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    triagem()
    estratos()
    cauda()


if __name__ == "__main__":
    main()
