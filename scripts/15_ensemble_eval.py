"""Ensemble evaluation over already-logged runs (no retraining, CPU, seconds).

Combines the multi-seed Stage A runs, the multi-seed SchNet runs and the
deterministic ETR+embeddings run into uniform-weight ensembles, evaluated on
the canonical test set. Weights are NOT fitted anywhere; the optional oracle
sweep (--oracle) is printed as a diagnostic upper bound only.

Usage:
    uv run python scripts/15_ensemble_eval.py            # evaluate + log run
    uv run python scripts/15_ensemble_eval.py --no-log   # evaluate only
    uv run python scripts/15_ensemble_eval.py --oracle   # + oracle pair sweep
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from analysis.ensemble import (
    EnsembleResult,
    latest_runs_by_name,
    oracle_pair_blend,
    uniform_ensemble,
    val_fitted_blend,
)
from training.run_logger import RunLogger

logger = logging.getLogger("ensemble-eval")

SEEDS = [42, 1, 2, 3, 4]
STAGEA_NAMES = [f"mace_ft_stageA_v2_seed{s}" for s in SEEDS]
SCHNET_NAMES = [f"schnet_v2_seed{s}" for s in SEEDS]
ETR_EMB_NAME = "etr_emb_all"

SUMMARY_PATH = Path("results/ensemble_summary.md")

KEY_METRICS = ["r2", "mae", "rmse", "mdae", "max_err", "spearman_rho", "frac_chem_acc"]


def fmt_row(name: str, m: dict[str, float]) -> str:
    return (f"| {name} | {m['r2']:.4f} | {m['mae']:.4f} | {m['rmse']:.4f} "
            f"| {m['mdae']:.4f} | {m['max_err']:.3f} | {m['spearman_rho']:.4f} "
            f"| {100 * m['frac_chem_acc']:.1f} |")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-log", action="store_true", help="skip RunLogger output")
    ap.add_argument("--oracle", action="store_true",
                    help="print oracle pair-blend sweep (diagnostic only)")
    args = ap.parse_args()

    stagea_dirs = latest_runs_by_name(STAGEA_NAMES)
    schnet_dirs = latest_runs_by_name(SCHNET_NAMES)
    etr_dir = latest_runs_by_name([ETR_EMB_NAME])

    results: list[EnsembleResult] = [
        uniform_ensemble("stagea_ens5", stagea_dirs),
        uniform_ensemble("schnet_ens5", schnet_dirs),
        uniform_ensemble("stagea_ens5+etr_emb", {**stagea_dirs, **etr_dir}),
        uniform_ensemble("stagea_ens5+schnet_ens5+etr_emb",
                         {**stagea_dirs, **schnet_dirs, **etr_dir}),
    ]
    # Single-model references on the same aligned test set
    etr_single = uniform_ensemble(ETR_EMB_NAME, etr_dir)

    header = ("| Ensemble | R² | MAE (eV) | RMSE (eV) | MDAE (eV) | max err (eV) "
              "| Spearman ρ | % < 43 meV |")
    sep = "|" + " :---: |" * 8
    lines = [
        "# Ensembles (média uniforme, sem ajuste de pesos)", "",
        "Membros combinados por média simples das predições no test set canônico",
        "(1172 estruturas). Nenhum peso é ajustado; runs individuais em",
        "`results/summary.json`.", "",
        header, sep.replace(":---: |", "--- |", 1),
        fmt_row(f"{ETR_EMB_NAME} (referência, 1 modelo)", etr_single.metrics),
    ]
    for r in results:
        lines.append(fmt_row(f"{r.name} (n={len(r.members)})", r.metrics))

    # Legitimate val-fitted blend (requires val predictions in each run).
    try:
        blend, weights = val_fitted_blend(
            "blend_val_fitted", {**stagea_dirs, **etr_dir})
        lines.append(fmt_row("blend_val_fitted (pesos ajustados na validação)",
                             blend.metrics))
        wtxt = ", ".join(f"{n}={w:.2f}" for n, w in zip(blend.members, weights, strict=True))
        lines += ["", f"Pesos do blend (NNLS na validação): {wtxt}"]
        results.append(blend)
    except (ValueError, FileNotFoundError) as exc:
        lines += ["", f"> blend val-fitted indisponível ({exc}); "
                  "re-treine o Stage A para logar predições de validação."]
        logger.warning("val-fitted blend skipped: %s", exc)

    print("\n".join(lines[5:]))

    if args.oracle:
        y_true, pair = align_pair(results, etr_single)
        w, m = oracle_pair_blend(y_true, pair[0], pair[1])
        print(f"\n[diagnóstico] oracle blend stagea_ens5/etr_emb: w={w:.2f} "
              f"-> R²={m['r2']:.4f}, MAE={m['mae']:.4f} (pesos escolhidos no teste; "
              "NÃO reportável como resultado)")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    logger.info("wrote %s", SUMMARY_PATH)

    if not args.no_log:
        best = max(results, key=lambda r: r.metrics["r2"])
        config = {"ensembles": {r.name: r.members for r in results},
                  "weighting": "uniform (no fitting)"}
        with RunLogger(name="ensemble_eval", config=config) as run:
            run.log_metrics({
                **{f"{k}_test": v for k, v in best.metrics.items()},
                "best_ensemble": best.name,
                **{f"{r.name}_r2_test": r.metrics["r2"] for r in results},
            })
            run.log_predictions(best.y_true, best.y_pred, "test", sid=best.sids)
            run.log_standard_figures(best.y_true, best.y_pred,
                                     model_label=f"ensemble {best.name}",
                                     color="#55a868")


def align_pair(results: list[EnsembleResult], etr_single: EnsembleResult):
    """y_true + (stagea_ens5 preds, etr preds) for the oracle diagnostic."""
    import numpy as np

    stagea = next(r for r in results if r.name == "stagea_ens5")
    assert stagea.sids == etr_single.sids
    return stagea.y_true, np.vstack([stagea.y_pred, etr_single.y_pred])


if __name__ == "__main__":
    main()
