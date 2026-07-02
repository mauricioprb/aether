"""Screen HER catalysts by composition (CLI).

Thin wrapper over ``src/screening.py:screen``. Filter dataset of 5860 structures
by required elements, predict ΔG_H with the chosen model, rank by |ΔG_H_pred|
(Sabatier principle: optimal HER catalyst has ΔG_H ≈ 0).

Usage:
    uv run python scripts/14_screen.py --elements Pt Ni --top 10
    uv run python scripts/14_screen.py --elements Pd --top 50 --model stagea
    uv run python scripts/14_screen.py --elements Cu --top 20 --model ensemble \\
        --exclude-train --output results/screen_Cu.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from screening import screen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--elements", nargs="+", required=True,
                   help="Required metal elements (e.g. --elements Pt Ni)")
    p.add_argument("--top", type=int, default=10,
                   help="Number of top candidates to return (default: 10)")
    p.add_argument("--model", choices=["etr_emb", "stagea", "ensemble"],
                   default="etr_emb",
                   help="Predictor model")
    p.add_argument("--exclude-train", action="store_true",
                   help="Restrict candidates to canonical test set (no train leakage)")
    p.add_argument("--dg-correction", type=float, default=None,
                   help="ΔG_H = ΔE_H + corr (eV). Default: 0.24 (Nørskov 2005); "
                        "use 0 to rank on raw ΔE_H")
    p.add_argument("--output", default=None,
                   help="Optional CSV output path")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    kwargs = {} if args.dg_correction is None else {"dg_correction": args.dg_correction}
    result = screen(elements=args.elements, top=args.top, model=args.model,
                     exclude_train=args.exclude_train, **kwargs)

    if result.n_candidates == 0:
        print(f"No structures contain all of {result.elements}.")
        return

    out = pd.DataFrame(result.rows)
    cols = ["chemical_formula", "facet", "site_type", "dG_pred", "dG_dft",
            "abs_dG_pred", "error_vs_dft", "id"]
    if args.model == "ensemble":
        cols.insert(4, "dG_pred_etr")
        cols.insert(5, "dG_pred_stagea")
    out = out[cols]

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", 40)
    print(f"\nTop {args.top} candidates (elements: {result.elements}, "
          f"model: {result.model}, n_candidates={result.n_candidates}, "
          f"ranked by |ΔG_H_pred|, ΔG = ΔE + {result.dg_correction:.2f} eV):\n")
    print(out.to_string(index=False))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
