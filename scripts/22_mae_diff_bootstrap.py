"""Paired bootstrap CI for the MAE difference between final predictors.

Usage:
    uv run python scripts/22_mae_diff_bootstrap.py

Complements the Wilcoxon tie (p = 0.13) between the embedding ETR and the
MACE-MP-0 head ensemble with an effect-size statement: the paired bootstrap
(10,000 resamples over the 1,172 test structures) gives a 95% CI for the MAE
difference, so "no significant difference" can be read alongside how large a
difference the data still allow. Writes results/mae_diff_bootstrap.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("bootstrap")

RUNS = Path("results/runs")
ETR_EMB_RUN = "20260529_203011_etr_emb_all"
# The five canonical multi-seed runs aggregated in results/multiseed_summary.md
# (a later re-train of seed 42 exists but is not part of the reported ensemble).
STAGEA_RUNS = [
    "20260529_165552_mace_ft_stageA_v2_seed42",
    "20260529_173755_mace_ft_stageA_v2_seed1",
    "20260529_182056_mace_ft_stageA_v2_seed2",
    "20260529_190526_mace_ft_stageA_v2_seed3",
    "20260529_194841_mace_ft_stageA_v2_seed4",
]
N_BOOT = 10_000
SEED = 42
OUT_JSON = "results/mae_diff_bootstrap.json"


def load_preds(run_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(run_dir / "predictions.parquet")
    df = df[df["split"] == "test"]
    return df.set_index("sid")[["y_true", "y_pred"]]


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    etr = load_preds(RUNS / ETR_EMB_RUN)

    stagea_dirs = [RUNS / name for name in STAGEA_RUNS]
    missing = [d.name for d in stagea_dirs if not d.exists()]
    assert not missing, f"missing stage A runs: {missing}"
    head_preds = [load_preds(d)["y_pred"].rename(d.name) for d in stagea_dirs]
    head = pd.concat(head_preds, axis=1).mean(axis=1)

    joined = etr.join(head.rename("y_pred_head"), how="inner")
    assert len(joined) == 1172, f"expected 1172 paired structures, got {len(joined)}"

    err_etr = np.abs(joined["y_pred"] - joined["y_true"]).to_numpy()
    err_head = np.abs(joined["y_pred_head"] - joined["y_true"]).to_numpy()
    diff = err_etr - err_head
    point = float(diff.mean())

    rng = np.random.default_rng(SEED)
    n = len(diff)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = diff[idx].mean(axis=1)
    lo, hi = (float(q) for q in np.percentile(boot, [2.5, 97.5]))

    out = {
        "comparison": "ETR+embeddings minus MACE-MP-0 head ensemble (n=5)",
        "n_pairs": n,
        "mae_etr_emb_eV": float(err_etr.mean()),
        "mae_head_ens_eV": float(err_head.mean()),
        "mae_diff_eV": point,
        "ci95_low_eV": lo,
        "ci95_high_eV": hi,
        "n_boot": N_BOOT,
        "seed": SEED,
    }
    Path(OUT_JSON).write_text(json.dumps(out, indent=2))
    logger.info("MAE diff = %.4f eV, 95%% CI [%.4f, %.4f] -> %s", point, lo, hi, OUT_JSON)


if __name__ == "__main__":
    main()
