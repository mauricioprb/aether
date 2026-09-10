"""Leak-free SHAP feature selection for the embedding reduction study.

Usage:
    uv run python scripts/08c_feature_reduction_valshap.py

Addresses the test-set leakage in scripts/08b_feature_reduction_sweep.py,
where the SHAP ranking that drives the top-k selection was computed over the
test set and the selected subsets were then evaluated on that same test set.
Here the selection uses only non-test data: an ETR is fit on the canonical
train block (4,220), the SHAP ranking is computed over the validation block
(468), and each top-k subset is refit on train+val (4,688) and evaluated once
on the untouched test set (1,172).

Each strategy becomes a run in results/runs/{ts}_etr_emb_valshap_{tag}/.
Also writes results/shap_ranking_embeddings_val.csv and reports the overlap
with the test-based ranking (results/shap_ranking_embeddings.csv).
"""

from __future__ import annotations

import logging

import pandas as pd

from analysis.feature_eda import load_split
from analysis.feature_importance import fit_etr, shap_importance
from analysis.feature_reduction import select, top_k
from training.evaluate import metrics_from_preds
from training.run_logger import RunLogger

logger = logging.getLogger("valshap")

EMB_GRID = {"n_estimators": [300], "max_depth": [None, 20], "min_samples_leaf": [1]}
TOP_KS = (10, 20, 50, 100)
RANKING_VAL_CSV = "results/shap_ranking_embeddings_val.csv"
RANKING_TEST_CSV = "results/shap_ranking_embeddings.csv"


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    import numpy as np

    tr, va, te = (load_split(n, suffix="_emb") for n in ("train", "val", "test"))
    names = tr["names"]
    logger.info("train=%d val=%d test=%d, %d features",
                len(tr["y"]), len(va["y"]), len(te["y"]), len(names))

    selector_model = fit_etr(tr["X"], tr["y"], EMB_GRID)
    ranking, _ = shap_importance(selector_model, va["X"], names)
    ranking.to_csv(RANKING_VAL_CSV)
    logger.info("validation SHAP top-10:\n%s", ranking.head(10).to_string())

    try:
        test_ranking = pd.read_csv(RANKING_TEST_CSV, index_col=0).squeeze("columns")
        for k in TOP_KS:
            inter = len(set(ranking.index[:k]) & set(test_ranking.index[:k]))
            logger.info("top-%d overlap val vs test ranking: %d/%d", k, inter, k)
    except FileNotFoundError:
        logger.warning("test-based ranking %s not found; skipping overlap", RANKING_TEST_CSV)

    X_fit = np.vstack([tr["X"], va["X"]])
    y_fit = np.concatenate([tr["y"], va["y"]])
    test_ids = list(te["ids"])

    for k in TOP_KS:
        chosen = top_k(ranking, k)
        model = fit_etr(select(X_fit, names, chosen), y_fit, EMB_GRID)
        pred = model.predict(select(te["X"], names, chosen))
        m = metrics_from_preds(te["y"], pred)
        logger.info("[valshap top%d] R2=%.4f MAE=%.4f RMSE=%.4f", k, m["r2"], m["mae"], m["rmse"])
        config = {"model": "ExtraTreesRegressor", "strategy": f"valshap_top{k}",
                  "selection": "SHAP over validation block (leak-free)",
                  "selector_fit": "train block only", "n_features": k,
                  "features": [str(f) for f in chosen]}
        with RunLogger(name=f"etr_emb_valshap_top{k}", config=config) as run:
            run.log_metrics({f"{key}_test": v for key, v in m.items()} | {"n_features": k})
            run.log_predictions(te["y"], pred, "test", sid=test_ids)


if __name__ == "__main__":
    main()
