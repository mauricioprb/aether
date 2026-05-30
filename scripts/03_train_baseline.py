"""Train the Extra Trees baseline and emit metrics + figures.

Usage:
    uv run python scripts/03_train_baseline.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from baseline import CV_FOLDS, PARAM_GRID, RANDOM_STATE, TEST_SIZE, run_baseline
from plots import plot_dG_hist, plot_parity, plot_shap_bar
from storage import load_features_frame
from training.run_logger import RunLogger

logger = logging.getLogger("train")

SQLITE_PATH = Path("data/metadata.sqlite")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    df = load_features_frame(SQLITE_PATH)
    logger.info("loaded %d rows with features", len(df))

    result = run_baseline(df)
    logger.info("=== TEST METRICS ===")
    for k, v in result.metrics_test.items():
        logger.info("  %-14s = %.6f", k, v)
    logger.info("best params: %s", result.best_params)

    # Recover IDs aligned with X_train/X_test row order via the original df index.
    train_sids = df.loc[result.X_train.index, "id"].tolist()
    test_sids = df.loc[result.X_test.index, "id"].tolist()

    config = {
        "model": "ExtraTreesRegressor",
        "features": result.features,
        "n_features": len(result.features),
        "n_train": len(result.X_train),
        "n_test": len(result.X_test),
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "param_grid": PARAM_GRID,
        "best_params": result.best_params,
    }
    with RunLogger(name="etr_baseline", config=config) as run:
        run.log_metrics({
            **{f"{k}_test": v for k, v in result.metrics_test.items()},
            **{f"{k}_train": v for k, v in result.metrics_train.items()},
            "n_params": None,
        })
        run.log_predictions(result.y_test, result.y_test_pred, "test", sid=test_sids)
        run.log_predictions(result.y_train, result.y_train_pred, "train", sid=train_sids)
        run.log_standard_figures(result.y_test, result.y_test_pred,
                                  model_label="ETR + 10 handcrafted",
                                  color="#dd8452")
        run.log_figure(plot_dG_hist(df["delta_G_H"].to_numpy()), "dG_hist.png")
        run.log_figure(plot_parity(result), "parity_train_test.png")
        run.log_figure(plot_shap_bar(result), "shap_importance.png")
    logger.info("figures written to results/runs/{ts}_etr_baseline/figures/")


if __name__ == "__main__":
    main()
