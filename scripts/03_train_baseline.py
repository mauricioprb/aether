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
        logger.info("  %-5s = %.4f", k, v)
    logger.info("best params: %s", result.best_params)

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
            **{f"{k.lower()}_test": v for k, v in result.metrics_test.items()},
            **{f"{k.lower()}_train": v for k, v in result.metrics_train.items()},
            "n_params": None,
        })
        run.log_predictions(result.y_test, result.y_test_pred, "test")
        run.log_predictions(result.y_train, result.y_train_pred, "train")
        run.log_figure(plot_dG_hist(df["delta_G_H"].to_numpy()), "fig3b_dG_hist.png")
        run.log_figure(plot_parity(result), "fig4f_parity.png")
        run.log_figure(plot_shap_bar(result), "fig6d_shap.png")
    logger.info("figures written to data/figures/ and results/runs/")


if __name__ == "__main__":
    main()
