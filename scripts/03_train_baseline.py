"""Train the Extra Trees baseline and emit metrics + figures.

Usage:
    uv run python scripts/03_train_baseline.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from her_gnn.baseline import run_baseline
from her_gnn.plots import plot_dG_hist, plot_parity, plot_shap_bar
from her_gnn.storage import load_features_frame

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

    plot_dG_hist(df["delta_G_H"].to_numpy())
    plot_parity(result)
    plot_shap_bar(result)
    logger.info("figures written to data/figures/")


if __name__ == "__main__":
    main()
