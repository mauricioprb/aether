"""Compare the four final dissertation models from results/ only - never trains.

Whitelist defined in ``src/analysis/comparison.py:WHITELIST``.

Usage:
    uv run python scripts/06_compare.py

Reads results/summary.json + each run's predictions.parquet and writes:
    results/comparison_table.md
    results/figures/comparison_parity_grid.png
    results/figures/comparison_cumulative_error.png
    results/figures/comparison_residual_overlay.png
    results/figures/comparison_mae_bar.png
"""

from __future__ import annotations

import logging
from pathlib import Path

from analysis.comparison import (
    WHITELIST,
    comparison_table,
    load_whitelist,
    plot_cumulative_error,
    plot_mae_bar,
    plot_parity_grid,
    plot_residual_overlay,
)

logger = logging.getLogger("compare")

RESULTS = Path("results")
FIG_DIR = RESULTS / "figures"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    runs = load_whitelist()
    if not runs:
        logger.error("no runs available - retrain models with the unified pipeline first")
        return

    logger.info("comparing %d/%d whitelist models: %s",
                len(runs), len(WHITELIST), [r.display for r in runs])

    table = comparison_table(runs)
    (RESULTS / "comparison_table.md").write_text(table)
    print(table)

    plot_parity_grid(runs).savefig(FIG_DIR / "comparison_parity_grid.png", dpi=300)
    plot_cumulative_error(runs).savefig(FIG_DIR / "comparison_cumulative_error.png", dpi=300)
    plot_residual_overlay(runs).savefig(FIG_DIR / "comparison_residual_overlay.png", dpi=300)
    plot_mae_bar(runs).savefig(FIG_DIR / "comparison_mae_bar.png", dpi=300)
    logger.info("comparison written to %s", RESULTS)


if __name__ == "__main__":
    main()
