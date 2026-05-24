from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


class RunLogger:
    def __init__(self, name: str, config: dict[str, Any], results_dir: Path = RESULTS_DIR):
        self.name = name
        self.config = config
        self.results_dir = Path(results_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / "runs" / f"{timestamp}_{name}"
        self.timestamp = timestamp
        self.metrics: dict[str, Any] = {}
        self._preds: list[pd.DataFrame] = []
        self._start = 0.0

    def __enter__(self) -> RunLogger:
        (self.run_dir / "figures").mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "config.yaml", "w") as fh:
            yaml.safe_dump(self.config, fh, sort_keys=False)
        self._write_env()
        self._start = time.time()
        logger.info("run started: %s", self.run_dir)
        return self

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics.update(metrics)

    def log_predictions(self, y_true, y_pred, split: str) -> None:
        self._preds.append(pd.DataFrame({
            "y_true": list(map(float, y_true)),
            "y_pred": list(map(float, y_pred)),
            "split": split,
        }))

    def log_figure(self, fig, name: str) -> None:
        fig.savefig(self.run_dir / "figures" / name, dpi=300, bbox_inches="tight")

    def _write_env(self) -> None:
        try:
            out = subprocess.run(["uv", "pip", "freeze"], capture_output=True, text=True, timeout=60)
            (self.run_dir / "env.txt").write_text(out.stdout)
        except Exception as exc:
            logger.warning("could not capture env: %s", exc)
            (self.run_dir / "env.txt").write_text("")

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            logger.error("run failed: %s", exc)
            return
        elapsed = time.time() - self._start
        self.metrics.setdefault("elapsed_sec", round(elapsed, 2))
        with open(self.run_dir / "metrics.json", "w") as fh:
            json.dump(self.metrics, fh, indent=2)
        if self._preds:
            pd.concat(self._preds, ignore_index=True).to_parquet(
                self.run_dir / "predictions.parquet", index=False)
        self._update_summary()
        logger.info("run saved: %s (%.1fs)", self.run_dir, elapsed)

    def _update_summary(self) -> None:
        summary_path = self.results_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
        summary = [e for e in summary if e.get("run_dir") != str(self.run_dir)]
        summary.append({
            "timestamp": self.timestamp,
            "name": self.name,
            "run_dir": str(self.run_dir),
            **self.metrics,
        })
        summary_path.write_text(json.dumps(summary, indent=2))
