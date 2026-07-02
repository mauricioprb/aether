"""Unit tests for the unified metric pack and ensemble helpers (no GPU/data)."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.ensemble import oracle_pair_blend, val_fitted_blend
from training.evaluate import metrics_from_preds


def test_perfect_prediction():
    y = np.array([-1.0, 0.0, 0.5, 1.2])
    m = metrics_from_preds(y, y)
    assert m["r2"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["frac_chem_acc"] == pytest.approx(1.0)


def test_known_values():
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.1, 0.9, 2.2, 2.8])  # errors: +.1 -.1 +.2 -.2
    m = metrics_from_preds(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.15)
    assert m["mae_meV"] == pytest.approx(150.0)
    assert m["rmse"] == pytest.approx(np.sqrt((0.01 + 0.01 + 0.04 + 0.04) / 4))
    assert m["max_err"] == pytest.approx(0.2)
    assert m["mdae"] == pytest.approx(0.15)
    # two of four within 43 meV? no: min |err| is 0.1 = 100 meV -> 0
    assert m["frac_chem_acc"] == pytest.approx(0.0)


def test_smape_bounded_across_zero():
    y_true = np.array([-0.5, 0.5])
    y_pred = np.array([0.5, -0.5])  # sign-flipped, worst case
    m = metrics_from_preds(y_true, y_pred)
    assert 0.0 <= m["smape"] <= 2.0


def test_oracle_blend_never_worse_than_members():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    a = y + rng.normal(scale=0.3, size=200)
    b = y + rng.normal(scale=0.3, size=200)
    r2_a = metrics_from_preds(y, a)["r2"]
    r2_b = metrics_from_preds(y, b)["r2"]
    _, m = oracle_pair_blend(y, a, b)
    assert m["r2"] >= max(r2_a, r2_b) - 1e-9


def test_val_fitted_blend_requires_val(tmp_path):
    # A run dir with only test predictions must raise a clear error.
    import pandas as pd

    run = tmp_path / "20260101_000000_fake"
    run.mkdir()
    pd.DataFrame({"sid": ["a", "b"], "y_true": [0.0, 1.0],
                  "y_pred": [0.1, 0.9], "split": ["test", "test"]}
                 ).to_parquet(run / "predictions.parquet")
    with pytest.raises(ValueError, match="no 'val' predictions"):
        val_fitted_blend("x", {"fake": run})
