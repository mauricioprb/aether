from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

from .features import FEATURE_NAMES

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 10

PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}


@dataclass
class BaselineResult:
    model: ExtraTreesRegressor
    best_params: dict[str, Any]
    metrics_train: dict[str, float]
    metrics_test: dict[str, float]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    y_train_pred: np.ndarray
    y_test_pred: np.ndarray
    features: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
    }


def run_baseline(df: pd.DataFrame, features: list[str] | None = None,
                 grid: dict[str, list] | None = None) -> BaselineResult:
    """Split, tune via 10-fold CV, evaluate on test. Returns a ``BaselineResult``."""
    features = features or list(FEATURE_NAMES)
    grid = grid or PARAM_GRID

    X = df[features].copy()
    y = df["delta_G_H"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("train=%d test=%d features=%d", len(X_train), len(X_test), len(features))

    search = GridSearchCV(
        ExtraTreesRegressor(random_state=RANDOM_STATE),
        grid,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    logger.info("best params: %s (cv R2=%.4f)", search.best_params_, search.best_score_)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    m_test = metrics(y_test, y_test_pred)
    logger.info("test metrics: %s", {k: round(v, 4) for k, v in m_test.items()})

    return BaselineResult(
        model=model,
        best_params=search.best_params_,
        metrics_train=metrics(y_train, y_train_pred),
        metrics_test=m_test,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
    )
