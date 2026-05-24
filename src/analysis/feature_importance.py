from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV

from baseline import CV_FOLDS, PARAM_GRID, RANDOM_STATE

logger = logging.getLogger(__name__)


def fit_etr(X_train, y_train, grid: dict | None = None) -> ExtraTreesRegressor:
    search = GridSearchCV(
        ExtraTreesRegressor(random_state=RANDOM_STATE),
        grid or PARAM_GRID, cv=CV_FOLDS, scoring="r2", n_jobs=-1,
    )
    search.fit(X_train, y_train)
    logger.info("ETR best params: %s (cv R2=%.4f)", search.best_params_, search.best_score_)
    return search.best_estimator_


def shap_importance(model, X_test, names: list[str]):
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_test)
    mean_abs = np.abs(values).mean(axis=0)
    ranking = (pd.Series(mean_abs, index=names)
               .sort_values(ascending=False).rename("mean_abs_shap"))
    return ranking, values


def pareto_count(ranking: pd.Series, frac: float = 0.9) -> int:
    cum = ranking.cumsum() / ranking.sum()
    return int((cum < frac).sum() + 1)


def permutation_ranking(model, X_test, y_test, names: list[str]) -> pd.Series:
    r = permutation_importance(model, X_test, y_test, n_repeats=20,
                               random_state=RANDOM_STATE, n_jobs=-1)
    return (pd.Series(r.importances_mean, index=names)
            .sort_values(ascending=False).rename("perm_importance"))


def rfecv_select(X_train, y_train, names: list[str]) -> tuple[int, list[str]]:
    selector = RFECV(
        ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE),
        step=1, cv=CV_FOLDS, scoring="r2", n_jobs=-1,
    )
    selector.fit(X_train, y_train)
    selected = [n for n, keep in zip(names, selector.support_, strict=True) if keep]
    logger.info("RFECV optimal: %d features %s", selector.n_features_, selected)
    return selector.n_features_, selected


def shap_bar(ranking: pd.Series, title: str = "mean(|SHAP|)"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ranking.sort_values().plot.barh(ax=ax, color="#55a868")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def shap_beeswarm(values, X_test, names: list[str]):
    import matplotlib.pyplot as plt
    import shap

    fig = plt.figure()
    shap.summary_plot(values, X_test, feature_names=names, show=False)
    plt.tight_layout()
    return fig
