from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from baseline import RANDOM_STATE


def top_k(ranking: pd.Series, k: int) -> list[str]:
    return ranking.index[:k].tolist()


def select(X: np.ndarray, names: list[str], chosen: list[str]) -> np.ndarray:
    idx = [names.index(c) for c in chosen]
    return X[:, idx]


def pca_reduce(X_train, X_test, var: float):
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=var, random_state=RANDOM_STATE)
    Xtr = pca.fit_transform(scaler.transform(X_train))
    Xte = pca.transform(scaler.transform(X_test))
    return Xtr, Xte, pca.n_components_


def compose_features(X: np.ndarray, names: list[str], base: list[str]) -> tuple[np.ndarray, list[str]]:
    """Derived features from the most important base features: ratios, products,
    squares (analogy to the paper's composite phi = Nd0^2 / psi0)."""
    cols = {n: X[:, names.index(n)] for n in base}
    new_cols, new_names = [], []
    for a in base:
        new_cols.append(cols[a] ** 2)
        new_names.append(f"{a}^2")
    for a, b in combinations(base, 2):
        denom = np.where(np.abs(cols[b]) < 1e-8, 1e-8, cols[b])
        new_cols.append(cols[a] / denom)
        new_names.append(f"{a}/{b}")
        new_cols.append(cols[a] * cols[b])
        new_names.append(f"{a}*{b}")
    X_aug = np.column_stack([X, *new_cols])
    return X_aug, names + new_names
