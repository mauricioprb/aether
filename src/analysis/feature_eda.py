from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MACE_DIR = Path("data/mace_features")


def load_split(name: str, mace_dir: Path = MACE_DIR, suffix: str = "") -> dict:
    d = np.load(Path(mace_dir) / f"{name}{suffix}.npz", allow_pickle=True)
    return {
        "ids": d["ids"],
        "X": d["X"].astype(np.float64),
        "y": d["y"].astype(np.float64),
        "names": [str(n) for n in d["feature_names"]],
    }


def load_xy(mace_dir: Path = MACE_DIR, suffix: str = ""):
    """ETR-style split: train = canonical train (train.npz + val.npz), test = test.npz.
    Matches the handcrafted-feature ETR baseline's 4688/1172 partition.
    suffix="_emb" loads the node-embedding features instead of the scalars."""
    tr, va, te = (load_split(n, mace_dir, suffix) for n in ("train", "val", "test"))
    X_train = np.vstack([tr["X"], va["X"]])
    y_train = np.concatenate([tr["y"], va["y"]])
    return X_train, y_train, te["X"], te["y"], tr["names"]


def as_frame(split: dict) -> pd.DataFrame:
    df = pd.DataFrame(split["X"], columns=split["names"])
    df["delta_G_H"] = split["y"]
    return df


def plot_distributions(df: pd.DataFrame, names: list[str]) -> plt.Figure:
    n = len(names)
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3 * rows))
    for ax, name in zip(axes.flat, names, strict=False):
        ax.hist(df[name], bins=50, color="#4c72b0")
        ax.set_title(name, fontsize=9)
    for ax in axes.flat[n:]:
        ax.set_axis_off()
    fig.tight_layout()
    return fig


def near_zero_variance(df: pd.DataFrame, names: list[str], thresh: float = 1e-8) -> list[str]:
    return [n for n in names if df[n].var() < thresh]


def correlation_matrix(df: pd.DataFrame, names: list[str]) -> plt.Figure:
    import seaborn as sns

    corr = df[[*names, "delta_G_H"]].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Pearson: features MACE x delta_G_H")
    fig.tight_layout()
    return fig, corr


def redundant_pairs(corr: pd.DataFrame, names: list[str], thresh: float = 0.95) -> list[tuple]:
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if abs(corr.loc[a, b]) > thresh:
                pairs.append((a, b, round(float(corr.loc[a, b]), 3)))
    return pairs


def target_correlation(df: pd.DataFrame, names: list[str]) -> pd.Series:
    return (df[names].corrwith(df["delta_G_H"]).abs()
            .sort_values(ascending=False).rename("abs_pearson"))


def plot_target_correlation(corr_y: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    corr_y.sort_values().plot.barh(ax=ax, color="#55a868")
    ax.set_xlabel(r"$|\rho(\mathrm{feature}, \Delta G_H)|$")
    ax.set_title("Correlacao feature x alvo")
    fig.tight_layout()
    return fig
