from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from analysis.feature_eda import load_xy
from analysis.feature_importance import fit_etr, shap_importance

logger = logging.getLogger("shap-emb")

FIG_DIR = Path("results/figures")
TOP_K = 20


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, names, _, _ = load_xy(suffix="_emb")
    logger.info("loaded MACE embeddings: train=%d test=%d dim=%d",
                len(X_train), len(X_test), len(names))

    grid = {"n_estimators": [300], "max_depth": [None, 20], "min_samples_leaf": [1]}
    model = fit_etr(X_train, y_train, grid)
    ranking, _ = shap_importance(model, X_test, names)
    top = ranking.head(TOP_K).iloc[::-1]
    logger.info("top-%d SHAP dims:\n%s", TOP_K, top.iloc[::-1].to_string())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.index, top.values, color="#55a868")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(f"ETR + MACE embeddings - top-{TOP_K} dims por importância SHAP")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "fig6_shap_etr_mace_emb.png"
    fig.savefig(out, dpi=300)
    logger.info("saved %s", out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
