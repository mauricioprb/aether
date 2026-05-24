"""Feature-reduction sweep: ETR on MACE features under several strategies.

Usage:
    uv run python scripts/08b_feature_reduction_sweep.py

Each strategy becomes a run in results/runs/{ts}_etr_mace_{tag}/. Produces
results/figures/feature_reduction_sweep.png (R² test vs # features) with the
ETR-handcrafted (0.934) and SchNet (0.908) baselines as reference lines.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt

from analysis.feature_eda import load_xy
from analysis.feature_importance import (
    fit_etr,
    rfecv_select,
    shap_importance,
)
from analysis.feature_reduction import compose_features, pca_reduce, select, top_k
from training.evaluate import metrics_from_preds
from training.run_logger import RunLogger

logger = logging.getLogger("sweep")

ETR_HANDCRAFTED_R2 = 0.934
SCHNET_R2 = 0.908
FIG_PATH = "results/figures/feature_reduction_sweep.png"


def evaluate_strategy(prefix, tag, X_train, y_train, X_test, y_test, n_features,
                      feature_list, grid=None):
    model = fit_etr(X_train, y_train, grid)
    pred = model.predict(X_test)
    m = metrics_from_preds(y_test, pred)
    n_features = int(n_features)
    feature_list = [str(f) for f in feature_list]
    logger.info("[%s] n=%d  R2=%.4f MAE=%.4f", tag, n_features, m["r2"], m["mae"])
    config = {"model": "ExtraTreesRegressor", "strategy": tag,
              "n_features": n_features, "features": feature_list}
    with RunLogger(name=f"{prefix}_{tag}", config=config) as run:
        run.log_metrics({f"{k}_test": v for k, v in m.items()} | {"n_features": n_features})
        run.log_predictions(y_test, pred, "test")
    return n_features, m["r2"]


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-set", choices=["scalar", "emb"], default="scalar")
    args = parser.parse_args()

    if args.feature_set == "scalar":
        suffix, prefix, fig = "", "etr_mace", FIG_PATH
        top_ks, grid, do_rfe, do_compose = (3, 5), None, True, True
    else:
        suffix, prefix = "_emb", "etr_emb"
        fig = "results/figures/feature_reduction_sweep_emb.png"
        top_ks = (10, 20, 50, 100)
        grid = {"n_estimators": [300], "max_depth": [None, 20], "min_samples_leaf": [1]}
        do_rfe, do_compose = False, False

    X_train, y_train, X_test, y_test, names = load_xy(suffix=suffix)
    logger.info("%s: train=%d test=%d, %d features", prefix, len(X_train), len(X_test), len(names))

    full = fit_etr(X_train, y_train, grid)
    ranking, _ = shap_importance(full, X_test, names)
    logger.info("SHAP top-10:\n%s", ranking.head(10).to_string())

    results = {}
    pred = full.predict(X_test)
    m = metrics_from_preds(y_test, pred)
    with RunLogger(name=f"{prefix}_all", config={"strategy": "all", "n_features": len(names)}) as run:
        run.log_metrics({f"{k}_test": v for k, v in m.items()} | {"n_features": len(names)})
        run.log_predictions(y_test, pred, "test")
    results["all"] = (len(names), m["r2"])
    logger.info("[all] n=%d R2=%.4f", len(names), m["r2"])

    for k in top_ks:
        chosen = top_k(ranking, k)
        results[f"top{k}"] = evaluate_strategy(prefix, f"top{k}",
            select(X_train, names, chosen), y_train,
            select(X_test, names, chosen), y_test, k, chosen, grid)

    for var, tag in [(0.90, "pca90"), (0.95, "pca95"), (0.99, "pca99")]:
        Xtr, Xte, nc = pca_reduce(X_train, X_test, var)
        results[tag] = evaluate_strategy(prefix, tag, Xtr, y_train, Xte, y_test, nc,
                                         [f"PC{i+1}" for i in range(nc)], grid)

    if do_rfe:
        _, rfe_sel = rfecv_select(X_train, y_train, names)
        results["rfe"] = evaluate_strategy(prefix, "rfe",
            select(X_train, names, rfe_sel), y_train,
            select(X_test, names, rfe_sel), y_test, len(rfe_sel), rfe_sel, grid)

    if do_compose:
        base = top_k(ranking, 3)
        Xtr_c, comp_names = compose_features(X_train, names, base)
        Xte_c, _ = compose_features(X_test, names, base)
        results["compose"] = evaluate_strategy(prefix, "compose", Xtr_c, y_train, Xte_c, y_test,
                                               len(comp_names), comp_names, grid)

    _plot(results, fig)
    logger.info("sweep done: %d strategies -> %s", len(results), fig)


def _plot(results: dict, fig_path: str = FIG_PATH) -> None:
    from pathlib import Path
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for tag, (n, r2) in results.items():
        ax.scatter(n, r2, s=60)
        ax.annotate(tag, (n, r2), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(ETR_HANDCRAFTED_R2, ls="--", color="#4c72b0",
               label=f"ETR 10 features ({ETR_HANDCRAFTED_R2})")
    ax.axhline(SCHNET_R2, ls="--", color="#dd8452", label=f"SchNet ({SCHNET_R2})")
    ax.set_xlabel("# features")
    ax.set_ylabel("R² test")
    ax.set_title("Reducao de features MACE (ETR)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)


if __name__ == "__main__":
    main()
