"""Avaliação em extrapolação: partição agrupada por composição química.

Cria (se ausente) ``data/splits_composition.json`` RESTRITO à base canônica de
5860 estruturas (GroupShuffleSplit por composição, 80/20, semente 42), de modo
que nenhuma composição apareça em treino e teste ao mesmo tempo, e avalia as
duas variantes de ETR sob esse protocolo:

  - ETR + 10 descritores (mesma grade e CV da linha de base canônica);
  - ETR + embeddings MACE-MP-0 (re-fatiamento dos .npz canônicos por id;
    o tronco congelado produz o mesmo embedding por estrutura, então nenhuma
    extração nova é necessária).

Cada avaliação é registrada em ``results/runs/{ts}_{nome}_comp`` via RunLogger,
com métricas e predições persistidas, comparáveis às execuções canônicas.

Uso:
    uv run python scripts/21_composition_split_eval.py
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from analysis.feature_importance import fit_etr
from baseline import run_baseline
from training.evaluate import metrics_from_preds
from training.run_logger import RunLogger

logger = logging.getLogger("comp-split")

SQLITE = Path("data/metadata.sqlite")
CANON = Path("data/splits.json")
COMP = Path("data/splits_composition.json")
MACE_DIR = Path("data/mace_features")
SEED = 42
TEST_SIZE = 0.20


def create_comp_splits() -> dict[str, list[str]]:
    if COMP.exists():
        splits = json.loads(COMP.read_text())
        logger.info("splits de composição existentes: %d train / %d test",
                    len(splits["train"]), len(splits["test"]))
        return splits
    canon = json.loads(CANON.read_text())
    ids = [str(i) for i in canon["train"] + canon["test"]]
    con = sqlite3.connect(SQLITE)
    meta = pd.read_sql_query("SELECT id, composition FROM structures", con)
    con.close()
    meta["id"] = meta["id"].astype(str)
    meta = meta.set_index("id").loc[ids].reset_index()

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr_idx, te_idx = next(gss.split(meta["id"], groups=meta["composition"]))
    splits = {
        "strategy": "composition",
        "train": meta.iloc[tr_idx]["id"].tolist(),
        "test": meta.iloc[te_idx]["id"].tolist(),
    }
    comp_tr = set(meta.iloc[tr_idx]["composition"])
    comp_te = set(meta.iloc[te_idx]["composition"])
    assert not (comp_tr & comp_te), "composição vazando entre treino e teste"
    COMP.write_text(json.dumps(splits))
    logger.info("splits de composição criados (base canônica): %d train / %d test; "
                "%d/%d composições", len(splits["train"]), len(splits["test"]),
                len(comp_tr), len(comp_te))
    return splits


def eval_etr_baseline(splits: dict[str, list[str]]) -> None:
    from storage import load_features_frame

    df = load_features_frame(SQLITE)
    result = run_baseline(df, splits=splits)
    train_sids = df.loc[result.X_train.index, "id"].tolist()
    test_sids = df.loc[result.X_test.index, "id"].tolist()
    config = {"model": "ExtraTreesRegressor", "split": "composition",
              "n_train": len(result.X_train), "n_test": len(result.X_test),
              "best_params": result.best_params}
    with RunLogger(name="etr_baseline_comp", config=config) as run:
        run.log_metrics({f"{k}_test": v for k, v in result.metrics_test.items()})
        run.log_predictions(result.y_test, result.y_test_pred, "test", sid=test_sids)
        run.log_predictions(result.y_train, result.y_train_pred, "train", sid=train_sids)
    logger.info("ETR baseline (composição): R2=%.4f MAE=%.4f",
                result.metrics_test["r2"], result.metrics_test["mae"])


def eval_etr_emb(splits: dict[str, list[str]]) -> None:
    # junta os três npz canônicos e re-fatia pela partição de composição
    parts = []
    for name in ("train", "val", "test"):
        d = np.load(MACE_DIR / f"{name}_emb.npz", allow_pickle=True)
        parts.append((d["ids"].astype(str), d["X"].astype(np.float64),
                      d["y"].astype(np.float64)))
    ids = np.concatenate([p[0] for p in parts])
    X = np.vstack([p[1] for p in parts])
    y = np.concatenate([p[2] for p in parts])
    pos = {i: k for k, i in enumerate(ids)}
    tr = [pos[i] for i in splits["train"]]
    te = [pos[i] for i in splits["test"]]

    grid = {"n_estimators": [300], "max_depth": [None, 20], "min_samples_leaf": [1]}
    model = fit_etr(X[tr], y[tr], grid)
    pred_te, pred_tr = model.predict(X[te]), model.predict(X[tr])
    m = metrics_from_preds(y[te], pred_te)
    config = {"model": "ETR + MACE embeddings 512", "split": "composition",
              "n_train": len(tr), "n_test": len(te)}
    with RunLogger(name="etr_emb_comp", config=config) as run:
        run.log_metrics({f"{k}_test": v for k, v in m.items()})
        run.log_predictions(y[te], pred_te, "test", sid=list(ids[te]))
        run.log_predictions(y[tr], pred_tr, "train", sid=list(ids[tr]))
    logger.info("ETR embeddings (composição): R2=%.4f MAE=%.4f", m["r2"], m["mae"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    splits = create_comp_splits()
    eval_etr_baseline(splits)
    eval_etr_emb(splits)
    logger.info("pronto. Runs em results/runs/*_comp")


if __name__ == "__main__":
    main()
