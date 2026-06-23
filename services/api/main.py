from __future__ import annotations

import logging
import os
import secrets
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from screening import (
    ModelName,
    available_elements,
    load_full_dataset,
    screen,
)

from analysis.comparison import CHEM_ACCURACY_EV, load_whitelist
from analysis.multiseed import DEFAULT_GROUPS, aggregate_groups

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                     datefmt="%H:%M:%S")

_DOCS_ENABLED = os.environ.get("ENABLE_DOCS", "0") == "1"

app = FastAPI(
    title="AETHER: HER catalyst screening API",
    description=(
        "Screen Catalysis Hub HER catalysts by composition. "
        "Filter by required metal elements, predict ΔG_H with MACE-MP-0 + ETR "
        "or MACE Stage A fine-tune, rank by |ΔG_H_pred| (Sabatier ≈ 0)."
    ),
    version="0.1.0",
    docs_url="/docs" if _DOCS_ENABLED else None,
    redoc_url="/redoc" if _DOCS_ENABLED else None,
    openapi_url="/openapi.json" if _DOCS_ENABLED else None,
)

_API_KEY = os.environ.get("API_KEY", "")


def require_api_key(x_api_key: str = Header(default="")) -> None:
    if not _API_KEY:
        return
    if not secrets.compare_digest(x_api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request, exc):
    """Never leak stack traces / internal paths to clients. Log full detail
    server-side, return a generic 500."""
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


class ScreenRequest(BaseModel):
    elements: list[str] = Field(..., min_length=1, max_length=20,
                                  description="Required metal elements (e.g. ['Pt', 'Ni'])",
                                  examples=[["Pt", "Ni"]])
    top: int = Field(default=10, ge=1, le=500,
                       description="Number of top candidates to return")
    model: Literal["etr_emb", "stagea", "ensemble"] = Field(
        default="etr_emb",
        description="Predictor: etr_emb (fast CPU, R²=0.961), "
                     "stagea (GNN, R²=0.956 ± 0.002), ensemble (mean)",
    )
    exclude_train: bool = Field(default=False,
                                  description="Restrict to canonical test set "
                                              "(1172 IDs) to avoid memorised picks")


class CandidateRow(BaseModel):
    id: str
    chemical_formula: str
    composition: str
    facet: str
    site_type: str
    coverage: float | None = None
    delta_G_H: float
    dG_pred: float
    abs_dG_pred: float
    error_vs_dft: float
    dG_pred_etr: float | None = None
    dG_pred_stagea: float | None = None


class ScreenResponse(BaseModel):
    elements: list[str]
    model: ModelName
    top: int
    exclude_train: bool
    n_candidates: int
    rows: list[CandidateRow]


class StatsResponse(BaseModel):
    n_structures: int
    n_test_canonical: int
    available_elements: list[str]
    available_models: list[str]


class ModelComparisonRow(BaseModel):
    display: str
    color: str
    kind: Literal["baseline", "gnn", "hybrid"]
    is_multiseed: bool
    n_seeds: int | None = None
    r2_test: float
    r2_test_std: float | None = None
    mae_test: float
    mae_test_std: float | None = None
    mae_meV_test: float
    rmse_test: float
    mdae_test: float | None = None
    pearson_r_test: float | None = None
    spearman_rho_test: float | None = None
    frac_chem_acc_test: float | None = None
    n_params: int | None = None
    elapsed_sec: float | None = None


class ComparisonResponse(BaseModel):
    models: list[ModelComparisonRow]
    chemical_accuracy_eV: float = CHEM_ACCURACY_EV


class ModelPredictions(BaseModel):
    display: str
    color: str
    y_true: list[float]
    y_pred: list[float]


class ComparisonPredictionsResponse(BaseModel):
    models: list[ModelPredictions]
    chemical_accuracy_eV: float = CHEM_ACCURACY_EV


KIND_BY_NAME: dict[str, Literal["baseline", "gnn", "hybrid"]] = {
    "etr_baseline": "baseline",
    "schnet_v2_seed2": "gnn",
    "mace_ft_stageA_v2_seed3": "gnn",
    "etr_emb_all": "hybrid",
}


@app.get("/", summary="API root", tags=["meta"])
def root():
    return {
        "name": app.title,
        "version": app.version,
        "docs": "/docs",
        "endpoints": ["/stats", "/elements", "/screen"],
    }


@app.get("/stats", response_model=StatsResponse, tags=["meta"])
def stats():
    """Dataset + service stats."""
    from screening import load_canonical_test_ids

    df = load_full_dataset()
    return StatsResponse(
        n_structures=len(df),
        n_test_canonical=len(load_canonical_test_ids()),
        available_elements=available_elements(),
        available_models=["etr_emb", "stagea", "ensemble"],
    )


@app.get("/elements", tags=["meta"])
def elements():
    """List metal element symbols present in the dataset."""
    return {"elements": available_elements()}


@app.get("/comparison/predictions", response_model=ComparisonPredictionsResponse,
          tags=["meta"])
def comparison_predictions():
    """Raw test-set predictions for the 4 whitelist models — for client-side charting
    (parity, residual histogram, cumulative error). ~600 KB JSON."""
    runs = load_whitelist()
    return ComparisonPredictionsResponse(
        models=[
            ModelPredictions(
                display=r.display,
                color=r.color,
                y_true=r.preds["y_true"].astype(float).tolist(),
                y_pred=r.preds["y_pred"].astype(float).tolist(),
            )
            for r in runs
        ]
    )


@app.get("/comparison", response_model=ComparisonResponse, tags=["meta"])
def comparison():
    """Whitelisted models with multi-seed mean ± std when applicable."""
    runs = load_whitelist()
    multiseed_index = {g.group: g for g in aggregate_groups(DEFAULT_GROUPS)}
    rows: list[ModelComparisonRow] = []
    for r in runs:
        ms = r.multiseed_stats
        entry = r.entry
        ms_group = multiseed_index.get(entry.get("name", "").rsplit("_seed", 1)[0])
        means = ms_group.means if ms_group else None
        stds = ms_group.stds if ms_group else None
        r2 = (means or {}).get("r2_test", entry.get("r2_test", 0.0))
        mae = (means or {}).get("mae_test", entry.get("mae_test", 0.0))
        rows.append(ModelComparisonRow(
            display=r.display,
            color=r.color,
            kind=KIND_BY_NAME.get(entry.get("name", ""), "hybrid"),
            is_multiseed=bool(ms),
            n_seeds=ms["n"] if ms else None,
            r2_test=r2,
            r2_test_std=(stds or {}).get("r2_test") if ms else None,
            mae_test=mae,
            mae_test_std=(stds or {}).get("mae_test") if ms else None,
            mae_meV_test=mae * 1000.0,
            rmse_test=(means or {}).get("rmse_test", entry.get("rmse_test", 0.0)),
            mdae_test=(means or {}).get("mdae_test", entry.get("mdae_test")),
            pearson_r_test=(means or {}).get("pearson_r_test", entry.get("pearson_r_test")),
            spearman_rho_test=(means or {}).get("spearman_rho_test", entry.get("spearman_rho_test")),
            frac_chem_acc_test=(means or {}).get("frac_chem_acc_test", entry.get("frac_chem_acc_test")),
            n_params=entry.get("n_params"),
            elapsed_sec=entry.get("elapsed_sec"),
        ))
    return ComparisonResponse(models=rows)


@app.post("/screen", response_model=ScreenResponse, tags=["screen"],
          dependencies=[Depends(require_api_key)])
def screen_endpoint(req: ScreenRequest):
    """Run screening with the given query and return ranked top-N candidates."""
    try:
        result = screen(elements=req.elements, top=req.top,
                        model=req.model, exclude_train=req.exclude_train)
    except FileNotFoundError as exc:
        logger.error("screen artifact missing: %s", exc)
        raise HTTPException(status_code=503, detail="service temporarily unavailable")
    except ValueError as exc:
        logger.warning("bad screen request: %s", exc)
        raise HTTPException(status_code=400, detail="invalid screening parameters")
    return ScreenResponse(**result.__dict__)


@app.get("/screen", response_model=ScreenResponse, tags=["screen"],
         dependencies=[Depends(require_api_key)])
def screen_get(
    elements: list[str] = Query(..., min_length=1, max_length=20,
                                  description="Required metals (repeatable: ?elements=Pt&elements=Ni)"),
    top: int = Query(10, ge=1, le=500),
    model: Literal["etr_emb", "stagea", "ensemble"] = Query("etr_emb"),
    exclude_train: bool = Query(False),
):
    """GET variant of /screen (browser-friendly)."""
    return screen_endpoint(ScreenRequest(elements=elements, top=top,
                                           model=model, exclude_train=exclude_train))
