from __future__ import annotations

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from screening import (
    ModelName,
    available_elements,
    load_full_dataset,
    screen,
)

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                     datefmt="%H:%M:%S")

app = FastAPI(
    title="AETHER HER catalyst screening API",
    description=(
        "Screen Catalysis Hub HER catalysts by composition. "
        "Filter by required metal elements, predict ΔG_H with MACE-MP-0 + ETR "
        "or MACE Stage A fine-tune, rank by |ΔG_H_pred| (Sabatier ≈ 0)."
    ),
    version="0.1.0",
)


class ScreenRequest(BaseModel):
    elements: list[str] = Field(..., min_length=1,
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


@app.post("/screen", response_model=ScreenResponse, tags=["screen"])
def screen_endpoint(req: ScreenRequest):
    """Run screening with the given query and return ranked top-N candidates."""
    try:
        result = screen(elements=req.elements, top=req.top,
                        model=req.model, exclude_train=req.exclude_train)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ScreenResponse(**result.__dict__)


@app.get("/screen", response_model=ScreenResponse, tags=["screen"])
def screen_get(
    elements: list[str] = Query(..., min_length=1,
                                  description="Required metals (repeatable: ?elements=Pt&elements=Ni)"),
    top: int = Query(10, ge=1, le=500),
    model: Literal["etr_emb", "stagea", "ensemble"] = Query("etr_emb"),
    exclude_train: bool = Query(False),
):
    """GET variant of /screen (browser-friendly)."""
    return screen_endpoint(ScreenRequest(elements=elements, top=top,
                                           model=model, exclude_train=exclude_train))
