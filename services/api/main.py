from __future__ import annotations

import logging
import os
import secrets
from functools import lru_cache
from statistics import median
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from screening import (
    SABATIER_CORRECTION_EV,
    ModelName,
    predict_atoms,
    available_elements,
    filter_candidates,
    load_full_dataset,
    predictable_mask,
    parse_metal_elements,
    screen,
)

from geometry import MIN_DISTANCE_RATIO, min_distance_ratio

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
    dg_correction: float = Field(default=0.24, ge=-1.0, le=1.0,
                                   description="ΔG_H = ΔE_H + corr (eV); "
                                               "0.24 = Nørskov 2005, 0 = raw ΔE_H")


class CandidateRow(BaseModel):
    id: str
    chemical_formula: str
    composition: str
    facet: str
    site_type: str
    coverage: float | None = None
    delta_G_H: float
    dE_pred: float
    dG_pred: float
    dG_dft: float
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
    dg_correction: float
    rows: list[CandidateRow]
    pool_dG_pred: list[float] = []


class ElementStat(BaseModel):
    symbol: str
    n: int
    median_dG: float
    best_abs_dG: float


class ElementStatsResponse(BaseModel):
    elements: list[ElementStat]
    min_sample: int
    chemical_accuracy_eV: float = CHEM_ACCURACY_EV


class CountResponse(BaseModel):
    elements: list[str]
    exclude_train: bool
    n_candidates: int


class StatsResponse(BaseModel):
    n_structures: int
    # Quantas o app consegue de fato triar: o cache de embeddings cobre so as
    # curadas, e as demais quebrariam o etr_emb. Sem este campo a interface
    # prometia 7238 e entregava 5860.
    n_screenable: int
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


def _kind_of(name: str) -> Literal["baseline", "gnn", "hybrid"]:
    """Categoria do run. Loga a falha em vez de cair num default silencioso: era
    assim que `mace_ft_stageA_v2_seed3` aparecia como "hybrid" na interface."""
    kind = KIND_BY_NAME.get(name)
    if kind is None:
        logger.warning("run sem categoria mapeada: %s", name)
        return "hybrid"
    return kind


@app.get("/", summary="API root", tags=["meta"])
def root():
    return {
        "name": app.title,
        "version": app.version,
        "docs": "/docs",
        "endpoints": ["/stats", "/elements", "/elements/stats", "/count", "/screen"],
    }


@app.get("/stats", response_model=StatsResponse, tags=["meta"])
def stats():
    """Dataset + service stats."""
    from screening import load_canonical_test_ids

    df = load_full_dataset()
    return StatsResponse(
        n_structures=len(df),
        n_screenable=int(predictable_mask(df, "ensemble").sum()),
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
            kind=_kind_of(entry.get("name", "")),
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
                        model=req.model, exclude_train=req.exclude_train,
                        dg_correction=req.dg_correction)
    except FileNotFoundError as exc:
        logger.error("screen artifact missing: %s", exc)
        raise HTTPException(status_code=503, detail="service temporarily unavailable")
    except ValueError as exc:
        logger.warning("bad screen request: %s", exc)
        raise HTTPException(status_code=400, detail="invalid screening parameters")
    return ScreenResponse(**result.__dict__)


# Limites do /predict. A borda aceita arquivo de terceiro, entao tudo que entra
# e limitado antes de virar objeto: 1 MB cobre CIF de supercelula com folga, e
# 512 atomos e o teto do que o Stage A processa em tempo de requisicao.
MAX_UPLOAD_BYTES = 1 << 20
MAX_ATOMS = 512
# Lista fechada de proposito: o ASE adivinha formato e tem leitor pra dezenas
# deles, e nada disso deve ficar exposto a arquivo de terceiro.
StructureFormat = Literal["cif", "extxyz", "vasp"]


class PredictResponse(BaseModel):
    chemical_formula: str
    n_atoms: int
    model: ModelName
    dE_pred: float = Field(description="Energia eletronica de adsorcao prevista (eV)")
    dG_pred: float = Field(description="dE_pred + correcao de Sabatier (eV)")
    abs_dG_pred: float = Field(description="Distancia do otimo de Sabatier (eV)")
    elements_in_training: bool = Field(
        description="False quando algum elemento esta fora do conjunto de treino: "
                    "a predicao e extrapolacao e nao herda o erro reportado.",
    )


def _parse_structure(raw: bytes, fmt: StructureFormat):
    """Bytes -> Atoms, recusando tudo que o modelo nao sabe tratar."""
    from io import StringIO

    from ase.io import read

    try:
        atoms = read(StringIO(raw.decode("utf-8")), format=fmt)
    except Exception as exc:  # ASE levanta de tudo conforme o leitor
        logger.warning("parse falhou (%s): %s", fmt, exc)
        raise HTTPException(status_code=400, detail="arquivo ilegivel para o formato informado")

    if len(atoms) > MAX_ATOMS:
        raise HTTPException(status_code=413, detail=f"estrutura com mais de {MAX_ATOMS} atomos")
    numbers = set(atoms.numbers.tolist())
    if 1 not in numbers:
        raise HTTPException(
            status_code=400,
            detail="nenhum H na estrutura: o modelo preve adsorcao de H, o adsorbato precisa estar presente",
        )
    if numbers == {1}:
        raise HTTPException(status_code=400, detail="estrutura so de H, sem superficie")
    if max(numbers) > 89:
        raise HTTPException(status_code=400, detail="elemento fora da tabela do modelo (Z > 89)")

    # Sem isto o modelo responde a geometria impossivel com a mesma confianca
    # com que acerta o Pt: H a 0.8 A de um atomo de Pt devolve +13.6 eV sem
    # nenhum aviso. Extrapolar e legitimo; extrapolar calado, nao.
    ratio = min_distance_ratio(atoms)
    if ratio < MIN_DISTANCE_RATIO:
        raise HTTPException(
            status_code=400,
            detail=f"atomos sobrepostos (menor distancia = {ratio:.2f} do raio covalente "
                   f"somado, minimo {MIN_DISTANCE_RATIO}); relaxe a estrutura antes de prever",
        )
    return atoms


@app.post("/predict", response_model=PredictResponse, tags=["screen"],
          dependencies=[Depends(require_api_key)])
async def predict_endpoint(
    request: Request,
    format: StructureFormat = Query(description="Formato do corpo da requisicao"),
    model: ModelName = Query(default="ensemble"),
):
    """Preve dG_H de uma estrutura que nao esta no dataset.

    E o que separa a triagem de uma consulta: aqui o usuario traz a propria
    estrutura, e a predicao sai dos mesmos pesos do /screen. O corpo da
    requisicao e o arquivo cru (sem multipart):

        curl -X POST "$API/predict?format=cif&model=ensemble" --data-binary @slab.cif
    """
    raw = await request.body()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"arquivo acima de {MAX_UPLOAD_BYTES} bytes")
    atoms = _parse_structure(raw, format)

    try:
        dE = float(predict_atoms([atoms], model)[0])
    except FileNotFoundError as exc:
        logger.error("predict artifact missing: %s", exc)
        raise HTTPException(status_code=503, detail="service temporarily unavailable")

    dG = dE + SABATIER_CORRECTION_EV
    known = set(available_elements())
    metals = parse_metal_elements(atoms.get_chemical_formula())
    return PredictResponse(
        chemical_formula=atoms.get_chemical_formula(),
        n_atoms=len(atoms),
        model=model,
        dE_pred=dE,
        dG_pred=dG,
        abs_dG_pred=abs(dG),
        elements_in_training=bool(metals) and metals.issubset(known),
    )


# Amostra minima para a mediana significar algo. Abaixo disto o elemento sai da
# escala de cor: C tem 1 estrutura e O tem 3, e mediana de n=1 e ruido colorido.
MIN_SAMPLE = 10


@lru_cache(maxsize=1)
def _element_stats() -> list[ElementStat]:
    """Mediana de ΔG_H por elemento, sobre o rotulo de referencia.

    Nao roda modelo: e agregacao sobre a coluna ja depositada, entao responde na
    hora e nao muda entre requisicoes.
    """
    df = load_full_dataset()
    buckets: dict[str, list[float]] = {}
    for formula, dE in zip(df["chemical_formula"], df["delta_G_H"], strict=True):
        dG = dE + SABATIER_CORRECTION_EV
        for el in parse_metal_elements(formula):
            buckets.setdefault(el, []).append(dG)
    out = [
        ElementStat(
            symbol=el,
            n=len(vals),
            median_dG=float(median(vals)),
            best_abs_dG=float(min(abs(v) for v in vals)),
        )
        for el, vals in buckets.items()
    ]
    return sorted(out, key=lambda e: e.symbol)


@app.get("/elements/stats", response_model=ElementStatsResponse, tags=["meta"])
def element_stats():
    """Por elemento: quantas estruturas e a mediana de ΔG_H."""
    return ElementStatsResponse(elements=_element_stats(), min_sample=MIN_SAMPLE)


@app.get("/count", response_model=CountResponse, tags=["screen"],
         dependencies=[Depends(require_api_key)])
def count_endpoint(
    elements: list[str] = Query(..., min_length=1, max_length=20),
    exclude_train: bool = Query(False),
    model: Literal["etr_emb", "stagea", "ensemble"] = Query("ensemble"),
):
    """Quantas estruturas contem todos os elementos pedidos.

    So o filtro do pandas: nao roda modelo, entao serve para dizer o tamanho do
    resultado antes de o usuario submeter a triagem.
    """
    required = {e.capitalize() for e in elements}
    df = load_full_dataset()
    cand = filter_candidates(df, required, exclude_train=exclude_train)
    # mesmo filtro do /screen: contar o que o modelo nao consegue prever faria o
    # botao prometer um numero que a triagem nao entrega
    n = int(predictable_mask(cand, model).sum())
    return CountResponse(elements=sorted(required), exclude_train=exclude_train,
                          n_candidates=n)


@app.get("/screen", response_model=ScreenResponse, tags=["screen"],
         dependencies=[Depends(require_api_key)])
def screen_get(
    elements: list[str] = Query(..., min_length=1, max_length=20,
                                  description="Required metals (repeatable: ?elements=Pt&elements=Ni)"),
    top: int = Query(10, ge=1, le=500),
    model: Literal["etr_emb", "stagea", "ensemble"] = Query("etr_emb"),
    exclude_train: bool = Query(False),
    dg_correction: float = Query(0.24, ge=-1.0, le=1.0),
):
    """GET variant of /screen (browser-friendly)."""
    return screen_endpoint(ScreenRequest(elements=elements, top=top,
                                           model=model, exclude_train=exclude_train,
                                           dg_correction=dg_correction))
