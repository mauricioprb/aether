.DEFAULT_GOAL := help

# Use uv-managed venv directly so PYTHONPATH=src resolves source imports.
export PYTHONPATH := src
PY := .venv/bin/python

SEEDS := 42 1 2 3 4

help:                       ## Show this help.
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[1m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ── Data pipeline ────────────────────────────────────────────────────────────
data-download:              ## Download raw Catalysis Hub dump (~15 min).
	$(PY) scripts/01_download.py

data-build:                 ## Build .traj + LMDB + SQLite from raw dump.
	$(PY) scripts/02_build_dataset.py

graphs:                     ## Build PyG cache + canonical splits.json.
	$(PY) scripts/04_build_graphs.py

# ── Model training ───────────────────────────────────────────────────────────
etr-baseline:               ## Train ETR + 10 handcrafted (CPU, ~6 min).
	$(PY) scripts/03_train_baseline.py

schnet-multiseed:           ## Train SchNet 5 seeds (GPU, ~75 min).
	@for s in $(SEEDS); do \
		echo ">>> SchNet seed=$$s"; \
		rm -rf logs/checkpoints/schnet_v2_seed$$s; \
		$(PY) scripts/05_train_schnet.py \
			--max-epochs 300 --batch-size 32 --lr 1e-3 --cutoff 6.0 \
			--patience 60 --early-stop-monitor val_r2 \
			--seed $$s --run-name schnet_v2_seed$$s; \
	done

mace-features:              ## Extract MACE-MP-0 scalar features (GPU, ~10 min).
	$(PY) scripts/07_extract_mace_features.py

mace-embeddings:            ## Extract MACE-MP-0 node embeddings (GPU, ~6 min).
	$(PY) scripts/09_extract_mace_embeddings.py

stagea-multiseed:           ## Fine-tune MACE Stage A 5 seeds (GPU, ~3.5 h).
	@for s in $(SEEDS); do \
		echo ">>> Stage A seed=$$s"; \
		rm -rf logs/checkpoints/mace_ft_stageA_v2_seed$$s; \
		$(PY) scripts/10_finetune_mace.py \
			--freeze-backbone --lr 1e-3 \
			--batch-size 16 --max-epochs 200 \
			--patience 60 --early-stop-monitor val_r2 \
			--seed $$s --run-name mace_ft_stageA_v2_seed$$s; \
	done

emb-sweep:                  ## ETR sweep over MACE embeddings (top-K SHAP + PCA).
	$(PY) scripts/08b_feature_reduction_sweep.py --feature-set emb

# ── Reporting ────────────────────────────────────────────────────────────────
aggregate:                  ## Aggregate multi-seed runs into mean ± std table.
	$(PY) scripts/12_aggregate_multiseed.py

compare:                    ## Generate cross-model comparison + diagnostic figs.
	$(PY) scripts/06_compare.py

figures:                    ## Generate dissertation figures (fig1-5).
	$(PY) scripts/11_figures_dissertacao.py

report: aggregate compare figures ## Run all reporting steps in order.

# ── Screening (catalyst recommendation) ──────────────────────────────────────
screen-help:                ## Show screen.py usage.
	$(PY) scripts/14_screen.py --help

# Example: make screen ELEMENTS="Pt Ni" TOP=10 MODEL=etr_emb
ELEMENTS ?= Pt Ni
TOP ?= 10
MODEL ?= etr_emb
screen:                     ## Rank catalysts by |ΔG_H_pred|. Vars: ELEMENTS, TOP, MODEL.
	$(PY) scripts/14_screen.py --elements $(ELEMENTS) --top $(TOP) --model $(MODEL)

# ── REST API ─────────────────────────────────────────────────────────────────
API_HOST ?= 0.0.0.0
API_PORT ?= 8000
api-dev:                    ## Run REST API in dev mode (auto-reload). Open http://localhost:$(API_PORT)/docs
	.venv/bin/uvicorn app.api:app --host $(API_HOST) --port $(API_PORT) --reload

api:                        ## Run REST API in production mode.
	.venv/bin/uvicorn app.api:app --host $(API_HOST) --port $(API_PORT) --workers 1

# ── Docker ──────────────────────────────────────────────────────────────────
docker-build:               ## Build CPU-only API image (aether-api:latest, ~3 GB).
	docker compose build api

docker-up:                  ## Start API container in background (port 8000).
	docker compose up -d api

docker-logs:                ## Tail API container logs.
	docker compose logs -f api

docker-down:                ## Stop + remove containers.
	docker compose down

docker-shell:               ## Open shell inside running api container.
	docker compose exec api bash

docker-test:                ## Smoke test running API (curl /stats + /screen).
	@curl -fsS http://localhost:$(API_PORT)/stats | head -1; echo
	@curl -fsS -X POST http://localhost:$(API_PORT)/screen \
		-H "Content-Type: application/json" \
		-d '{"elements":["Pt"],"top":3,"model":"etr_emb","exclude_train":true}' | head -1; echo

# ── End-to-end ───────────────────────────────────────────────────────────────
pipeline: etr-baseline emb-sweep schnet-multiseed stagea-multiseed report ## Full pipeline (GPU, ~6 h).

clean-checkpoints:          ## Remove training checkpoints (regenerable).
	rm -rf logs/checkpoints/*

clean-results:              ## Remove all run dirs (KEEPS summary + tables + figs).
	rm -rf results/runs/*

.PHONY: help data-download data-build graphs etr-baseline schnet-multiseed \
         mace-features mace-embeddings stagea-multiseed emb-sweep aggregate \
         compare figures report pipeline screen screen-help api api-dev \
         docker-build docker-up docker-logs docker-down docker-shell docker-test \
         clean-checkpoints clean-results
