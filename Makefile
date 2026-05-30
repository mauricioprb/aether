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

# ── End-to-end ───────────────────────────────────────────────────────────────
pipeline: etr-baseline emb-sweep schnet-multiseed stagea-multiseed report ## Full pipeline (GPU, ~6 h).

clean-checkpoints:          ## Remove training checkpoints (regenerable).
	rm -rf logs/checkpoints/*

clean-results:              ## Remove all run dirs (KEEPS summary + tables + figs).
	rm -rf results/runs/*

.PHONY: help data-download data-build graphs etr-baseline schnet-multiseed \
         mace-features mace-embeddings stagea-multiseed emb-sweep aggregate \
         compare figures report pipeline clean-checkpoints clean-results
