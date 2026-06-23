# HIDRA: IA para Descoberta de Catalisadores por Redes em Grafos Atômicos

### Predição de catalisadores nanoestruturados para produção de hidrogênio verde via redes neurais de grafos equivariantes

Dissertação: predição de ΔG_H\* para catalisadores de HER (Hydrogen Evolution
Reaction). Compara quatro modelos sobre o mesmo dataset (5860 estruturas) e
split canônico (4220 train / 468 val / 1172 test):

- **ETR + 10 handcrafted** - baseline (descritores eletrônicos/estruturais)
- **SchNet (GNN do zero)** - GNN sem pré-treino, referência sem prior
- **MACE Stage A (GNN)** - fine-tune do MACE-MP-0 pré-treinado (frozen + MLP head)
- **ETR + MACE Embeddings 512** - ETR sobre embeddings invariantes congelados

Reportes seguem prática padrão: **mean ± std sobre 5 seeds** para os modelos
não-determinísticos (SchNet, Stage A); valor único para os determinísticos
(ETR baseline, ETR + MACE emb).

## Reprodução

Gerenciado por [uv](https://docs.astral.sh/uv/) (Python 3.14) + Makefile.

```bash
uv sync                                  # ambiente
make help                                # lista todos os targets disponíveis
make data-download                       # dump bruto Catalysis Hub (~15 min)
make data-build                          # .traj + LMDB + SQLite
make graphs                              # cache PyG + data/splits.json
make pipeline                            # end-to-end: ETR + multi-seed SchNet + Stage A + sweep + report (~6 h GPU)
```

`scripts/01_download.py` usa duas passadas: metadata de ~60k reações `products="H"`,
depois estruturas só das ~9k reações HER limpas (`0.5H2(g) + * -> H*`) que
passam os filtros de metadata. Saídas: `data/raw/`,
`data/processed/her_dataset.traj`, `data/lmdb/her_dataset.lmdb`,
`data/metadata.sqlite`.

## Resultados

5860 estruturas curadas, mesmo split de teste (1172) para todos os modelos.
Fonte auditável: `results/summary.json` + `results/multiseed_summary.md`.
Métricas completas + legendas: `results/dissertacao_metrics.md`.

### Tabela principal (mean ± std, n=5 para não-determinísticos)

| Modelo                          |       R² test       |    MAE (eV)     |    RMSE (eV)    | # Params |   Tipo   |
| ------------------------------- | :-----------------: | :-------------: | :-------------: | :------: | :------: |
| **ETR + MACE embeddings 512**   |     **0.9613**      |     0.0714      |     0.1232      |    -     | Híbrido  |
| **MACE Stage A (frozen + MLP)** | **0.9564 ± 0.0015** | 0.0706 ± 0.0014 | 0.1309 ± 0.0022 |   328K   |   GNN    |
| ETR + 10 features handcrafted   |       0.9341        |     0.0955      |     0.1609      |    -     | Baseline |
| SchNet (do zero)                |   0.9105 ± 0.0511   | 0.0734 ± 0.0145 | 0.1804 ± 0.0573 |   456K   |   GNN    |

> SchNet std 34× maior que Stage A (0.0511 vs 0.0015) → pré-treino MACE-MP-0
> fornece prior estável; SchNet sem prior depende de sorte na inicialização.

### Glossário

| Termo                | Significado                                                               |
| -------------------- | ------------------------------------------------------------------------- |
| **Backbone**         | Corpo principal da GNN (MACE: 4.7M parâmetros, message passing).          |
| **Frozen/Congelado** | Backbone não é treinado - só faz inferência. Apenas a cabeça MLP aprende. |
| **Full/Completo**    | Backbone inteiro é treinado (fine-tune end-to-end).                       |
| **Híbrido**          | GNN como extrator de features + ETR (árvore) como preditor.               |
| **GNN pura**         | A rede neural é o modelo preditivo (proposta da dissertação).             |
| **Embeddings**       | Vetores latentes das camadas internas do MACE (640-dim por átomo).        |

**Ordenação final**: ETR + MACE emb (0.961) > MACE Stage A (0.956) > ETR
handcrafted (0.934) > SchNet (0.911 ± 0.051). Representação pré-treinada
MACE-MP-0 é a fonte do ganho - em performance E em reprodutibilidade.

Nota: o R² do ETR baseline re-ancorou de 0.910 para 0.934 ao fixar o split
canônico por id (`data/splits.json`), agora reproduzível e compartilhado com
todos os modelos.

## Como reproduzir métricas (granular)

```bash
# Targets individuais (ver `make help` para lista completa)
make etr-baseline           # ETR + 10 handcrafted (CPU, ~6 min)
make mace-embeddings        # MACE-MP-0 node embeddings (GPU, ~6 min)
make emb-sweep              # ETR sweep over MACE embeddings
make schnet-multiseed       # SchNet 5 seeds (GPU, ~75 min)
make stagea-multiseed       # MACE Stage A 5 seeds (GPU, ~3.5 h)
make report                 # aggregate + compare + figures
```

Toda métrica fica em `results/runs/{timestamp}_{name}/`:

- `config.yaml` - hyperparams
- `metrics.json` - pacote unificado (R², MAE, RMSE, MDAE, Pearson r, Spearman ρ, sMAPE, MAE meV, % < 43 meV, ...)
- `predictions.parquet` - colunas `[sid, y_true, y_pred, split]`
- `figures/` - parity, residual_hist, residual_vs_pred, cumulative_error
- `env.txt` - git hash + dependency snapshot

Agregada em `results/summary.json`. Multi-seed sumarizado em
`results/multiseed_summary.md` (mean ± std + per-seed tabela).

## Screening de catalisadores (recomendação)

Filtra os 5860 catalisadores curados por composição, prediz ΔG_H, ranqueia
por |ΔG_H_pred| (Sabatier: ótimo HER tem ΔG_H ≈ 0).

```bash
# Top 10 contendo Pt E Ni, ETR + MACE emb (CPU, ~30s)
python scripts/14_screen.py --elements Pt Ni --top 10

# Top 50 contendo Pd, MACE Stage A (GPU)
python scripts/14_screen.py --elements Pd --top 50 --model stagea

# Ensemble ETR + Stage A
python scripts/14_screen.py --elements Cu --top 20 --model ensemble \
    --output results/screen_Cu.csv

# --exclude-train: restringe ao test canônico (1172 IDs) pra evitar
# top picks com dG_pred = dG_dft (ETR memoriza train data)
python scripts/14_screen.py --elements Pt --top 10 --exclude-train

# Via Makefile (defaults: ELEMENTS="Pt Ni" TOP=10 MODEL=etr_emb)
make screen ELEMENTS="Pt Au" TOP=20 MODEL=etr_emb
```

Saída: tabela com `chemical_formula`, `facet`, `site_type`, `dG_pred`,
`delta_G_H` (DFT real), `abs_dG_pred`, `error_vs_dft`, `id`.

Filtro `--elements E1 E2 ...`: retorna estruturas que contêm **todos** os
metais informados (pode ter outros). H é sempre adsorbato, ignorado no filtro.

## Docker (stack completo: API + Web)

Dois services em `docker-compose.yml`:

| Service | Image                       | Porta | Descrição                                   |
| ------- | --------------------------- | ----- | ------------------------------------------- |
| `api`   | `hidra-api:latest` (~7 GB)  | 8000  | FastAPI + ETR + MACE (CPU)                  |
| `web`   | `hidra-web:latest` (~70 MB) | 5173  | Vue 3 + Nginx servindo SPA + proxy `/api/*` |

```bash
make docker-build        # builda os 2 (api ~15min, web ~2min na 1ª vez)
make docker-up           # sobe stack completo em background
make docker-logs         # tail dos dois services
make docker-test         # curl /stats direto + via proxy + SPA HTTP 200
make docker-down         # para tudo + remove network
make docker-shell        # bash no api container

make docker-up-api       # apenas API (sem frontend)
```

Acesso após `make docker-up`:

- Web UI: <http://localhost:5173>
- API direta: <http://localhost:8000>
- API via proxy do web: <http://localhost:5173/api/...> (mesma origem, sem CORS)
- OpenAPI docs: <http://localhost:8000/docs>

Volumes montados no container da API:

- `./data` → `/app/data` (read-only) — SQLite + embeddings MACE
- `./logs/checkpoints` → `/app/logs/checkpoints` (read-only) — MACE Stage A ckpts
- `./data/model_cache` → `/app/data/model_cache` (writable) — pickle ETR persiste entre restarts

GPU **não** é obrigatória. `etr_emb` é CPU-only por design; `stagea` roda CPU
mas ~40× mais lento por estrutura.

## REST API

FastAPI service expõe o screening como HTTP. Documentação OpenAPI auto em
`/docs` (Swagger) e `/redoc`.

```bash
make api-dev               # auto-reload, dev. http://localhost:8000/docs
make api                   # production mode
```

### Endpoints

| Método | Rota                                       | Descrição                                             |
| ------ | ------------------------------------------ | ----------------------------------------------------- |
| GET    | `/`                                        | metadata da API                                       |
| GET    | `/stats`                                   | n_structures, n_test, elementos + modelos disponíveis |
| GET    | `/elements`                                | lista de metais                                       |
| POST   | `/screen`                                  | top-N catalisadores ranqueados                        |
| GET    | `/screen?elements=Pt&top=10&model=etr_emb` | mesmo, GET browser-friendly                           |

### Exemplo POST

```bash
curl -X POST http://localhost:8000/screen \
  -H "Content-Type: application/json" \
  -d '{"elements":["Pt","Ni"],"top":5,"model":"etr_emb","exclude_train":true}'
```

Retorno JSON:

```json
{
  "elements": ["Ni", "Pt"],
  "model": "etr_emb",
  "n_candidates": 24,
  "rows": [
    {"chemical_formula": "...", "dG_pred": 0.001, "abs_dG_pred": 0.001, ...}
  ]
}
```

Backend (`src/screening.py`) cacheia modelos em 2 níveis:

- **Disco** (`data/model_cache/etr_emb.pkl`): ETR treinado 1x, reutilizado entre processos.
  Fingerprint por SHA-1 dos embeddings → invalida automaticamente se mudarem.
- **Processo** (`lru_cache`): modelo carregado 1x por processo, sem disk hit por request.

Custos por modelo (sem cache → com pickle):

| Modelo     | 1ª request                         | Requests subsequentes             | GPU obrigatória?        |
| ---------- | ---------------------------------- | --------------------------------- | ----------------------- |
| `etr_emb`  | ~16s (fit + save) → ~2s (load pkl) | ~10ms                             | NÃO (sklearn CPU)       |
| `stagea`   | ~5s (load ckpt)                    | CPU ~1s/struct • GPU ~25ms/struct | NÃO mas 40× mais rápido |
| `ensemble` | soma dos 2                         | soma dos 2                        | depende do `stagea`     |

Pra deploy CPU-only (HuggingFace Spaces grátis, VPS barato): usar `etr_emb` como
default. `stagea` opcional pra cross-check.

## Pacote de métricas unificado

Todo modelo reporta o mesmo conjunto via `training.evaluate.metrics_from_preds`:

```
r2, mae, rmse                  - base regressão
mdae, max_err                  - robusto + worst-case
pearson_r, spearman_rho        - correlação + ranking
smape                          - MAPE simétrico (estável quando y cruza zero)
mae_meV                        - escala literatura
frac_chem_acc                  - fração com |erro| < 43 meV (chemical accuracy)
```

Toda run salva 4 figuras padronizadas (parity, residual_hist, residual_vs_pred,
cumulative_error) via `RunLogger.log_standard_figures`.

## Embeddings de nó do MACE-MP-0

Extrai embeddings invariantes (L=0) do MACE-MP-0 (medium) para cada estrutura,
em `data/mace_features/{train,val,test}_emb.npz`.

```bash
uv run python scripts/09_extract_mace_embeddings.py          # ~6 min GPU
uv run python scripts/08b_feature_reduction_sweep.py --feature-set emb
```

Pooling: `[emb(H), mean(emb(vizinhos<2.4A))]` → 512 features.

| Estratégia   | # feat | R² test | MAE (eV) | MAE (meV) |
| ------------ | ------ | ------- | -------- | --------- |
| all          | 512    | 0.9613  | 0.0714   | 71        |
| top-100 SHAP | 100    | 0.9611  | 0.0725   | 73        |
| top-50 SHAP  | 50     | 0.9577  | 0.0765   | 77        |
| top-20 SHAP  | 20     | 0.9517  | 0.0846   | 85        |
| top-10 SHAP  | 10     | 0.9397  | 0.0956   | 96        |
| PCA 99% var  | 77     | 0.9600  | 0.0767   | 77        |
| PCA 95% var  | 38     | 0.9568  | 0.0791   | 79        |
| PCA 90% var  | 26     | 0.9562  | 0.0798   | 80        |

SHAP: dimensões do embedding do **H adsorvido** dominam. Sweet spot **top-20**
(0.952, sem perda significativa); top-50 (0.958) se quiser quase o teto.

## Fine-tune do MACE-MP-0 (GNN end-to-end)

Stage A (frozen backbone + MLP head) é o protocolo principal. Multi-seed n=5:

```bash
for seed in 42 1 2 3 4; do
    uv run python scripts/10_finetune_mace.py --freeze-backbone --lr 1e-3 \
        --batch-size 16 --max-epochs 200 --patience 60 --early-stop-monitor val_r2 \
        --seed $seed --run-name mace_ft_stageA_v2_seed${seed}
done
```

| Estágio     |  Backbone  |     Cabeça      |    R² test (n=5)    |      MAE (eV)       |  Params  | Tempo/seed  |
| ----------- | :--------: | :-------------: | :-----------------: | :-----------------: | :------: | :---------: |
| **Stage A** | **Frozen** | **MLP 2-layer** | **0.9564 ± 0.0015** | **0.0706 ± 0.0014** | **328K** | **~40 min** |

Comparação histórica (1 seed cada, runs anteriores):

| Estágio                                   | R² test | Status        |
| ----------------------------------------- | ------- | ------------- |
| Stage A+ (frozen + MLP 3-layer + dropout) | 0.945   | run histórico |
| Stage C (full fine-tune)                  | 0.940   | run histórico |

Stage A vence Stage A+ e Stage C nesse dataset - backbone congelado + cabeça
leve é o sweet spot. Full fine-tune sofre overfit com 5M params em 4220 train.

## Por que SchNet tem variância alta?

Multi-seed SchNet (n=5): R²=0.9105 ± 0.0511, spread 0.123 (worst 0.847,
best 0.970). Spread 34× maior que Stage A (0.004).

Causa: **não-determinismo CUDA + early stopping ruidoso**. Sem prior de
pré-treino, SchNet converge pra mínimos diferentes a cada execução. Mesmo
com `seed_everything(seed)` + monitor=val_r2 mode=max + patience=60, val_r2
epoch-to-epoch noisy → early stop dispara em pico isolado.

Implicação: **transfer learning estabiliza o aprendizado**. Stage A herda
representação MACE-MP-0 (treinada em milhões de estruturas) e só treina a
cabeça MLP → trajetória de treino determinada pelo prior.

## Pipeline

- `ingest.py` - query GraphQL (metadata + busca por id em lote) + reconstrução
  de `ase.Atoms` (sistema `Hstar`)
- `filters.py` - equação HER limpa (`0.5H2(g) + * -> H*`, exclui co-adsorção
  H2S/NH3/...), `delta_G in [-2,2]`, cobertura <= 25%, ligação H-superficie
  em [1,3] A; sítio (top/bridge/hollow) por coordenação do H em 2.4 A
- `features.py` - as 10 features handcrafted (médias geométricas via `mendeleev`)
- `geometry.py` - átomos centrais (2.4 A do H) e vizinhos (1a camada, raios
  covalentes)
- `storage.py` - `.traj`, LMDB (estilo fairchem) e SQLite, com IDs alinhados
- `baseline.py` - ExtraTrees, split 80/20, GridSearchCV 10-fold
- `plots.py` - parity, dG histograma, SHAP importance

Arquitetura GNN/embeddings:

- `data/graph_builder.py` - `ase.Atoms` -> `Data` PyG; arestas PBC-aware via
  `neighbor_list("ijdS")` com `pbc=[True,True,False]` (sem cruzar vacuo em z)
- `data/dataset.py` - `HERDataset(InMemoryDataset)`, cacheia `her_pyg.pt` (idempotente)
- `data/splits.py` - split canônico por id (mesmo para todos modelos), `data/splits.json`
- `data/mace_dataset.py` - dataset para fine-tune MACE
- `models/schnet.py` - `LitSchNet`; forward usa arestas precomputadas (PBC, dispensa
  `torch-cluster`), alvo normalizado, métricas via `torchmetrics`
- `models/mace_features.py` - extrator de escalares/embeddings MACE-MP-0
- `models/mace_finetune.py` - `LitMACEFineTune` (Stage A/A+/C)
- `training/run_logger.py` - `RunLogger`: predictions.parquet com `sid`,
  4 figuras padronizadas, env.txt com git hash
- `training/train.py`, `training/evaluate.py` - loop Lightning + pacote unificado
  de métricas + figuras padrão
- `analysis/comparison.py` - whitelist 4 modelos + tabela cross-model
- `analysis/multiseed.py` - agregação mean ± std por grupo de seeds

Notebooks (`01` explora, `02` filtros, `04` grafos, `07` mace smoke,
`08` mace features EDA) importam de `src/`.

### Decisões de implementação

- Vizinhos e CN usam lista de vizinhança por raios covalentes (o cutoff de
  2.4 A define só os átomos centrais; é curto demais para a 1a camada
  metal-metal).
- `Out_e` = `nvalence` do mendeleev (consistente por grupo); `Nd`/`Np` =
  contagem total de eletrons d/p.
- Contagem final do dataset = 5860 estruturas, conforme filtros aplicados
  sobre o snapshot disponível do Catalysis Hub (a API limita páginas a 200
  itens e o cursor padrão repete/pula registros).
- Multi-seed (n=5) reportado para modelos não-determinísticos (SchNet, Stage A).
  ETR (sklearn) determinístico per seed=42.

### GPU

Treino e extração de features MACE/SchNet exigem CUDA (RTX 5060 Ti 16GB).
`scripts/05_train_schnet.py` falha ruidosamente se CUDA ausente, usa
`precision="16-mixed"` e reporta VRAM pico. SchNet com batch=32 fica < 1 GB
de VRAM. Stage A: ~1 GB VRAM.

## Estrutura

```
src/           ingest, filters, features, geometry, storage, baseline, plots, dataset
               data/ (graph_builder, dataset, splits, mace_dataset),
               models/ (schnet, mace_features, mace_finetune),
               training/ (run_logger, train, evaluate, config),
               analysis/ (feature_eda, feature_importance, feature_reduction,
                          comparison, multiseed)
notebooks/     01 explora, 02 filtros, 03 baseline (ETR), 04 grafos,
               07 mace smoke, 08 mace features EDA
scripts/       01_download, 02_build_dataset, 03_train_baseline,
               04_build_graphs, 05_train_schnet, 06_compare,
               07_extract_mace_features, 08b_feature_reduction_sweep,
               09_extract_mace_embeddings, 10_finetune_mace,
               11_figures_dissertacao, 12_aggregate_multiseed
data/          raw, processed (.traj + her_pyg.pt), lmdb, mace_features/, metadata.sqlite
results/       runs/{ts}_{name}/, summary.json, multiseed_summary.md,
               comparison_table.md, dissertacao_metrics.md, figures/
```
