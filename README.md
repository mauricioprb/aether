# Predição de catalisadores nanoestruturados para produção de hidrogênio verde via redes neurais de grafos equivariantes

Dissertação: predição de catalisadores de HER (Hydrogen Evolution Reaction).
Etapa 1 = baseline Extra Trees sobre 10 features (réplica Wang et al., npj Comput.
Mater. (2025) 11:111, [DOI 10.1038/s41524-025-01607-4](https://doi.org/10.1038/s41524-025-01607-4)).
Etapa 2 = SchNet (GNN) sobre os mesmos dados, com comparação honesta (mesmo split).

## Reprodução (5 comandos)

Gerenciado por [uv](https://docs.astral.sh/uv/) (Python 3.14):

```bash
uv sync                                          # 1. ambiente
uv run python scripts/01_download.py             # 2. dump bruto do Catalysis Hub
uv run python scripts/02_build_dataset.py        # 3. .traj + LMDB + SQLite (filtros + features)
uv run python scripts/03_train_baseline.py       # 4. treina ETR, métricas, figuras
uv run jupyter lab                               # 5. notebooks exploratórios
```

`01_download.py` usa duas passadas: metadata de ~60k reações `products="H"`,
depois estruturas só das ~9k reações HER limpas (`0.5H2(g) + * -> H*`) que
passam os filtros de metadata (~15min). Saídas: `data/raw/`,
`data/processed/her_dataset.traj`, `data/lmdb/her_dataset.lmdb`,
`data/metadata.sqlite`, `data/figures/`.

## Resultados

5860 estruturas curadas, mesmo split de teste (1172) para todos os modelos.
Fonte auditável: `results/summary.json` (gerado pelos scripts, sem retreinar).
Métricas completas e legendas: `results/dissertação_metrics.md`.

### Tabela principal

| Modelo                               |  R² test  |  MAE  |  RMSE   | # Params |   Tipo   |
| ------------------------------------ | :-------: | :---: | :-----: | :------: | :------: |
| **ETR + MACE embeddings (frozen)**   | **0.961** | 0.071 |  0.123  |    -     | Híbrido  |
| ETR + MACE top-50 SHAP               |   0.958   | 0.077 |  0.129  |    -     | Híbrido  |
| ETR + MACE PCA 90% (26 comp)         |   0.956   | 0.080 |  0.131  |    -     | Híbrido  |
| ETR + MACE top-20 SHAP               |   0.952   | 0.085 |  0.138  |    -     | Híbrido  |
| **MACE Stage A (frozen + MLP)**      | **0.947** | 0.083 |  0.144  |   328K   |   GNN    |
| MACE Stage A+ (frozen + MLP 3-layer) |   0.945   | 0.072 |  0.148  |   919K   |   GNN    |
| MACE Stage C (full fine-tune)        |   0.940   | 0.084 |  0.153  |   5.0M   |   GNN    |
| ETR + MACE top-10 SHAP               |   0.940   | 0.096 |  0.154  |    -     | Híbrido  |
| ETR (10 features handcrafted)        |   0.934   | 0.096 |  0.161  |    -     | Baseline |
| _Paper Wang et al. (2025)_           |  _0.922_  |   -   | _0.186_ |    -     |   Ref.   |
| SchNet (do zero)                     |   0.908   | 0.060 |  0.190  |   456K   |   GNN    |
| ETR + MACE escalares (7 feat)        |   0.864   | 0.158 |  0.231  |    -     | Baseline |

= GNN (rede neural). = árvore (ETR). = referência externa.

### Glossário

| Termo                | Significado                                                               |
| -------------------- | ------------------------------------------------------------------------- |
| **Backbone**         | Corpo principal da GNN (MACE: 4.7M parâmetros, message passing).          |
| **Frozen/Congelado** | Backbone não é treinado - só faz inferência. Apenas a cabeça MLP aprende. |
| **Full/Completo**    | Backbone inteiro é treinado (fine-tune end-to-end).                       |
| **Híbrido**          | GNN como extrator de features + ETR (árvore) como preditor.               |
| **GNN pura**         | A rede neural é o modelo preditivo (proposta da dissertação).             |
| **Embeddings**       | Vetores latentes das camadas internas do MACE (640-dim por átomo).        |

Ordenação final: **embeddings MACE > GNN fine-tune > handcrafted > SchNet > escalares MACE**.
A GNN pré-treinada com fine-tune (Stage A, R²=0.947) supera o baseline handcrafted
(0.934) e o paper de referência (0.922), confirmando a hipótese da dissertação.

Nota: o R² do ETR re-ancorou de 0.910 (Etapa 1, ordem de linha pre-dedupe) para
0.934 ao fixar o split canônico por id (`data/splits.json`), agora reproduzível e
compartilhado com todos os modelos.

## Como reproduzir métricas

```bash
uv sync
# Etapa 1
uv run python scripts/01_download.py
uv run python scripts/02_build_dataset.py
uv run python scripts/03_train_baseline.py     # ETR -> results/runs/{ts}_etr_baseline
# Etapa 2 (GPU obrigatória)
uv run python scripts/04_build_graphs.py        # grafos PyG + data/splits.json
uv run python scripts/05_train_schnet.py --smoke           # checagem rapida (<2min)
uv run python scripts/05_train_schnet.py --run-name schnet_baseline
uv run python scripts/06_compare.py             # tabela + figuras (sem GPU, <30s)
# Etapa 3 - Tarefa 2 / Fase A.5 / Fase B
uv run python scripts/07_extract_mace_features.py            # escalares MACE (GPU)
uv run python scripts/08b_feature_reduction_sweep.py         # sweep escalares (Fase A.5)
uv run python scripts/09_extract_mace_embeddings.py          # embeddings 512-dim (GPU)
uv run python scripts/08b_feature_reduction_sweep.py --feature-set emb   # sweep embeddings (Fase B)
# Etapa 4 - Fine-tune MACE (GPU obrigatória)
uv run python scripts/10_finetune_mace.py --freeze-backbone --lr 1e-3 \
    --batch-size 16 --max-epochs 100 --patience 30 \
    --run-name mace_ft_stageA                    # Stage A: frozen backbone + MLP (~12 min)
uv run python scripts/10_finetune_mace.py --freeze-backbone --lr 1e-3 \
    --hidden-dim 512 --head-layers 3 --dropout 0.1 \
    --batch-size 16 --max-epochs 150 --run-name mace_ft_stageA_plus  # Stage A+ (~33 min)
uv run python scripts/10_finetune_mace.py --no-freeze-backbone --lr 1e-5 \
    --batch-size 4 --max-epochs 50 --patience 15 \
    --run-name mace_ft_stageC                    # Stage C: full fine-tune (~48 min)
# Figuras da dissertação
uv run python scripts/11_figures_dissertação.py  # 5 figuras em results/figures/ (~2 min)
```

Toda metrica fica em `results/runs/{timestamp}_{name}/` (config.yaml, metrics.json,
predictions.parquet, figures/, env.txt) e e agregada em `results/summary.json`.

## Etapa 3 - Tarefa 2: extração de features MACE-MP-0

Extrai descritores do potencial de fundação MACE-MP-0 (medium) para cada
estrutura, em `data/mace_features/{train,val,test}.npz`, alinhados ao split
canônico (`splits.json`; val recortado do train como no SchNet).

```bash
uv run python scripts/07_extract_mace_features.py   # ~10 min na RTX 5060 Ti
```

7 features escalares por estrutura (energias per-átom do MACE):
`mace_E_total`, `mace_E_per_atom`, `mace_E_H`, `mace_E_neighbors_mean`,
`mace_E_neighbors_min`, `mace_E_surface_mean`, `mace_n_neighbors`. Embeddings de
no ficam adiados (a API expoe energias per-átom facilmente; embeddings exigem
forward manual) - escalares já bastam para a EDA da Fase A.5. Idempotente
(`--force` para reextrair). Stats em `results/runs/{ts}_mace_extraction/feature_stats.json`.
Validação + smoke: `notebooks/07_mace_smoke.ipynb`.

## Etapa 3 - Fase A.5: EDA e redução de features MACE

EDA, importância (SHAP/permutation/RFE) e sweep de redução sobre as 7 features
escalares. `notebooks/08_mace_features_eda.ipynb` +
`scripts/08b_feature_reduction_sweep.py` (gera 8 runs + gráfico R² vs #features).

```bash
uv run python scripts/08b_feature_reduction_sweep.py   # ~6 min, sem GPU
```

Sweep (ETR, split canônico, vs baselines ETR-handcrafted=0.934 e SchNet=0.908):

| Estratégia  | # feat | R² test |
| ----------- | ------ | ------- |
| all / RFE   | 7      | 0.864   |
| top-5 SHAP  | 5      | 0.863   |
| top-3 SHAP  | 3      | 0.807   |
| PCA 99% var | 5      | 0.846   |
| PCA 95% var | 4      | 0.837   |
| PCA 90% var | 3      | 0.816   |
| composição  | 16     | 0.863   |

SHAP: `mace_E_H` (0.23) >> `E_neighbors_mean` (0.15) ~ `E_neighbors_min` (0.14)

> `n_neighbors` (0.06) >> energias globais (`E_total`/`E_per_atom`/`E_surface_mean`,
> ~0.02 cada). A energia de ligação do H e suas vizinhanças carregam quase toda a
> informação.

### Achado e decisão (Fase B)

- **Número final de features: 5** (`E_H`, `E_neighbors_mean`, `E_neighbors_min`,
  `n_neighbors` + 1). Estratégia vencedora: **top-5 por SHAP** (R²=0.863, contra
  0.864 com todas - perda desprezível cortando as 3 energias globais redundantes).
  PCA nunca supera top-K; composição (analogia ao phi) não gerou nada melhor.
- **Porem o teto das 7 escalares (0.864) fica abaixo dos baselines** (ETR-handcrafted
  0.934, SchNet 0.908). Energias per-átom do MACE-MP-0, sozinhas, são menos
  preditivas que os descritores eletrônicos handcrafted para HER.
- **Decisao: vale ir para a Fase B.** Como ate o conjunto completo de escalares
  não alcança os baselines (features complementares, não redundantes - heurística
  do prompt), o ganho deve vir dos **embeddings de no do MACE** (adiados na
  Tarefa 2) e/ou fine-tune. Próximo passo concreto: extrair os embeddings
  invariantes do MACE e repetir esta EDA no espaço ~300-dim antes de decidir
  fine-tune.
- **Validação cross-model (MLP)**: top-5 (0.817) >= all (0.812) > top-3 (0.783),
  mesma ordem do ETR - o achado "5 features bastam, 3 globais são redundantes"
  não e especifico ao tipo de modelo.

## Etapa 3 - Fase B: embeddings de no do MACE

A Fase A.5 mostrou que os escalares não bastam, então extraimos os **embeddings
de no invariantes (L=0) do MACE-MP-0** e repetimos a EDA/sweep no espaço 512-dim.

```bash
uv run python scripts/09_extract_mace_embeddings.py          # ~6 min GPU -> *_emb.npz
uv run python scripts/08b_feature_reduction_sweep.py --feature-set emb
```

Pooling: `[emb(H), mean(emb(vizinhos<2.4A))]` dos descritores invariantes ->
512 features. Agora samples/features ~9, então a redução importa de fato.

| Estratégia   | # feat | R² test |
| ------------ | ------ | ------- |
| all          | 512    | 0.961   |
| top-100 SHAP | 100    | 0.961   |
| top-50 SHAP  | 50     | 0.958   |
| top-20 SHAP  | 20     | 0.952   |
| top-10 SHAP  | 10     | 0.940   |
| PCA 99% var  | 77     | 0.960   |
| PCA 95% var  | 38     | 0.957   |
| PCA 90% var  | 26     | 0.956   |

SHAP: as dimensões do embedding do **H adsorvido** (`embH_*`) dominam - coerente
com `mace_E_H` ser a escalar mais importante.

### Achado e decisão

- **Os embeddings MACE batem todos os baselines** (0.961 vs handcrafted 0.934,
  SchNet 0.908, escalares 0.864). Mesmo **top-10 dims (0.940) > 10 handcrafted**.
- **Número recomendado: top-20 (0.952)** - sweet spot interpretabilidade/acurácia;
  top-50 (0.958) se quiser quase o teto. PCA 90% (26 comp, 0.956) também ótimo.
- **Decisao: representação MACE pré-treinada e o caminho.** A cabeça leve (ETR)
  sobre embeddings frozen já e o melhor modelo do projeto, sem fine-tune. Próximo
  passo: fine-tune end-to-end do MACE para obter uma GNN pura como preditor.

## Etapa 4 - Fine-tune do MACE-MP-0 (GNN end-to-end)

Fine-tune do MACE-MP-0 para predizer ΔG_H\* diretamente, sem ETR e sem features
manuais. Três estágios progressivos:

```bash
# Stage A: backbone congelado + MLP 2-layer (recomendado)
uv run python scripts/10_finetune_mace.py --freeze-backbone --lr 1e-3 \
    --batch-size 16 --max-epochs 100 --patience 30 --run-name mace_ft_stageA

# Stage A+: backbone congelado + MLP 3-layer + dropout
uv run python scripts/10_finetune_mace.py --freeze-backbone --lr 1e-3 \
    --hidden-dim 512 --head-layers 3 --dropout 0.1 \
    --batch-size 16 --max-epochs 150 --run-name mace_ft_stageA_plus

# Stage C: fine-tune completo (backbone + cabeça)
uv run python scripts/10_finetune_mace.py --no-freeze-backbone --lr 1e-5 \
    --batch-size 4 --max-epochs 50 --patience 15 --run-name mace_ft_stageC
```

| Estágio  | Backbone |        Cabeça         |  R² test  |  MAE  | RMSE  | Params | Tempo  |
| -------- | :------: | :-------------------: | :-------: | :---: | :---: | :----: | :----: |
| Stage A  |  Frozen  |      MLP 2-layer      | **0.947** | 0.083 | 0.144 |  328K  | 12 min |
| Stage A+ |  Frozen  | MLP 3-layer + dropout |   0.945   | 0.072 | 0.148 |  919K  | 33 min |
| Stage C  |   Full   |      MLP 2-layer      |   0.940   | 0.084 | 0.153 |  5.0M  | 48 min |

**Stage A é o melhor modelo GNN puro**: backbone congelado (transfer learning
do pré-treino no OCP) + MLP head treinada. Supera o paper (0.922) e o baseline
handcrafted (0.934). O gap para o ETR+frozen (0.961) é atribuído à maior
capacidade de árvores de decisão em espaços de alta dimensionalidade com
poucos exemplos.

## Pipeline

- `ingest.py` - query GraphQL (metadata + busca por id em lote) + reconstrução
  de `ase.Atoms` (sistema `Hstar`)
- `filters.py` - equação HER limpa (`0.5H2(g) + * -> H*`, exclui co-adsorção
  H2S/NH3/...), `delta_G in [-2,2]`, cobertura <= 25%, ligação H-superficie
  em [1,3] A; sítio (top/bridge/hollow) por coordenação do H em 2.4 A
- `features.py` - as 10 features (medias geométricas via `mendeleev`)
- `geometry.py` - átomos centrais (2.4 A do H) e vizinhos (1a camada, raios
  covalentes)
- `storage.py` - `.traj`, LMDB (estilo fairchem) e SQLite, com IDs alinhados
- `baseline.py` - ExtraTrees, split 80/20, GridSearchCV 10-fold, R2/MAE/MSE/RMSE
- `plots.py` - Fig. 3b (histograma), 4f (parity), 6d (SHAP)

Etapa 2 (GNN), so arquivos novos, Etapa 1 intacta:

- `data/graph_builder.py` - `ase.Atoms` -> `Data` PyG; arestas PBC-aware via
  `neighbor_list("ijdS")` com `pbc=[True,True,False]` (sem cruzar vacuo em z)
- `data/dataset.py` - `HERDataset(InMemoryDataset)`, cacheia `her_pyg.pt` (idempotente)
- `data/splits.py` - split canônico por id (mesmo do ETR), `data/splits.json`
- `models/schnet.py` - `LitSchNet`; forward usa arestas precomputadas (PBC, dispensa
  `torch-cluster`), alvo normalizado, métricas via `torchmetrics`
- `training/run_logger.py` - `RunLogger`: cria `results/runs/{...}` e popula
- `training/train.py`, `training/evaluate.py` - loop Lightning + métricas/parity

Notebooks (`01` explora, `02` filtros/features, `03` baseline, `04` grafos)
importam de `src/her_gnn`, sem duplicar logica.

### Desvios documentados do paper

- Vizinhos e CN usam lista de vizinhanca por raios covalentes (o cutoff de
  2.4 A do paper define so os átomos centrais; e curto demais para a 1a camada
  metal-metal).
- `Out_e` = `nvalence` do mendeleev (consistente por grupo); `Nd`/`Np` =
  contagem total de eletrons d/p.
- Contagem final (5860) fica abaixo dos 10.855 do paper: a API do Catalysis Hub
  limita paginas a 200 itens e o cursor padrao repete/pula registros; o snapshot
  de 2025 do paper não e reproduzível hoje. A paginacao de metadata usa
  `order:"id"` e deduplica por id. O alvo principal (R2 >= 0.90) foi atingido.

### GPU

Etapa 2 exige CUDA (treino na RTX 5060 Ti 16GB). `scripts/05_train_schnet.py`
falha ruidosamente se CUDA ausente, usa `precision="16-mixed"` e reporta VRAM
pico. SchNet com batch=32 fica < 1 GB de VRAM neste dataset.

## Estrutura

```
src/           ingest, filters, features, geometry, storage, baseline, plots, dataset
               data/ (graph_builder, dataset, splits, mace_dataset),
               models/ (schnet, mace_features, mace_finetune),
               training/ (run_logger, train, evaluate),
               analysis/ (feature_eda, feature_importance, feature_reduction)
notebooks/     01 explora, 02 filtros/features, 03 baseline, 04 grafos,
               07 mace smoke, 08 mace features EDA, 10 mace finetune smoke
scripts/       01_download, 02_build_dataset, 03_train_baseline,
               04_build_graphs, 05_train_schnet, 06_compare,
               07_extract_mace_features, 08b_feature_reduction_sweep,
               09_extract_mace_embeddings, 10_finetune_mace,
               11_figures_dissertação
data/          raw, processed (.traj + her_pyg.pt), lmdb, mace_features/, figures, metadata.sqlite
results/       runs/{ts}_{name}/ + summary.json, dissertação_metrics.md, figures/
```
