# her-gnn

Etapa 1 da dissertacao: coleta de dados do Catalysis Hub e
reproducao do baseline Extra Trees para predicao de catalisadores de HER
(Hydrogen Evolution Reaction). Replica a metodologia de Wang et al., npj Comput.
Mater. (2025) 11:111 ([DOI 10.1038/s41524-025-01607-4](https://doi.org/10.1038/s41524-025-01607-4)).

## Reproducao (5 comandos)

Gerenciado por [uv](https://docs.astral.sh/uv/) (Python 3.14):

```bash
uv sync                                          # 1. ambiente
uv run python scripts/01_download.py             # 2. dump bruto do Catalysis Hub
uv run python scripts/02_build_dataset.py        # 3. .traj + LMDB + SQLite (filtros + features)
uv run python scripts/03_train_baseline.py       # 4. treina ETR, metricas, figuras
uv run jupyter lab                               # 5. notebooks exploratorios
```

`01_download.py` usa duas passadas: metadata de ~60k reacoes `products="H"`,
depois estruturas so das ~9k reacoes HER limpas (`0.5H2(g) + * -> H*`) que
passam os filtros de metadata (~15min). Saidas: `data/raw/`,
`data/processed/her_dataset.traj`, `data/lmdb/her_dataset.lmdb`,
`data/metadata.sqlite`, `data/figures/`.

## Resultado (baseline)

| metrica | teste | paper |
|---------|-------|-------|
| R2      | 0.910 | 0.922 |
| MAE     | 0.100 | -     |
| RMSE    | 0.182 | -     |

5860 estruturas curadas (split 80/20). Melhor ETR: `n_estimators=500`,
`max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`.

## Pipeline

- `ingest.py` - query GraphQL (metadata + busca por id em lote) + reconstrucao
  de `ase.Atoms` (sistema `Hstar`)
- `filters.py` - equacao HER limpa (`0.5H2(g) + * -> H*`, exclui co-adsorcao
  H2S/NH3/...), `delta_G in [-2,2]`, cobertura <= 25%, ligacao H-superficie
  em [1,3] A; sitio (top/bridge/hollow) por coordenacao do H em 2.4 A
- `features.py` - as 10 features (medias geometricas via `mendeleev`)
- `geometry.py` - atomos centrais (2.4 A do H) e vizinhos (1a camada, raios
  covalentes)
- `storage.py` - `.traj`, LMDB (estilo fairchem) e SQLite, com IDs alinhados
- `baseline.py` - ExtraTrees, split 80/20, GridSearchCV 10-fold, R2/MAE/MSE/RMSE
- `plots.py` - Fig. 3b (histograma), 4f (parity), 6d (SHAP)

Notebooks (`01` explora, `02` filtros/features, `03` baseline) importam de
`src/her_gnn`, sem duplicar logica.

### Desvios documentados do paper

- Vizinhos e CN usam lista de vizinhanca por raios covalentes (o cutoff de
  2.4 A do paper define so os atomos centrais; e curto demais para a 1a camada
  metal-metal).
- `Out_e` = `nvalence` do mendeleev (consistente por grupo); `Nd`/`Np` =
  contagem total de eletrons d/p.
- Contagem final (5860) fica abaixo dos 10.855 do paper: a API do Catalysis Hub
  limita paginas a 200 itens e o cursor padrao repete/pula registros; o snapshot
  de 2025 do paper nao e reproduzivel hoje. A paginacao de metadata usa
  `order:"id"` e deduplica por id. O alvo principal (R2 >= 0.90) foi atingido.

## Estrutura

```
src/her_gnn/   ingest, filters, features, geometry, storage, baseline, plots, dataset
notebooks/     exploracao e orquestracao
scripts/       01_download, 02_build_dataset, 03_train_baseline
data/          raw, processed (.traj), lmdb, figures, metadata.sqlite
```
