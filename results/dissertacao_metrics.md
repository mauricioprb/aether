# Métricas

Dataset: 5.860 estruturas HER do Catalysis Hub
Split: 4.220 treino / 468 validação / 1.172 teste (canônico por id, seed=42)
Target: ΔG_H\* (eV) ∈ [-2, 2]
GPU: NVIDIA GeForce RTX 5060 Ti 16 GB

Multi-seed (n=5, seeds 42, 1, 2, 3, 4) reportado como **mean ± std** para modelos
não-determinísticos. ETR (sklearn) é determinístico per seed.

---

## Legenda: tipos de modelo

| Termo                    | Significado                                                                                                                                                                                                     |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ETR**                  | Extremely Randomized Trees (scikit-learn). Modelo de árvore, não é rede neural.                                                                                                                                 |
| **MACE**                 | Message-passing Atomic Cluster Expansion. GNN equivariante 3D para sistemas atômicos.                                                                                                                           |
| **MACE-MP-0**            | MACE pré-treinado no Materials Project / Open Catalyst Project (milhões de estruturas).                                                                                                                         |
| **SchNet**               | GNN para materiais treinada do zero (sem pré-treino) nos 5.860 exemplos do Catalysis Hub.                                                                                                                       |
| **Backbone**             | Corpo principal da rede neural (MACE: message passing + interações equivariantes, 4.7M parâmetros).                                                                                                             |
| **Cabeça (MLP head)**    | Camadas finais que mapeiam os embeddings de nó → predição de ΔG_H\*.                                                                                                                                            |
| **Congelado (frozen)**   | Os pesos do backbone **não são atualizados** durante o treino. Só a cabeça aprende. O backbone só faz inferência (forward pass), usando conhecimento pré-treinado. Ideal para poucos dados - evita overfitting. |
| **Completo (full)**      | O backbone **inteiro é treinado** junto com a cabeça (fine-tune end-to-end). Requer mais VRAM e mais dados; risco de overfitting com datasets pequenos.                                                         |
| **Híbrido**              | Combinação de GNN como extrator de features + modelo clássico (ETR) como preditor. A GNN **não é treinada** na tarefa-alvo - só extrai representações.                                                          |
| **GNN pura**             | A rede neural **é o modelo preditivo**: recebe a estrutura 3D e produz ΔG_H\* diretamente. É a abordagem proposta pela dissertação.                                                                             |
| **Features handcrafted** | Descritores eletrônicos/estruturais calculados manualmente (10 features), sem uso de aprendizado profundo.                                                                                                       |
| **Embeddings**           | Vetores latentes extraídos das camadas internas do MACE-MP-0 (640 dimensões por átomo). Capturam o ambiente químico local.                                                                                      |
| **SHAP**                 | SHapley Additive exPlanations - método para medir importância de cada feature na predição.                                                                                                                      |
| **PCA**                  | Principal Component Analysis - redução de dimensionalidade linear.                                                                                                                                              |

---

## Pacote unificado de métricas

Todo modelo reporta o mesmo conjunto:

| Métrica         | Significado                                                       |
| --------------- | ----------------------------------------------------------------- |
| **R²**          | Coeficiente de determinação (1.0 = predição perfeita).            |
| **MAE**         | Mean Absolute Error (eV).                                         |
| **RMSE**        | Root Mean Squared Error (eV) - sensível a outliers.              |
| **MDAE**        | Median Absolute Error (eV) - robusto a outliers.                 |
| **max err**     | Pior erro absoluto (eV).                                          |
| **Pearson r**   | Correlação linear pred vs DFT.                                    |
| **Spearman ρ**  | Correlação de ranking (preserva ordem).                           |
| **sMAPE**       | Symmetric MAPE - estável quando alvo cruza zero.                  |
| **MAE (meV)**   | MAE em meV (comparação com literatura DFT).                       |
| **% < 43 meV**  | Fração de predições com \|erro\| menor que chemical accuracy.    |

---

## Resultados principais (mean ± std, n=5)

| Modelo                                            |  R² test            |  MAE (eV)          | RMSE (eV)         | # Params |    Tipo    |
| ------------------------------------------------- | :-----------------: | :----------------: | :---------------: | :------: | :--------: |
| **ETR + MACE embeddings (512-dim, frozen)**       | **0.9613**          | 0.0714             | 0.1232            |    -     |  Híbrido   |
| **MACE fine-tune Stage A (frozen + MLP 2-layer)** | **0.9564 ± 0.0015** | 0.0706 ± 0.0014    | 0.1309 ± 0.0022   |   328K   |    GNN     |
| ETR + 10 features handcrafted                     | 0.9341              | 0.0955             | 0.1609            |    -     |  Baseline  |
| SchNet (GNN treinada do zero)                     | 0.9105 ± 0.0511     | 0.0734 ± 0.0145    | 0.1804 ± 0.0573   |   456K   |    GNN     |

**Stage A 34× mais estável que SchNet** (std 0.0015 vs 0.0511).

---

## Per-seed (sanity check)

### SchNet v2 (val_r2 mode=max patience=60)

| seed | R²     | MAE (eV) | epoch best |
| ---- | ------ | -------- | ---------- |
| 42   | 0.9540 | 0.0598   | 130        |
| 1    | 0.9703 | 0.0643   | 79         |
| 2    | 0.8991 | 0.0916   | 37         |
| 3    | 0.8822 | 0.0865   | 68         |
| 4    | 0.8469 | 0.0650   | 108        |

### MACE Stage A v2

| seed | R²     | MAE (eV) | epoch best |
| ---- | ------ | -------- | ---------- |
| 42   | 0.9539 | 0.0719   | 166        |
| 1    | 0.9575 | 0.0705   | 192        |
| 2    | 0.9574 | 0.0688   | 198        |
| 3    | 0.9562 | 0.0720   | 176        |
| 4    | 0.9571 | 0.0699   | 185        |

---

## Ablação - Redução de dimensionalidade nos embeddings MACE (ETR, determinístico)

| Estratégia          | # Features | R² test |  MAE (eV) | MAE (meV) | RMSE (eV) |
| ------------------- | :--------: | :-----: | :-------: | :-------: | :-------: |
| Todos embeddings    |    512     | 0.9613  | 0.0714    | 71        | 0.1232    |
| Top-100 SHAP        |    100     | 0.9611  | 0.0725    | 73        | 0.1236    |
| PCA 99% var         |     77     | 0.9600  | 0.0767    | 77        | 0.1254    |
| Top-50 SHAP         |     50     | 0.9577  | 0.0765    | 77        | 0.1289    |
| PCA 95% var         |     38     | 0.9568  | 0.0791    | 79        | 0.1302    |
| PCA 90% var         |     26     | 0.9562  | 0.0798    | 80        | 0.1312    |
| Top-20 SHAP         |     20     | 0.9517  | 0.0846    | 85        | 0.1378    |
| Top-10 SHAP         |     10     | 0.9397  | 0.0956    | 96        | 0.1539    |

Sweet spot: **top-20** (R²=0.952, sem perda significativa); **top-50** (0.958)
se quiser quase o teto.

---

## Visualização hierárquica dos resultados

```
                ETR + MACE embeddings (frozen, determinístico)
                              0.9613  ← teto da representação MACE
                              │
                              ▼
                  MACE Stage A (GNN pura)
                              0.9564 ± 0.0015  ← multi-seed estável
                              │
                              ▼
                  ETR + 10 handcrafted (baseline)
                              0.9341
                              │
                              ▼
                  SchNet (GNN do zero)
                              0.9105 ± 0.0511  ← alta variância
```

---

## Hipótese

Representação aprendida pelo MACE-MP-0 (pré-treinado em milhões de estruturas
Materials Project/OCP) supera descritores handcrafted **tanto em performance
quanto em reprodutibilidade**. Transfer learning fornece prior estável que
elimina a variância inerente ao treino do zero.

## Evidência

1. **MACE Stage A** (GNN pura, backbone frozen + MLP head): R²=0.9564 ± 0.0015
   (n=5). Supera baseline ETR+handcrafted (0.9341) em 2.2 pontos R².

2. **ETR + MACE embeddings** (híbrido): R²=0.9613. Estabelece teto de informação
   da representação MACE pré-treinada (sem fine-tune).

3. **SchNet do zero**: R²=0.9105 ± 0.0511 (n=5). Pior média e **34× mais
   variância** que Stage A. Sem prior, converge erraticamente.

4. **Mesmo top-10 dimensões dos embeddings MACE** (0.940) supera as 10 features
   handcrafted (0.934) - representação aprendida concentra informação melhor.

5. **Auditoria de splits**: todos os 5 modelos avaliados no **mesmo test set
   canônico** (1172 IDs, y_true bit-identical cross-run). Zero leakage
   train/test, validado por `sid` em `predictions.parquet`.

## Limitações

- Dataset de 5.860 estruturas após filtros aplicados sobre o snapshot
  disponível do Catalysis Hub.
- ETR sobre embeddings (0.961) supera Stage A com MLP (0.956), sugerindo que
  modelos baseados em árvore extraem mais informação de representações de
  alta dimensionalidade com poucos exemplos (4220 train) do que MLP head leve.
- Full fine-tune MACE (Stage C, run histórico R²=0.940) sofre overfit com
  5M parâmetros treináveis e 4220 amostras de treino.
- SchNet alta variância não foi resolvida por ajustes de early stopping
  (patience=60 + monitor=val_r2 mode=max). Causa: não-determinismo CUDA +
  ausência de prior. Determinismo total exigiria
  `torch.use_deterministic_algorithms(True)` + flags adicionais.
