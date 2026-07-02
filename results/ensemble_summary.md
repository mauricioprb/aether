# Ensembles (média uniforme, sem ajuste de pesos)

Membros combinados por média simples das predições no test set canônico
(1172 estruturas). Nenhum peso é ajustado; runs individuais em
`results/summary.json`.

| Ensemble | R² | MAE (eV) | RMSE (eV) | MDAE (eV) | max err (eV) | Spearman ρ | % < 43 meV |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| etr_emb_all (referência, 1 modelo) | 0.9613 | 0.0714 | 0.1232 | 0.0463 | 1.752 | 0.9819 | 47.1 |
| stagea_ens5 (n=5) | 0.9584 | 0.0670 | 0.1278 | 0.0451 | 1.769 | 0.9837 | 48.0 |
| schnet_ens5 (n=5) | 0.9734 | 0.0585 | 0.1022 | 0.0384 | 1.582 | 0.9884 | 54.2 |
| stagea_ens5+etr_emb (n=6) | 0.9614 | 0.0646 | 0.1231 | 0.0434 | 1.765 | 0.9847 | 49.5 |
| stagea_ens5+schnet_ens5+etr_emb (n=11) | 0.9710 | 0.0560 | 0.1067 | 0.0371 | 1.653 | 0.9883 | 55.5 |

> blend val-fitted indisponível (results/runs/20260529_165552_mace_ft_stageA_v2_seed42 has no 'val' predictions logged); re-treine o Stage A para logar predições de validação.
