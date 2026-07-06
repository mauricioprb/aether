export type ModelName = "etr_emb" | "stagea" | "ensemble";

export interface ScreenRequest {
  elements: string[];
  top: number;
  model: ModelName;
  exclude_train: boolean;
}

export interface CandidateRow {
  id: string;
  chemical_formula: string;
  composition: string;
  facet: string;
  site_type: string;
  coverage: number | null;
  delta_G_H: number;
  dG_pred: number;
  dG_dft: number;
  abs_dG_pred: number;
  error_vs_dft: number;
  dG_pred_etr: number | null;
  dG_pred_stagea: number | null;
}

export interface ScreenResponse {
  elements: string[];
  model: ModelName;
  top: number;
  exclude_train: boolean;
  n_candidates: number;
  rows: CandidateRow[];
}

export interface StatsResponse {
  n_structures: number;
  n_test_canonical: number;
  available_elements: string[];
  available_models: string[];
}

export type ModelKind = "baseline" | "gnn" | "hybrid";

export interface ModelComparisonRow {
  display: string;
  color: string;
  kind: ModelKind;
  is_multiseed: boolean;
  n_seeds: number | null;
  r2_test: number;
  r2_test_std: number | null;
  mae_test: number;
  mae_test_std: number | null;
  mae_meV_test: number;
  rmse_test: number;
  mdae_test: number | null;
  pearson_r_test: number | null;
  spearman_rho_test: number | null;
  frac_chem_acc_test: number | null;
  n_params: number | null;
  elapsed_sec: number | null;
}

export interface ComparisonResponse {
  models: ModelComparisonRow[];
  chemical_accuracy_eV: number;
}

export interface ModelPredictions {
  display: string;
  color: string;
  y_true: number[];
  y_pred: number[];
}

export interface ComparisonPredictionsResponse {
  models: ModelPredictions[];
  chemical_accuracy_eV: number;
}

export interface ApiError {
  status: number;
  message: string;
  detail?: unknown;
}
