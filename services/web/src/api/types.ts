export type ModelName = 'etr_emb' | 'stagea' | 'ensemble'

export interface ScreenRequest {
  elements: string[]
  top: number
  model: ModelName
  exclude_train: boolean
}

export interface CandidateRow {
  id: string
  chemical_formula: string
  composition: string
  facet: string
  site_type: string
  coverage: number | null
  delta_G_H: number
  dG_pred: number
  abs_dG_pred: number
  error_vs_dft: number
  dG_pred_etr: number | null
  dG_pred_stagea: number | null
}

export interface ScreenResponse {
  elements: string[]
  model: ModelName
  top: number
  exclude_train: boolean
  n_candidates: number
  rows: CandidateRow[]
}

export interface StatsResponse {
  n_structures: number
  n_test_canonical: number
  available_elements: string[]
  available_models: string[]
}

export interface ApiError {
  status: number
  message: string
  detail?: unknown
}
