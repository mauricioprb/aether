import axios, { AxiosError, type AxiosInstance } from 'axios'
import type { ApiError } from './types'

const baseURL = import.meta.env.VITE_API_BASE_URL ?? '/api'

export const http: AxiosInstance = axios.create({
  baseURL,
  timeout: 60_000,
  headers: { 'Content-Type': 'application/json' },
})

http.interceptors.response.use(
  (r) => r,
  (error: AxiosError<{ detail?: string }>) => {
    const apiError: ApiError = {
      status: error.response?.status ?? 0,
      message:
        error.response?.data?.detail ??
        error.message ??
        'Falha na comunicação com a API',
      detail: error.response?.data,
    }
    return Promise.reject(apiError)
  },
)
