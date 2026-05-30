import { http } from './client'
import type { ScreenRequest, ScreenResponse, StatsResponse } from './types'

export async function fetchStats(): Promise<StatsResponse> {
  const { data } = await http.get<StatsResponse>('/stats')
  return data
}

export async function fetchElements(): Promise<string[]> {
  const { data } = await http.get<{ elements: string[] }>('/elements')
  return data.elements
}

export async function runScreen(req: ScreenRequest): Promise<ScreenResponse> {
  const { data } = await http.post<ScreenResponse>('/screen', req)
  return data
}
