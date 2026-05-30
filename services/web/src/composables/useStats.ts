import { useQuery } from '@tanstack/vue-query'
import { fetchStats } from '@/api'

export function useStats() {
  return useQuery({
    queryKey: ['stats'],
    queryFn: fetchStats,
    staleTime: 5 * 60_000,
  })
}
