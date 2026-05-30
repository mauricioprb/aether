import { useQuery } from '@tanstack/vue-query'
import { fetchComparison } from '@/api'

export function useComparison() {
  return useQuery({
    queryKey: ['comparison'],
    queryFn: fetchComparison,
    staleTime: Infinity,
  })
}
