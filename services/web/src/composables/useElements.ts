import { useQuery } from '@tanstack/vue-query'
import { fetchElements } from '@/api'

export function useElements() {
  return useQuery({
    queryKey: ['elements'],
    queryFn: fetchElements,
    staleTime: Infinity,
  })
}
