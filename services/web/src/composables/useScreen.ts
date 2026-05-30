import { useMutation } from '@tanstack/vue-query'
import { runScreen } from '@/api'
import type { ScreenRequest, ScreenResponse, ApiError } from '@/api'
import { useScreenStore } from '@/stores/screen'

export function useScreen() {
  const store = useScreenStore()

  return useMutation<ScreenResponse, ApiError, ScreenRequest>({
    mutationFn: runScreen,
    onSuccess: (data, variables) => {
      store.setLastQuery(variables)
      store.setLastResult(data)
    },
  })
}
