import { computed, type Ref } from "vue";
import { useQuery } from "@tanstack/vue-query";
import { fetchCount } from "@/api";

/** Contagem de candidatas para o filtro atual, para o botão dizer o tamanho do
 *  resultado antes de submeter. Só o filtro roda no servidor, então é barato. */
export function useCount(elements: Ref<string[]>, excludeTrain: Ref<boolean>) {
  return useQuery({
    queryKey: computed(() => ["count", [...elements.value].sort(), excludeTrain.value]),
    queryFn: () => fetchCount(elements.value, excludeTrain.value),
    enabled: computed(() => elements.value.length > 0),
    staleTime: 5 * 60_000,
  });
}
