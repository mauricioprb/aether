import { useQuery } from "@tanstack/vue-query";
import { fetchElementStats } from "@/api";

/** Agregado por elemento. Estático entre requisições, então cacheia largo. */
export function useElementStats() {
  return useQuery({
    queryKey: ["element-stats"],
    queryFn: fetchElementStats,
    staleTime: Infinity,
  });
}
