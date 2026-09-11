import { useMutation } from "@tanstack/vue-query";
import { runPredict } from "@/api";
import type { ApiError, ModelName, PredictResponse } from "@/api";

export function usePredict() {
  return useMutation<PredictResponse, ApiError, { file: File; model: ModelName }>({
    mutationFn: ({ file, model }) => runPredict(file, model),
  });
}
