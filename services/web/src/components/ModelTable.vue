<script setup lang="ts">
import { computed } from "vue";
import { useDark } from "@vueuse/core";
import { modelColor } from "@/charts/palette";
import { num } from "@/format";
import type { ModelComparisonRow } from "@/api";

const props = defineProps<{ models: ModelComparisonRow[] }>();

type Col = {
  key: string;
  label: string;
  hint: string;
  best: "max" | "min";
  get: (m: ModelComparisonRow) => number | null;
  fmt: (m: ModelComparisonRow) => string;
};

/** Modelos em linha e métricas em coluna: comparar vira leitura vertical, em
 *  vez de decorar quatro cartões. O melhor de cada coluna é marcado ali mesmo. */
const columns: Col[] = [
  {
    key: "r2",
    label: "R²",
    hint: "variância explicada, maior é melhor",
    best: "max",
    get: (m) => m.r2_test,
    fmt: (m) =>
      m.r2_test_std != null ? `${num(m.r2_test, 4)} ± ${num(m.r2_test_std, 4)}` : num(m.r2_test, 4),
  },
  {
    key: "mae",
    label: "MAE",
    hint: "erro absoluto médio, menor é melhor",
    best: "min",
    get: (m) => m.mae_meV_test,
    fmt: (m) =>
      m.mae_test_std != null
        ? `${num(m.mae_meV_test, 0)} ± ${num(m.mae_test_std * 1000, 0)} meV`
        : `${num(m.mae_meV_test, 0)} meV`,
  },
  {
    key: "rmse",
    label: "RMSE",
    hint: "penaliza erro grande, menor é melhor",
    best: "min",
    get: (m) => m.rmse_test,
    fmt: (m) => `${num(m.rmse_test, 4)} eV`,
  },
  {
    key: "chem",
    label: "Dentro de 43 meV",
    hint: "fração com erro abaixo da acurácia química",
    best: "max",
    get: (m) => m.frac_chem_acc_test,
    fmt: (m) => (m.frac_chem_acc_test != null ? `${num(m.frac_chem_acc_test * 100, 1)}%` : "n/d"),
  },
  {
    key: "rho",
    label: "ρ de Spearman",
    hint: "acerto na ordem do ranqueamento",
    best: "max",
    get: (m) => m.spearman_rho_test,
    fmt: (m) => (m.spearman_rho_test != null ? num(m.spearman_rho_test, 4) : "n/d"),
  },
  {
    key: "params",
    label: "Parâmetros",
    hint: "tamanho do modelo",
    best: "min",
    get: () => null, // tamanho não é qualidade: nunca marca melhor
    fmt: (m) => (m.n_params != null ? m.n_params.toLocaleString("pt-BR") : "n/d"),
  },
];

const bestByCol = computed(() => {
  const out: Record<string, number | null> = {};
  for (const c of columns) {
    const vals = props.models.map(c.get).filter((v): v is number => v != null);
    out[c.key] = vals.length ? (c.best === "max" ? Math.max(...vals) : Math.min(...vals)) : null;
  }
  return out;
});

function isBest(c: Col, m: ModelComparisonRow) {
  const v = c.get(m);
  const b = bestByCol.value[c.key];
  return v != null && b != null && Math.abs(v - b) < 1e-9;
}

const isDark = useDark();

/** Índice na ordem original da API: a cor identifica o modelo e não pode mudar
 *  quando a tabela é reordenada. */
const colorOf = (m: ModelComparisonRow) =>
  modelColor(
    props.models.findIndex((x) => x.display === m.display),
    isDark.value,
  );

const sorted = computed(() => [...props.models].sort((a, b) => b.r2_test - a.r2_test));
</script>

<template>
  <div
    class="overflow-x-auto rounded-lg border border-surface-200 bg-surface-0 dark:border-surface-800 dark:bg-surface-950"
  >
    <table class="w-full min-w-[54rem] border-collapse text-sm">
      <thead>
        <tr class="border-b border-surface-200 dark:border-surface-800">
          <th class="px-4 py-2.5 text-left text-xs font-semibold text-surface-500">Modelo</th>
          <th
            v-for="c in columns"
            :key="c.key"
            class="px-4 py-2.5 text-right text-xs font-semibold whitespace-nowrap text-surface-500"
            :title="c.hint"
          >
            {{ c.label }}
          </th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="m in sorted"
          :key="m.display"
          class="border-b border-surface-100 last:border-0 dark:border-surface-800/60"
        >
          <td class="px-4 py-3">
            <div class="flex items-center gap-2.5">
              <span
                class="h-2.5 w-2.5 shrink-0 rounded-full"
                :style="{ backgroundColor: colorOf(m) }"
              ></span>
              <div class="min-w-0">
                <div class="font-medium text-surface-900 dark:text-surface-0">
                  {{ m.display }}
                </div>
                <div class="text-xs text-surface-500">
                  {{ m.is_multiseed ? `média de ${m.n_seeds} seeds` : "run único" }}
                  <span v-if="m.kind === 'baseline'">, baseline</span>
                </div>
              </div>
            </div>
          </td>
          <td
            v-for="c in columns"
            :key="c.key"
            class="px-4 py-3 text-right font-mono tabular-nums whitespace-nowrap"
            :class="
              isBest(c, m)
                ? 'font-semibold text-primary-600 dark:text-primary-400'
                : 'text-surface-700 dark:text-surface-300'
            "
          >
            {{ c.fmt(m) }}
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
