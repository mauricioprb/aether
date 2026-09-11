<script setup lang="ts">
import { computed } from "vue";
import VChart from "vue-echarts";
import { useDark } from "@vueuse/core";
import { chartTheme, dgRamp } from "@/charts/palette";
import { num } from "@/format";
import type { CandidateRow } from "@/api";

const props = withDefaults(
  defineProps<{ rows: CandidateRow[]; pool: number[]; chemicalAccuracyEv?: number }>(),
  { chemicalAccuracyEv: 0.043 },
);

const isDark = useDark();

// Nørskov 2005: i0 ∝ 1 / (1 + exp(|ΔG_H| / kT)). O pico fica em ΔG_H = 0, que é
// o próprio critério de Sabatier pelo qual a triagem ranqueia.
const KT_300K = 0.02585;

/** Normalizado pelo ápice: 0 é o melhor caso possível (ΔG_H = 0) e cada unidade
 *  abaixo é uma ordem de grandeza a menos de corrente de troca. Sem normalizar,
 *  o topo ficava em −0,3 e os números não significavam nada sozinhos. */
const activity = (dg: number) => Math.log10(2 / (1 + Math.exp(Math.abs(dg) / KT_300K)));

const LIMIT = 0.5;

const curve = computed(() =>
  Array.from({ length: 201 }, (_, i) => {
    const dg = -LIMIT + (2 * LIMIT * i) / 200;
    return [dg, activity(dg)];
  }),
);

const poolPoints = computed(() =>
  props.pool.filter((v) => Math.abs(v) <= LIMIT).map((v) => [v, activity(v)]),
);

const topPoints = computed(() =>
  props.rows.map((r) => ({
    value: [r.dG_pred, activity(r.dG_pred)],
    name: r.chemical_formula,
    itemStyle: { color: dgRamp(r.dG_pred, isDark.value) },
  })),
);

const offScale = computed(() => props.pool.filter((v) => Math.abs(v) > LIMIT).length);

const option = computed(() => {
  const { text, grid, surface, faint } = chartTheme(isDark.value);

  return {
    grid: { left: 0, right: 4, top: 8, bottom: 34, containLabel: true },
    tooltip: {
      trigger: "item" as const,
      backgroundColor: surface,
      borderColor: grid,
      textStyle: { color: text, fontSize: 12 },
      formatter: (p: { name?: string; value: [number, number]; seriesName: string }) =>
        p.seriesName === "top"
          ? `<b>${p.name}</b><br/>ΔG_H ${num(p.value[0], 3)} eV<br/>${num(p.value[1], 1)} ordens abaixo do ótimo`
          : `ΔG_H ${num(p.value[0], 3)} eV<br/>${num(p.value[1], 1)} ordens abaixo do ótimo`,
    },
    xAxis: {
      type: "value" as const,
      name: "ΔG_H previsto (eV)",
      nameLocation: "middle" as const,
      nameGap: 26,
      min: -LIMIT,
      max: LIMIT,
      axisLine: { lineStyle: { color: grid } },
      axisLabel: {
        color: text,
        fontSize: 10,
        hideOverlap: true,
        formatter: (v: number) => num(v, 1),
      },
      splitLine: { show: false },
      nameTextStyle: { color: text, fontSize: 10 },
    },
    yAxis: {
      type: "value" as const,
      axisLine: { lineStyle: { color: grid } },
      axisLabel: { color: text, fontSize: 10, formatter: (v: number) => num(v, 0) },
      splitLine: { lineStyle: { color: grid, opacity: 0.35 } },
    },
    series: [
      {
        name: "curva",
        type: "line" as const,
        data: curve.value,
        showSymbol: false,
        silent: true,
        lineStyle: { color: faint, width: 1.5 },
        areaStyle: { color: faint, opacity: 0.1 },
        markLine: {
          silent: true,
          symbol: "none",
          label: { show: false },
          lineStyle: { color: text, type: "dotted" as const, width: 1 },
          data: [{ xAxis: 0 }],
        },
      },
      {
        name: "pool",
        type: "scatter" as const,
        data: poolPoints.value,
        symbolSize: 6,
        itemStyle: { color: faint, opacity: 0.7 },
      },
      {
        name: "top",
        type: "scatter" as const,
        data: topPoints.value,
        symbolSize: 12,
        itemStyle: {
          opacity: 1,
          borderColor: isDark.value ? "#101415" : "#ffffff",
          borderWidth: 1.5,
        },
        z: 5,
      },
    ],
  };
});
</script>

<template>
  <section
    class="rounded-lg border border-surface-200 bg-surface-0 p-4 dark:border-surface-800 dark:bg-surface-950"
  >
    <header class="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
      <h3 class="text-sm font-semibold text-surface-900 dark:text-surface-0">Vulcão da HER</h3>
      <p class="text-xs text-surface-500">
        Atividade prevista pelo modelo de Nørskov, i₀ ∝ 1/(1 + e<sup>|ΔG_H|/kT</sup>) a 300 K. O
        topo é ΔG_H = 0, e descer uma unidade no eixo vertical é perder dez vezes em corrente de
        troca. Cinza: as {{ pool.length }} candidatas do filtro. Colorido: as {{ rows.length }} da
        tabela.
        <span v-if="offScale">{{ offScale }} ficam fora de ±0,5 eV.</span>
      </p>
    </header>
    <!-- Rótulo do eixo em HTML, não no ECharts: lá, um nome rotacionado é
         cortado (containLabel só reserva espaço para os rótulos, nunca para o
         nome) e nameLocation "end" ancora no meio do gráfico. -->
    <p class="mb-1 text-2xs text-surface-500 sm:hidden">ordens de grandeza abaixo do ótimo</p>
    <div class="flex gap-1.5">
      <p
        class="hidden self-center text-2xs text-surface-500 [writing-mode:vertical-rl] rotate-180 sm:block"
      >
        ordens de grandeza abaixo do ótimo
      </p>
      <div class="h-[17rem] min-w-0 flex-1 sm:h-[26rem]">
        <VChart :option="option" class="h-full w-full" autoresize />
      </div>
    </div>
  </section>
</template>
