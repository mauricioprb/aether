<script setup lang="ts">
import { num } from "@/format";
import { computed } from "vue";
import VChart from "vue-echarts";
import { useDark } from "@vueuse/core";
import { chartAccent, chartTheme } from "@/charts/palette";

const props = withDefaults(
  defineProps<{
    yTrue: number[];
    yPred: number[];
    title?: string;
    color?: string;
    /** Mesma escala nos quatro: sem isto cada um se ajusta aos próprios dados
     *  (um chega a 1,8 eV, outro a 4,0) e a comparação visual não existe. */
    domain?: [number, number];
    chemicalAccuracyEv?: number;
  }>(),
  { chemicalAccuracyEv: 0.043 },
);

const isDark = useDark();
const seriesColor = computed(() => props.color ?? chartAccent(isDark.value));

const stats = computed(() => {
  const n = props.yTrue.length;
  if (n === 0) return null;
  let sumErr2 = 0;
  let sumAbsErr = 0;
  for (let i = 0; i < n; i++) {
    const err = (props.yPred[i] ?? 0) - (props.yTrue[i] ?? 0);
    sumErr2 += err * err;
    sumAbsErr += Math.abs(err);
  }
  // R² fica de fora de propósito: estas linhas foram selecionadas por terem
  // |ΔG previsto| ≈ 0, o que zera a variância de y e torna R² negativo por
  // construção. O R² do modelo é medido no conjunto de teste inteiro, em
  // Comparação. MAE sobre o subconjunto continua válido.
  return {
    mae: sumAbsErr / n,
    rmse: Math.sqrt(sumErr2 / n),
    within: props.yTrue.filter(
      (v, i) => Math.abs((props.yPred[i] ?? 0) - v) < props.chemicalAccuracyEv,
    ).length,
    n,
  };
});

const option = computed(() => {
  const all = [...props.yTrue, ...props.yPred];
  const min = Math.min(...all);
  const max = Math.max(...all);
  const pad = (max - min) * 0.05 || 0.1;
  const lo = props.domain ? props.domain[0] : Math.floor((min - pad) * 10) / 10;
  const hi = props.domain ? props.domain[1] : Math.ceil((max + pad) * 10) / 10;
  const data = props.yTrue.map((v, i) => [v, props.yPred[i] ?? 0]);
  const { text: textColor, grid: gridColor, surface } = chartTheme(isDark.value);
  const fmt = (v: number) => num(v, 1);

  return {
    grid: { left: 0, right: 4, top: 8, bottom: 34, containLabel: true },
    tooltip: {
      trigger: "item" as const,
      formatter: (p: { value: [number, number] }) =>
        `<div class="text-xs">referência: <b>${num(p.value[0], 3)}</b> eV<br/>predito: <b>${num(p.value[1], 3)}</b> eV<br/>resíduo: ${num(p.value[1] - p.value[0], 3)} eV</div>`,
      backgroundColor: surface,
      borderColor: gridColor,
      textStyle: { color: textColor },
    },
    xAxis: {
      type: "value" as const,
      name: "ΔG_H referência (eV)",
      nameLocation: "middle" as const,
      nameGap: 24,
      min: lo,
      max: hi,
      splitNumber: 4,
      axisLine: { lineStyle: { color: gridColor } },
      axisLabel: { color: textColor, fontSize: 10, formatter: fmt, hideOverlap: true },
      splitLine: { lineStyle: { color: gridColor, opacity: 0.4 } },
      nameTextStyle: { color: textColor, fontSize: 10 },
    },
    yAxis: {
      type: "value" as const,
      min: lo,
      max: hi,
      splitNumber: 4,
      axisLine: { lineStyle: { color: gridColor } },
      axisLabel: { color: textColor, fontSize: 10, formatter: fmt, hideOverlap: true },
      splitLine: { lineStyle: { color: gridColor, opacity: 0.4 } },
      nameTextStyle: { color: textColor, fontSize: 10 },
    },
    series: [
      {
        type: "scatter" as const,
        data,
        symbolSize: 6,
        itemStyle: { color: seriesColor.value, opacity: 0.55 },
        markLine: {
          silent: true,
          symbol: "none",
          label: { show: false },
          lineStyle: { color: textColor, type: "dashed" as const, width: 1 },
          data: [
            [{ coord: [lo, lo] }, { coord: [hi, hi] }],
            [
              {
                coord: [lo, lo + props.chemicalAccuracyEv],
                lineStyle: { color: seriesColor.value, type: "dotted" as const, width: 1 },
              },
              { coord: [hi, hi + props.chemicalAccuracyEv] },
            ],
            [
              {
                coord: [lo, lo - props.chemicalAccuracyEv],
                lineStyle: { color: seriesColor.value, type: "dotted" as const, width: 1 },
              },
              { coord: [hi, hi - props.chemicalAccuracyEv] },
            ],
          ],
        },
      },
    ],
  };
});
</script>

<template>
  <div
    class="w-full rounded-lg border border-surface-200 bg-surface-0 p-4 dark:border-surface-800 dark:bg-surface-950"
  >
    <header class="mb-3 flex items-baseline justify-between">
      <div>
        <h3 class="text-sm font-semibold">{{ title ?? "Predito vs. referência" }}</h3>
        <p v-if="stats" class="mt-0.5 text-xs text-surface-500">
          {{ stats.n }} estruturas, MAE {{ num(stats.mae * 1000, 0) }} meV, {{ stats.within }} com
          erro abaixo de {{ num(chemicalAccuracyEv * 1000, 0) }} meV
        </p>
      </div>
    </header>
    <div class="flex gap-1.5">
      <p class="self-center text-2xs text-surface-500 [writing-mode:vertical-rl] rotate-180">
        ΔG_H previsto (eV)
      </p>
      <div class="aspect-square min-w-0 flex-1">
        <VChart :option="option" class="h-full w-full" autoresize />
      </div>
    </div>
  </div>
</template>
