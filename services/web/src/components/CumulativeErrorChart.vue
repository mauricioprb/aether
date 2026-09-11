<script setup lang="ts">
import { num } from "@/format";
import { computed } from "vue";
import VChart from "vue-echarts";
import { useDark } from "@vueuse/core";
import { chartRef, chartTheme, modelColor } from "@/charts/palette";
import type { ModelPredictions } from "@/api";

const props = withDefaults(
  defineProps<{
    models: ModelPredictions[];
    chemicalAccuracyEv?: number;
    height?: string;
  }>(),
  { chemicalAccuracyEv: 0.043, height: "380px" },
);

const isDark = useDark();

const series = computed(() =>
  props.models.map((m, i) => {
    const abs = m.y_pred.map((p, i) => Math.abs(p - (m.y_true[i] ?? 0))).sort((a, b) => a - b);
    const n = abs.length;
    const data = abs.map((x, i) => [x, (i + 1) / n]);
    return {
      name: m.display,
      type: "line" as const,
      data,
      showSymbol: false,
      smooth: false,
      lineStyle: { color: modelColor(i, isDark.value), width: 2 },
      itemStyle: { color: modelColor(i, isDark.value) },
    };
  }),
);

const option = computed(() => {
  const { text: textColor, grid: gridColor, surface } = chartTheme(isDark.value);
  return {
    grid: { left: 0, right: 4, top: 30, bottom: 34, containLabel: true },
    legend: {
      data: props.models.map((m) => m.display),
      top: 0,
      textStyle: { color: textColor, fontSize: 11 },
      itemWidth: 18,
      itemHeight: 4,
    },
    tooltip: {
      trigger: "axis" as const,
      backgroundColor: surface,
      borderColor: gridColor,
      textStyle: { color: textColor },
      valueFormatter: (v: number) => num(v, 4),
    },
    xAxis: {
      type: "value" as const,
      name: "limiar de |erro| (eV)",
      nameLocation: "middle" as const,
      nameGap: 24,
      // Recortado em 0,6 eV: 99% das estruturas erram menos que isso, e as
      // curvas só se separam abaixo de 0,3. No eixo completo, um único outlier
      // do SchNet (5,08 eV) empurrava a comparação inteira para 5% da largura.
      min: 0,
      max: 0.6,
      axisLine: { lineStyle: { color: gridColor } },
      axisLabel: {
        color: textColor,
        fontSize: 10,
        formatter: (v: number) => num(v, 2),
        hideOverlap: true,
      },
      splitLine: { lineStyle: { color: gridColor, opacity: 0.4 } },
      nameTextStyle: { color: textColor, fontSize: 10 },
    },
    yAxis: {
      type: "value" as const,
      min: 0,
      max: 1.02,
      axisLine: { lineStyle: { color: gridColor } },
      axisLabel: {
        color: textColor,
        fontSize: 10,
        formatter: (v: number) => `${num(v * 100, 0)}%`,
      },
      splitLine: { lineStyle: { color: gridColor, opacity: 0.4 } },
      nameTextStyle: { color: textColor, fontSize: 10 },
    },
    series: [
      ...series.value,
      {
        name: "acurácia química",
        type: "line" as const,
        data: [],
        lineStyle: { color: chartRef(isDark.value) },
        itemStyle: { color: chartRef(isDark.value) },
        markLine: {
          symbol: "none",
          silent: true,
          label: {
            color: textColor,
            fontSize: 10,
            formatter: "acurácia química (43 meV)",
          },
          lineStyle: { color: chartRef(isDark.value), type: "dashed" as const, width: 1 },
          data: [{ xAxis: props.chemicalAccuracyEv }],
        },
      },
    ],
  };
});
</script>

<template>
  <div
    class="rounded-lg border border-surface-200 bg-surface-0 p-4 dark:border-surface-800 dark:bg-surface-950"
  >
    <header class="mb-3">
      <h3 class="text-sm font-semibold">Erro acumulado</h3>
      <p class="mt-0.5 text-xs text-surface-500">
        Fração de estruturas com |erro| abaixo do limiar do eixo x.
      </p>
    </header>
    <div class="flex gap-1.5">
      <p class="self-center text-2xs text-surface-500 [writing-mode:vertical-rl] rotate-180">
        fração de estruturas
      </p>
      <div class="min-w-0 flex-1" :style="{ height }">
        <VChart :option="option" class="h-full w-full" autoresize />
      </div>
    </div>
  </div>
</template>
