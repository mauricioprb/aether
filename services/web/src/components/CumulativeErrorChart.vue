<script setup lang="ts">
import { computed } from "vue";
import VChart from "vue-echarts";
import { useDark } from "@vueuse/core";
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
  props.models.map((m) => {
    const abs = m.y_pred.map((p, i) => Math.abs(p - (m.y_true[i] ?? 0))).sort((a, b) => a - b);
    const n = abs.length;
    const data = abs.map((x, i) => [x, (i + 1) / n]);
    return {
      name: m.display,
      type: "line" as const,
      data,
      showSymbol: false,
      smooth: false,
      lineStyle: { color: m.color, width: 2 },
      itemStyle: { color: m.color },
    };
  }),
);

const option = computed(() => {
  const textColor = isDark.value ? "#cbd5e1" : "#475569";
  const gridColor = isDark.value ? "#334155" : "#e2e8f0";
  return {
    grid: { left: 8, right: 16, top: 32, bottom: 40, containLabel: true },
    legend: {
      top: 0,
      textStyle: { color: textColor, fontSize: 11 },
      itemWidth: 18,
      itemHeight: 4,
    },
    tooltip: {
      trigger: "axis" as const,
      backgroundColor: isDark.value ? "#1e293b" : "#fff",
      borderColor: gridColor,
      textStyle: { color: textColor },
      valueFormatter: (v: number) => v.toFixed(4),
    },
    xAxis: {
      type: "value" as const,
      name: "limiar de |erro| (eV)",
      nameLocation: "middle" as const,
      nameGap: 24,
      axisLine: { lineStyle: { color: gridColor } },
      axisLabel: {
        color: textColor,
        fontSize: 10,
        formatter: (v: number) => v.toFixed(2),
        hideOverlap: true,
      },
      splitLine: { lineStyle: { color: gridColor, opacity: 0.4 } },
      nameTextStyle: { color: textColor, fontSize: 10 },
    },
    yAxis: {
      type: "value" as const,
      name: "fração de estruturas",
      nameLocation: "middle" as const,
      nameGap: 30,
      min: 0,
      max: 1.02,
      axisLine: { lineStyle: { color: gridColor } },
      axisLabel: {
        color: textColor,
        fontSize: 10,
        formatter: (v: number) => `${(v * 100).toFixed(0)}%`,
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
        markLine: {
          symbol: "none",
          silent: true,
          label: {
            color: textColor,
            fontSize: 10,
            formatter: "acurácia química (43 meV)",
          },
          lineStyle: { color: "#f59e0b", type: "dashed" as const, width: 1 },
          data: [{ xAxis: props.chemicalAccuracyEv }],
        },
      },
    ],
  };
});
</script>

<template>
  <div
    class="rounded-xl border border-surface-200 bg-surface-0 p-4 shadow-sm dark:border-surface-800 dark:bg-surface-950"
  >
    <header class="mb-3">
      <h3 class="text-sm font-semibold">Erro acumulado</h3>
      <p class="mt-0.5 text-xs text-surface-500">
        Fração de estruturas com |erro| abaixo do limiar do eixo x.
      </p>
    </header>
    <VChart :option="option" :style="{ height }" autoresize />
  </div>
</template>
