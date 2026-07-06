<script setup lang="ts">
import { computed } from "vue";
import VChart from "vue-echarts";
import { useDark } from "@vueuse/core";

const props = withDefaults(
  defineProps<{
    yTrue: number[];
    yPred: number[];
    bins?: number;
    title?: string;
    color?: string;
    height?: string;
    chemicalAccuracyEv?: number;
  }>(),
  { bins: 40, color: "#10b981", height: "280px", chemicalAccuracyEv: 0.043 },
);

const isDark = useDark();

const histogram = computed(() => {
  const residuals = props.yPred.map((v, i) => v - (props.yTrue[i] ?? 0));
  if (residuals.length === 0) return { centers: [] as number[], counts: [] as number[] };
  const min = Math.min(...residuals);
  const max = Math.max(...residuals);
  const range = max - min || 1;
  const width = range / props.bins;
  const counts = Array.from({ length: props.bins }, () => 0);
  for (const r of residuals) {
    const idx = Math.min(props.bins - 1, Math.floor((r - min) / width));
    counts[idx]! += 1;
  }
  const centers = Array.from({ length: props.bins }, (_, i) => min + width * (i + 0.5));
  return { centers, counts };
});

const option = computed(() => {
  const textColor = isDark.value ? "#cbd5e1" : "#475569";
  const gridColor = isDark.value ? "#334155" : "#e2e8f0";
  const data = histogram.value.centers.map((c, i) => [c, histogram.value.counts[i] ?? 0]);

  return {
    grid: { left: 8, right: 14, top: 16, bottom: 32, containLabel: true },
    tooltip: {
      trigger: "axis" as const,
      backgroundColor: isDark.value ? "#1e293b" : "#fff",
      borderColor: gridColor,
      textStyle: { color: textColor },
      formatter: (params: Array<{ value: [number, number] }>) => {
        const p = params[0];
        if (!p) return "";
        return `<div class="text-xs">centro: <b>${p.value[0].toFixed(3)}</b> eV<br/>contagem: <b>${p.value[1]}</b></div>`;
      },
    },
    xAxis: {
      type: "value" as const,
      name: "resíduo (eV)",
      nameLocation: "middle" as const,
      nameGap: 22,
      splitNumber: 5,
      axisLine: { lineStyle: { color: gridColor } },
      axisLabel: {
        color: textColor,
        fontSize: 10,
        formatter: (v: number) => v.toFixed(2),
        hideOverlap: true,
      },
      splitLine: { show: false },
      nameTextStyle: { color: textColor, fontSize: 10 },
    },
    yAxis: {
      type: "value" as const,
      minInterval: 1,
      axisLine: { lineStyle: { color: gridColor } },
      axisLabel: { color: textColor, fontSize: 10 },
      splitLine: { lineStyle: { color: gridColor, opacity: 0.4 } },
    },
    series: [
      {
        type: "bar" as const,
        data,
        barWidth: "95%",
        itemStyle: { color: props.color, opacity: 0.85, borderRadius: [2, 2, 0, 0] },
        markLine: {
          silent: true,
          symbol: "none",
          label: { show: false },
          data: [
            { xAxis: 0, lineStyle: { color: textColor, type: "dashed" as const, width: 1 } },
            {
              xAxis: props.chemicalAccuracyEv,
              lineStyle: { color: "#f59e0b", type: "dotted" as const, width: 1 },
            },
            {
              xAxis: -props.chemicalAccuracyEv,
              lineStyle: { color: "#f59e0b", type: "dotted" as const, width: 1 },
            },
          ],
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
    <header class="mb-3 flex items-baseline justify-between">
      <h3 class="text-sm font-semibold">{{ title ?? "Distribuição dos resíduos" }}</h3>
      <span class="text-2xs text-surface-500"
        >linhas: resíduo nulo e acurácia química (±43 meV)</span
      >
    </header>
    <VChart :option="option" :style="{ height }" autoresize />
  </div>
</template>
