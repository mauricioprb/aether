<script setup lang="ts">
import { computed } from "vue";
import VChart from "vue-echarts";
import { useDark } from "@vueuse/core";

const props = withDefaults(
  defineProps<{
    yTrue: number[];
    yPred: number[];
    title?: string;
    color?: string;
    height?: string;
    chemicalAccuracyEv?: number;
  }>(),
  { color: "#10b981", height: "360px", chemicalAccuracyEv: 0.043 },
);

const isDark = useDark();

const stats = computed(() => {
  const n = props.yTrue.length;
  if (n === 0) return null;
  let sumErr2 = 0;
  let sumAbsErr = 0;
  let meanY = 0;
  for (let i = 0; i < n; i++) meanY += props.yTrue[i]!;
  meanY /= n;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const err = (props.yPred[i] ?? 0) - (props.yTrue[i] ?? 0);
    sumErr2 += err * err;
    sumAbsErr += Math.abs(err);
    const dy = (props.yTrue[i] ?? 0) - meanY;
    ssTot += dy * dy;
  }
  return {
    r2: ssTot === 0 ? 1 : 1 - sumErr2 / ssTot,
    mae: sumAbsErr / n,
    rmse: Math.sqrt(sumErr2 / n),
    n,
  };
});

const option = computed(() => {
  const all = [...props.yTrue, ...props.yPred];
  const min = Math.min(...all);
  const max = Math.max(...all);
  const pad = (max - min) * 0.05 || 0.1;
  const lo = Math.floor((min - pad) * 10) / 10;
  const hi = Math.ceil((max + pad) * 10) / 10;
  const data = props.yTrue.map((v, i) => [v, props.yPred[i] ?? 0]);
  const textColor = isDark.value ? "#cbd5e1" : "#475569";
  const gridColor = isDark.value ? "#334155" : "#e2e8f0";
  const fmt = (v: number) => v.toFixed(1);

  return {
    grid: { left: 8, right: 16, top: 16, bottom: 36, containLabel: true },
    tooltip: {
      trigger: "item" as const,
      formatter: (p: { value: [number, number] }) =>
        `<div class="text-xs">DFT: <b>${p.value[0].toFixed(3)}</b> eV<br/>pred: <b>${p.value[1].toFixed(3)}</b> eV<br/>erro: ${(p.value[1] - p.value[0]).toFixed(3)} eV</div>`,
      backgroundColor: isDark.value ? "#1e293b" : "#fff",
      borderColor: gridColor,
      textStyle: { color: textColor },
    },
    xAxis: {
      type: "value" as const,
      name: "ΔGₕ DFT (eV)",
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
      name: "ΔGₕ previsto (eV)",
      nameLocation: "middle" as const,
      nameGap: 30,
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
        itemStyle: { color: props.color, opacity: 0.55 },
        markLine: {
          silent: true,
          symbol: "none",
          label: { show: false },
          lineStyle: { color: textColor, type: "dashed" as const, width: 1 },
          data: [[{ coord: [lo, lo] }, { coord: [hi, hi] }]],
        },
        markArea: {
          silent: true,
          itemStyle: { color: props.color, opacity: 0.08 },
          data: [
            [
              { coord: [lo, lo - props.chemicalAccuracyEv] },
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
    class="rounded-xl border border-surface-200 bg-surface-0 p-4 shadow-sm dark:border-surface-800 dark:bg-surface-950"
  >
    <header class="mb-3 flex items-baseline justify-between">
      <div>
        <h3 class="text-sm font-semibold">{{ title ?? "Predição vs. DFT" }}</h3>
        <p v-if="stats" class="mt-0.5 text-xs text-surface-500">
          {{ stats.n }} estruturas, R² {{ stats.r2.toFixed(2).replace(".", ",") }}, MAE
          {{ (stats.mae * 1000).toFixed(0) }} meV
        </p>
      </div>
    </header>
    <VChart :option="option" :style="{ height }" autoresize />
  </div>
</template>
