<script setup lang="ts">
import { computed } from 'vue'
import type { ModelComparisonRow } from '@/api'

const props = defineProps<{
  models: ModelComparisonRow[]
  chemicalAccuracyEv: number
}>()

const maxMae = computed(() => {
  const vals = props.models.map((m) => m.mae_meV_test + (m.mae_test_std ?? 0) * 1000)
  return Math.max(...vals, props.chemicalAccuracyEv * 1000 * 4)
})

const sorted = computed(() => [...props.models].sort((a, b) => b.mae_meV_test - a.mae_meV_test))

function widthPct(value: number): string {
  return `${(value / maxMae.value) * 100}%`
}

function errPct(std: number | null): string {
  if (!std) return '0%'
  return `${((std * 1000) / maxMae.value) * 100}%`
}

const chemAccPct = computed(() => `${((props.chemicalAccuracyEv * 1000) / maxMae.value) * 100}%`)
</script>

<template>
  <div class="rounded-xl border border-surface-200 bg-surface-0 p-5 shadow-sm dark:border-surface-800 dark:bg-surface-950">
    <header class="mb-4 flex items-center justify-between">
      <div>
        <h3 class="text-sm font-semibold">MAE em teste (meV)</h3>
        <p class="mt-0.5 text-xs text-surface-500">menor é melhor · barras de erro = ± std multi-seed</p>
      </div>
      <span class="text-[11px] text-surface-500">
        <span class="mr-1 inline-block h-2 w-2 rounded-full bg-amber-400"></span>
        chemical accuracy (43 meV)
      </span>
    </header>

    <div class="relative space-y-2.5">
      <div
        class="pointer-events-none absolute inset-y-0 w-px border-l-2 border-dashed border-amber-400/70"
        :style="{ left: chemAccPct }"
        :title="`${chemicalAccuracyEv * 1000} meV`"
      ></div>

      <div v-for="m in sorted" :key="m.display" class="flex items-center gap-3">
        <div class="w-44 shrink-0 text-xs font-medium text-surface-700 dark:text-surface-200">
          {{ m.display }}
        </div>
        <div class="relative h-6 flex-1 rounded-md bg-surface-100 dark:bg-surface-800">
          <div
            class="h-full rounded-md transition-all"
            :style="{ width: widthPct(m.mae_meV_test), backgroundColor: m.color, opacity: 0.85 }"
          ></div>
          <div
            v-if="m.mae_test_std != null"
            class="absolute top-1/2 h-1.5 -translate-y-1/2 border-x-2 border-surface-600/70 dark:border-surface-200/70"
            :style="{ left: widthPct(m.mae_meV_test - m.mae_test_std * 1000), width: errPct((m.mae_test_std ?? 0) * 2) }"
            :title="`± ${(m.mae_test_std * 1000).toFixed(1)} meV`"
          ></div>
        </div>
        <div class="w-28 shrink-0 text-right font-mono text-xs tabular-nums">
          {{ m.mae_meV_test.toFixed(0) }}
          <span v-if="m.mae_test_std != null" class="text-surface-500">± {{ (m.mae_test_std * 1000).toFixed(0) }}</span>
          meV
        </div>
      </div>
    </div>
  </div>
</template>
