<script setup lang="ts">
import { computed } from 'vue'
import Tag from 'primevue/tag'
import type { ModelComparisonRow } from '@/api'

const props = defineProps<{
  model: ModelComparisonRow
  bestR2: number
  bestMaeMeV: number
  bestFracChem: number
  bestRmse: number
}>()

const isBestR2 = computed(() => Math.abs(props.model.r2_test - props.bestR2) < 1e-9)
const isBestMae = computed(() => Math.abs(props.model.mae_meV_test - props.bestMaeMeV) < 1e-6)
const isBestFrac = computed(
  () => props.model.frac_chem_acc_test != null
    && Math.abs(props.model.frac_chem_acc_test - props.bestFracChem) < 1e-9,
)
const isBestRmse = computed(() => Math.abs(props.model.rmse_test - props.bestRmse) < 1e-9)

const kindStyle = computed(() => {
  switch (props.model.kind) {
    case 'baseline':
      return { label: 'Baseline', severity: 'secondary' as const }
    case 'gnn':
      return { label: 'GNN', severity: 'info' as const }
    case 'hybrid':
      return { label: 'Híbrido', severity: 'success' as const }
  }
})

function fmt(n: number | null, digits = 3) {
  return n == null ? '—' : n.toFixed(digits)
}

function fmtInt(n: number | null) {
  return n == null ? '—' : n.toLocaleString('pt-BR')
}

const accentStyle = computed(() => ({
  borderTopColor: props.model.color,
}))
</script>

<template>
  <article
    class="relative flex flex-col rounded-xl border border-surface-200 bg-surface-0 p-5 shadow-sm transition hover:shadow-md dark:border-surface-800 dark:bg-surface-950 border-t-4"
    :style="accentStyle"
  >
    <header class="mb-4 flex items-start justify-between gap-2">
      <div>
        <h3 class="text-sm font-semibold leading-tight">{{ model.display }}</h3>
        <p class="mt-0.5 text-xs text-surface-500">
          {{ model.is_multiseed ? `multi-seed n=${model.n_seeds}` : 'determinístico' }}
        </p>
      </div>
      <Tag :severity="kindStyle.severity" :value="kindStyle.label" rounded />
    </header>

    <dl class="grid grid-cols-2 gap-3">
      <div>
        <dt class="text-[10px] font-medium uppercase tracking-wide text-surface-500">R² test</dt>
        <dd class="flex items-baseline gap-1 font-mono tabular-nums">
          <span class="text-lg font-semibold" :class="isBestR2 ? 'text-primary-600 dark:text-primary-400' : ''">
            {{ fmt(model.r2_test, 4) }}
          </span>
          <span v-if="model.r2_test_std != null" class="text-[11px] text-surface-500">
            ± {{ model.r2_test_std.toFixed(4) }}
          </span>
          <i v-if="isBestR2" class="pi pi-star-fill text-[10px] text-amber-500" title="melhor" />
        </dd>
      </div>

      <div>
        <dt class="text-[10px] font-medium uppercase tracking-wide text-surface-500">MAE (meV)</dt>
        <dd class="flex items-baseline gap-1 font-mono tabular-nums">
          <span class="text-lg font-semibold" :class="isBestMae ? 'text-primary-600 dark:text-primary-400' : ''">
            {{ model.mae_meV_test.toFixed(0) }}
          </span>
          <span v-if="model.mae_test_std != null" class="text-[11px] text-surface-500">
            ± {{ (model.mae_test_std * 1000).toFixed(0) }}
          </span>
          <i v-if="isBestMae" class="pi pi-star-fill text-[10px] text-amber-500" />
        </dd>
      </div>

      <div>
        <dt class="text-[10px] font-medium uppercase tracking-wide text-surface-500">RMSE (eV)</dt>
        <dd class="font-mono tabular-nums">
          <span :class="isBestRmse ? 'font-semibold text-primary-600 dark:text-primary-400' : ''">
            {{ fmt(model.rmse_test, 4) }}
          </span>
          <i v-if="isBestRmse" class="ml-1 pi pi-star-fill text-[10px] text-amber-500" />
        </dd>
      </div>

      <div>
        <dt class="text-[10px] font-medium uppercase tracking-wide text-surface-500">% &lt; 43 meV</dt>
        <dd class="font-mono tabular-nums">
          <span :class="isBestFrac ? 'font-semibold text-primary-600 dark:text-primary-400' : ''">
            {{ model.frac_chem_acc_test != null ? (model.frac_chem_acc_test * 100).toFixed(1) + '%' : '—' }}
          </span>
          <i v-if="isBestFrac" class="ml-1 pi pi-star-fill text-[10px] text-amber-500" />
        </dd>
      </div>

      <div>
        <dt class="text-[10px] font-medium uppercase tracking-wide text-surface-500">Spearman ρ</dt>
        <dd class="font-mono tabular-nums">{{ fmt(model.spearman_rho_test, 4) }}</dd>
      </div>

      <div>
        <dt class="text-[10px] font-medium uppercase tracking-wide text-surface-500">Parâmetros</dt>
        <dd class="font-mono tabular-nums">{{ fmtInt(model.n_params) }}</dd>
      </div>
    </dl>
  </article>
</template>
