<script setup lang="ts">
import { computed } from 'vue'
import Skeleton from 'primevue/skeleton'
import Message from 'primevue/message'
import DataTable from 'primevue/datatable'
import Column from 'primevue/column'
import Tag from 'primevue/tag'

import ModelCard from '@/components/ModelCard.vue'
import MaeBarChart from '@/components/MaeBarChart.vue'
import { useComparison } from '@/composables'

const { data, isLoading, error } = useComparison()

const sortedByR2 = computed(() =>
  [...(data.value?.models ?? [])].sort((a, b) => b.r2_test - a.r2_test),
)

const bestR2 = computed(() => Math.max(...(data.value?.models ?? []).map((m) => m.r2_test), -Infinity))
const bestMaeMeV = computed(() => Math.min(...(data.value?.models ?? []).map((m) => m.mae_meV_test), Infinity))
const bestFracChem = computed(() =>
  Math.max(...(data.value?.models ?? []).map((m) => m.frac_chem_acc_test ?? -Infinity)),
)
const bestRmse = computed(() => Math.min(...(data.value?.models ?? []).map((m) => m.rmse_test), Infinity))

function fmtR2(m: { r2_test: number; r2_test_std: number | null }) {
  return m.r2_test_std != null
    ? `${m.r2_test.toFixed(4)} ± ${m.r2_test_std.toFixed(4)}`
    : m.r2_test.toFixed(4)
}

function fmtMae(m: { mae_test: number; mae_test_std: number | null }) {
  return m.mae_test_std != null
    ? `${m.mae_test.toFixed(4)} ± ${m.mae_test_std.toFixed(4)}`
    : m.mae_test.toFixed(4)
}

function kindLabel(k: string) {
  return k === 'baseline' ? 'Baseline' : k === 'gnn' ? 'GNN' : 'Híbrido'
}

function kindSeverity(k: string): 'secondary' | 'info' | 'success' {
  return k === 'baseline' ? 'secondary' : k === 'gnn' ? 'info' : 'success'
}
</script>

<template>
  <section class="mx-auto max-w-7xl px-6 py-8">
    <header class="mb-6">
      <h1 class="text-2xl font-semibold tracking-tight">Comparação de modelos</h1>
      <p class="mt-1 text-sm text-surface-500">
        4 modelos finais avaliados no mesmo test set canônico (1172 estruturas). Modelos não-determinísticos
        reportados como mean ± std sobre 5 seeds.
      </p>
    </header>

    <div v-if="isLoading" class="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      <Skeleton v-for="i in 4" :key="i" height="14rem" />
    </div>

    <Message v-else-if="error" severity="error" :closable="false">
      Falha ao carregar comparação: {{ error.message }}
    </Message>

    <template v-else-if="data">
      <div class="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <ModelCard
          v-for="m in sortedByR2"
          :key="m.display"
          :model="m"
          :best-r2="bestR2"
          :best-mae-me-v="bestMaeMeV"
          :best-frac-chem="bestFracChem"
          :best-rmse="bestRmse"
        />
      </div>

      <div class="mt-6">
        <MaeBarChart :models="data.models" :chemical-accuracy-ev="data.chemical_accuracy_eV" />
      </div>

      <div
        class="mt-6 rounded-xl border border-surface-200 bg-surface-0 shadow-sm dark:border-surface-800 dark:bg-surface-950"
      >
        <header class="border-b border-surface-200 px-5 py-3 text-sm font-semibold dark:border-surface-800">
          Tabela completa
        </header>
        <DataTable :value="sortedByR2" striped-rows class="text-sm!">
          <Column field="display" header="Modelo">
            <template #body="{ data: row }">
              <div class="flex items-center gap-2">
                <span class="inline-block h-2.5 w-2.5 rounded-full" :style="{ backgroundColor: row.color }"></span>
                {{ row.display }}
              </div>
            </template>
          </Column>
          <Column field="kind" header="Tipo">
            <template #body="{ data: row }">
              <Tag :severity="kindSeverity(row.kind)" :value="kindLabel(row.kind)" rounded />
            </template>
          </Column>
          <Column field="r2_test" header="R² test" sortable>
            <template #body="{ data: row }">
              <span class="font-mono tabular-nums">{{ fmtR2(row) }}</span>
            </template>
          </Column>
          <Column field="mae_test" header="MAE (eV)" sortable>
            <template #body="{ data: row }">
              <span class="font-mono tabular-nums">{{ fmtMae(row) }}</span>
            </template>
          </Column>
          <Column field="rmse_test" header="RMSE (eV)" sortable>
            <template #body="{ data: row }">
              <span class="font-mono tabular-nums">{{ row.rmse_test.toFixed(4) }}</span>
            </template>
          </Column>
          <Column field="spearman_rho_test" header="Spearman ρ" sortable>
            <template #body="{ data: row }">
              <span class="font-mono tabular-nums">{{ row.spearman_rho_test?.toFixed(4) ?? '—' }}</span>
            </template>
          </Column>
          <Column field="frac_chem_acc_test" header="% < 43 meV" sortable>
            <template #body="{ data: row }">
              <span class="font-mono tabular-nums">
                {{ row.frac_chem_acc_test != null ? (row.frac_chem_acc_test * 100).toFixed(1) + '%' : '—' }}
              </span>
            </template>
          </Column>
          <Column field="n_params" header="Parâmetros" sortable>
            <template #body="{ data: row }">
              <span class="font-mono tabular-nums">
                {{ row.n_params != null ? row.n_params.toLocaleString('pt-BR') : '—' }}
              </span>
            </template>
          </Column>
          <Column field="n_seeds" header="Seeds">
            <template #body="{ data: row }">
              {{ row.is_multiseed ? `n=${row.n_seeds}` : '—' }}
            </template>
          </Column>
        </DataTable>
      </div>
    </template>
  </section>
</template>
