<script setup lang="ts">
import { computed } from 'vue'
import DataTable from 'primevue/datatable'
import Column from 'primevue/column'
import Tag from 'primevue/tag'
import Button from 'primevue/button'
import IconField from 'primevue/iconfield'
import InputIcon from 'primevue/inputicon'
import InputText from 'primevue/inputtext'
import { FilterMatchMode } from '@primevue/core/api'
import { ref } from 'vue'
import type { CandidateRow, ScreenResponse } from '@/api'

const props = defineProps<{ result: ScreenResponse }>()

const filters = ref({ global: { value: null as string | null, matchMode: FilterMatchMode.CONTAINS } })

const rows = computed(() => props.result.rows)
const isEnsemble = computed(() => props.result.model === 'ensemble')

function fmt(n: number, digits = 4) {
  return n.toFixed(digits)
}

function chemAccTag(absDg: number) {
  if (absDg < 0.043) return { severity: 'success' as const, text: '< 43 meV' }
  if (absDg < 0.1) return { severity: 'info' as const, text: '< 100 meV' }
  if (absDg < 0.2) return { severity: 'warn' as const, text: '< 200 meV' }
  return { severity: 'danger' as const, text: `${(absDg * 1000).toFixed(0)} meV` }
}

function rowClass(row: CandidateRow) {
  if (row.abs_dG_pred < 0.043) return 'aether-row-best'
  return ''
}

function downloadCsv() {
  const headers = [
    'rank', 'chemical_formula', 'composition', 'facet', 'site_type',
    'dG_pred', 'delta_G_H', 'abs_dG_pred', 'error_vs_dft',
    ...(isEnsemble.value ? ['dG_pred_etr', 'dG_pred_stagea'] : []),
    'id',
  ]
  const lines = rows.value.map((r, i) => [
    i + 1, r.chemical_formula, r.composition, r.facet, r.site_type,
    r.dG_pred, r.delta_G_H, r.abs_dG_pred, r.error_vs_dft,
    ...(isEnsemble.value ? [r.dG_pred_etr ?? '', r.dG_pred_stagea ?? ''] : []),
    r.id,
  ].join(','))
  const csv = [headers.join(','), ...lines].join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `aether_${props.result.model}_${props.result.elements.join('-')}_top${props.result.top}.csv`
  a.click()
  URL.revokeObjectURL(url)
}
</script>

<template>
  <div
    class="rounded-xl border border-surface-200 bg-surface-0 shadow-sm dark:border-surface-800 dark:bg-surface-950"
  >
    <header class="flex flex-wrap items-center justify-between gap-3 border-b border-surface-200 px-5 py-3 dark:border-surface-800">
      <div>
        <div class="text-sm font-semibold">
          {{ result.rows.length }} resultados
          <span class="text-surface-500 font-normal">
            de {{ result.n_candidates }} candidatos contendo
            <span v-for="el in result.elements" :key="el" class="ml-1 rounded bg-surface-100 px-1.5 py-0.5 text-xs font-medium dark:bg-surface-800">{{ el }}</span>
          </span>
        </div>
        <div class="mt-0.5 text-xs text-surface-500">
          Ordenado por |ΔG_H_pred| ascendente · modelo
          <span class="font-medium text-surface-700 dark:text-surface-200">{{ result.model }}</span>
          <span v-if="result.exclude_train" class="ml-2 text-[10px] rounded bg-surface-100 px-1.5 py-0.5 uppercase tracking-wide dark:bg-surface-800">test only</span>
        </div>
      </div>
      <div class="flex items-center gap-2">
        <IconField>
          <InputIcon class="pi pi-search" />
          <InputText
            v-model="filters.global.value"
            placeholder="Buscar fórmula, sítio…"
            size="small"
          />
        </IconField>
        <Button label="CSV" icon="pi pi-download" size="small" severity="secondary" outlined @click="downloadCsv" />
      </div>
    </header>

    <DataTable
      v-model:filters="filters"
      :value="rows"
      :global-filter-fields="['chemical_formula', 'composition', 'facet', 'site_type']"
      :row-class="rowClass"
      striped-rows
      paginator
      :rows="15"
      :rows-per-page-options="[10, 15, 25, 50]"
      removable-sort
      sort-mode="single"
      class="!text-sm"
    >
      <Column header="#" header-style="width: 3rem" body-style="font-variant-numeric: tabular-nums; color: var(--p-text-muted-color)">
        <template #body="{ index }">{{ index + 1 }}</template>
      </Column>
      <Column field="chemical_formula" header="Fórmula" sortable>
        <template #body="{ data }">
          <span class="font-mono text-[13px]">{{ data.chemical_formula }}</span>
        </template>
      </Column>
      <Column field="facet" header="Facet" sortable header-style="width: 5rem" />
      <Column field="site_type" header="Sítio" sortable header-style="width: 6rem">
        <template #body="{ data }">
          <span class="capitalize">{{ data.site_type }}</span>
        </template>
      </Column>
      <Column field="dG_pred" header="ΔG_H pred (eV)" sortable header-style="width: 9rem">
        <template #body="{ data }">
          <span class="font-mono tabular-nums">{{ fmt(data.dG_pred) }}</span>
        </template>
      </Column>
      <Column field="abs_dG_pred" header="|ΔG_H|" sortable header-style="width: 8rem">
        <template #body="{ data }">
          <Tag :severity="chemAccTag(data.abs_dG_pred).severity" :value="chemAccTag(data.abs_dG_pred).text" />
        </template>
      </Column>
      <Column field="delta_G_H" header="ΔG_H DFT (eV)" sortable header-style="width: 9rem">
        <template #body="{ data }">
          <span class="font-mono tabular-nums text-surface-500">{{ fmt(data.delta_G_H) }}</span>
        </template>
      </Column>
      <Column field="error_vs_dft" header="Erro pred-DFT (eV)" sortable header-style="width: 10rem">
        <template #body="{ data }">
          <span
            class="font-mono tabular-nums"
            :class="Math.abs(data.error_vs_dft) > 0.1 ? 'text-orange-500' : 'text-surface-500'"
          >{{ data.error_vs_dft >= 0 ? '+' : '' }}{{ fmt(data.error_vs_dft) }}</span>
        </template>
      </Column>
      <Column v-if="isEnsemble" field="dG_pred_etr" header="ETR" sortable header-style="width: 6rem">
        <template #body="{ data }">
          <span class="font-mono tabular-nums text-xs text-surface-500">{{ fmt(data.dG_pred_etr ?? 0, 3) }}</span>
        </template>
      </Column>
      <Column v-if="isEnsemble" field="dG_pred_stagea" header="StageA" sortable header-style="width: 6rem">
        <template #body="{ data }">
          <span class="font-mono tabular-nums text-xs text-surface-500">{{ fmt(data.dG_pred_stagea ?? 0, 3) }}</span>
        </template>
      </Column>
      <Column field="id" header="ID Catalysis Hub" header-style="width: 12rem">
        <template #body="{ data }">
          <span class="font-mono text-[11px] text-surface-400">{{ data.id }}</span>
        </template>
      </Column>
    </DataTable>
  </div>
</template>

<style>
.aether-row-best {
  background-color: color-mix(in oklab, var(--p-primary-500) 6%, transparent) !important;
}
</style>
