<script setup lang="ts">
import { computed } from "vue";
import DataTable from "primevue/datatable";
import Column from "primevue/column";
import Button from "primevue/button";
import IconField from "primevue/iconfield";
import InputIcon from "primevue/inputicon";
import InputText from "primevue/inputtext";
import { FilterMatchMode } from "@primevue/core/api";
import { ref } from "vue";
import { useDark } from "@vueuse/core";
import { dgColor } from "@/charts/palette";
import { num } from "@/format";
import type { CandidateRow, ScreenResponse } from "@/api";

const props = defineProps<{ result: ScreenResponse }>();

const filters = ref({
  global: { value: null as string | null, matchMode: FilterMatchMode.CONTAINS },
});

const isDark = useDark();
const rows = computed(() => props.result.rows);
const isEnsemble = computed(() => props.result.model === "ensemble");

const summary = computed(() => {
  const abs = rows.value.map((r) => r.abs_dG_pred).sort((a, b) => a - b);
  if (abs.length === 0) return null;
  const mid = Math.floor(abs.length / 2);
  return {
    median: abs.length % 2 ? abs[mid]! : (abs[mid - 1]! + abs[mid]!) / 2,
    withinChemAcc: abs.filter((v) => v < CHEM_ACC_EV).length,
    agreeing: rows.value.filter((r) => {
      const sp = spread(r);
      return sp != null && sp <= CHEM_ACC_EV;
    }).length,
  };
});

function fmt(n: number, digits = 3) {
  return num(n, digits);
}

const CHEM_ACC_EV = 0.043;

/** Lado da adsorção, não severidade: o zero de Sabatier é o ótimo, e o sinal
 *  diz se a ligação é forte ou fraca demais. */
/** Quanto os dois modelos discordam nesta estrutura. Discordância acima da
 *  acurácia química significa que a predição não é confiável, mesmo que a média
 *  caia perto do ótimo: é este o motivo de rodar os dois. */
function spread(row: CandidateRow): number | null {
  if (row.dG_pred_etr == null || row.dG_pred_stagea == null) return null;
  return Math.abs(row.dG_pred_etr - row.dG_pred_stagea);
}

function bindingLabel(dg: number) {
  if (Math.abs(dg) < CHEM_ACC_EV) return "ótimo";
  return dg < 0 ? "forte" : "fraca";
}

function rowClass(row: CandidateRow) {
  if (row.abs_dG_pred < CHEM_ACC_EV) return "aether-row-best";
  return "";
}

function downloadCsv() {
  const headers = [
    "rank",
    "chemical_formula",
    "composition",
    "facet",
    "site_type",
    "dG_pred",
    "dE_H_referencia",
    "abs_dG_pred",
    "error_vs_dft",
    ...(isEnsemble.value ? ["dG_pred_etr", "dG_pred_stagea"] : []),
    "id",
  ];
  const lines = rows.value.map((r, i) =>
    [
      i + 1,
      r.chemical_formula,
      r.composition,
      r.facet,
      r.site_type,
      r.dG_pred,
      r.delta_G_H,
      r.abs_dG_pred,
      r.error_vs_dft,
      ...(isEnsemble.value ? [r.dG_pred_etr ?? "", r.dG_pred_stagea ?? ""] : []),
      r.id,
    ].join(","),
  );
  const csv = [headers.join(","), ...lines].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `aether_${props.result.model}_${props.result.elements.join("-")}_top${props.result.top}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}
</script>

<template>
  <div
    class="aether-results reveal-rows overflow-hidden rounded-lg border border-surface-200 bg-surface-0 dark:border-surface-800 dark:bg-surface-950"
  >
    <header
      class="flex flex-col gap-3 border-b border-surface-200 px-5 py-4 sm:flex-row sm:items-end sm:justify-between dark:border-surface-800"
    >
      <div class="min-w-0">
        <h1 class="text-base font-semibold tracking-tight text-surface-900 dark:text-surface-0">
          {{ result.rows.length }} de {{ result.n_candidates }} candidatas
        </h1>
        <p class="mt-0.5 flex flex-wrap items-center gap-x-1.5 gap-y-1 text-xs text-surface-500">
          <span>estruturas com</span>
          <span
            v-for="el in result.elements"
            :key="el"
            class="rounded bg-surface-100 px-1.5 py-0.5 font-mono text-xs text-surface-700 dark:bg-surface-800 dark:text-surface-200"
            >{{ el }}</span
          >
          <span>, as mais próximas do ótimo de Sabatier</span>
          <span v-if="result.exclude_train">, só no conjunto de teste</span>
        </p>
      </div>
      <div class="flex shrink-0 items-center gap-2">
        <IconField>
          <InputIcon class="pi pi-search" />
          <InputText v-model="filters.global.value" placeholder="Filtrar…" size="small" />
        </IconField>
        <Button
          label="CSV"
          icon="pi pi-download"
          size="small"
          severity="secondary"
          outlined
          @click="downloadCsv"
        />
      </div>
    </header>

    <DataTable
      v-model:filters="filters"
      :value="rows"
      :global-filter-fields="['chemical_formula', 'composition', 'facet', 'site_type']"
      :row-class="rowClass"
      paginator
      :rows="15"
      :rows-per-page-options="[10, 15, 25, 50]"
      removable-sort
      sort-mode="single"
      scrollable
      class="aether-dt text-sm!"
    >
      <Column
        header="#"
        header-style="width: 3rem"
        body-style="font-variant-numeric: tabular-nums; color: var(--p-text-muted-color)"
      >
        <template #body="{ index }">{{ index + 1 }}</template>
      </Column>
      <Column field="chemical_formula" header="Fórmula" sortable>
        <template #body="{ data }">
          <span class="font-mono text-sm">{{ data.chemical_formula }}</span>
        </template>
      </Column>
      <Column field="facet" header="Faceta" sortable header-style="width: 5rem" />
      <Column field="site_type" header="Sítio" sortable header-style="width: 6rem">
        <template #body="{ data }">
          <span class="capitalize">{{ data.site_type }}</span>
        </template>
      </Column>
      <Column class="text-right!" field="dG_pred" header="ΔG_H previsto (eV)" sortable>
        <template #body="{ data }">
          <span class="font-mono tabular-nums">{{ fmt(data.dG_pred) }}</span>
        </template>
      </Column>
      <Column field="abs_dG_pred" header="Adsorção" sortable>
        <template #body="{ data }">
          <span class="flex items-center gap-2">
            <span
              class="inline-block h-2 w-2 shrink-0 rounded-full"
              :style="{ backgroundColor: dgColor(data.dG_pred, CHEM_ACC_EV, isDark) }"
            ></span>
            <span class="text-xs text-surface-600 dark:text-surface-300">{{
              bindingLabel(data.dG_pred)
            }}</span>
          </span>
        </template>
      </Column>
      <Column class="text-right!" field="dG_dft" header="Referência (eV)" sortable>
        <template #body="{ data }">
          <span class="font-mono tabular-nums text-surface-500">{{ fmt(data.dG_dft) }}</span>
        </template>
      </Column>
      <Column class="text-right!" field="error_vs_dft" header="Resíduo (eV)" sortable>
        <template #body="{ data }">
          <span
            class="font-mono tabular-nums"
            :class="
              Math.abs(data.error_vs_dft) > 0.1
                ? 'font-medium text-surface-900 dark:text-surface-0'
                : 'text-surface-500'
            "
            >{{ data.error_vs_dft >= 0 ? "+" : "" }}{{ fmt(data.error_vs_dft) }}</span
          >
        </template>
      </Column>
      <Column
        v-if="isEnsemble"
        class="text-right!"
        field="dG_pred_etr"
        header="ETR (eV)"
        sortable
        header-style="width: 6rem"
      >
        <template #body="{ data }">
          <span class="font-mono tabular-nums text-xs text-surface-500">{{
            fmt(data.dG_pred_etr ?? 0, 3)
          }}</span>
        </template>
      </Column>
      <Column
        v-if="isEnsemble"
        class="text-right!"
        field="dG_pred_stagea"
        header="MACE (eV)"
        sortable
        header-style="width: 6rem"
      >
        <template #body="{ data }">
          <span class="font-mono tabular-nums text-xs text-surface-500">{{
            fmt(data.dG_pred_stagea ?? 0, 3)
          }}</span>
        </template>
      </Column>
      <Column
        v-if="isEnsemble"
        class="text-right!"
        field="spread"
        header="Divergência (eV)"
        sortable
      >
        <template #body="{ data }">
          <span
            class="font-mono tabular-nums"
            :class="
              (spread(data) ?? 0) > CHEM_ACC_EV
                ? 'font-medium text-surface-900 dark:text-surface-0'
                : 'text-surface-500'
            "
            :title="
              (spread(data) ?? 0) > CHEM_ACC_EV
                ? 'Os dois modelos discordam acima da acurácia química'
                : 'Os dois modelos concordam'
            "
          >
            {{ spread(data) != null ? fmt(spread(data)!) : "n/d" }}
          </span>
        </template>
      </Column>
    </DataTable>

    <footer
      v-if="summary"
      class="flex flex-wrap items-center gap-x-8 gap-y-1 border-t border-surface-200 bg-surface-50 px-5 py-3 text-sm text-surface-600 dark:border-surface-800 dark:bg-surface-900/50 dark:text-surface-300"
    >
      <span>
        mediana |ΔG_H|
        <b class="ml-1 font-mono font-semibold tabular-nums text-surface-900 dark:text-surface-0"
          >{{ fmt(summary.median) }} eV</b
        >
      </span>
      <span v-if="isEnsemble">
        modelos concordam
        <b class="ml-1 font-mono font-semibold tabular-nums text-surface-900 dark:text-surface-0"
          >{{ summary.agreeing }} de {{ rows.length }}</b
        >
      </span>
      <span>
        no ótimo, |ΔG_H| &lt; 43 meV
        <b class="ml-1 font-mono font-semibold tabular-nums text-surface-900 dark:text-surface-0"
          >{{ summary.withinChemAcc }} de {{ rows.length }}</b
        >
      </span>
    </footer>
  </div>
</template>

<style>
/* Cadência na chegada: as linhas entram em cascata a partir do topo, que é a
   ordem em que elas importam (mais perto do ótimo primeiro). */
.reveal-rows .aether-dt tbody > tr {
  animation: row-in 460ms cubic-bezier(0.22, 1, 0.36, 1) both;
}
.reveal-rows .aether-dt tbody > tr:nth-child(1) {
  animation-delay: 326ms;
}
.reveal-rows .aether-dt tbody > tr:nth-child(2) {
  animation-delay: 352ms;
}
.reveal-rows .aether-dt tbody > tr:nth-child(3) {
  animation-delay: 378ms;
}
.reveal-rows .aether-dt tbody > tr:nth-child(4) {
  animation-delay: 404ms;
}
.reveal-rows .aether-dt tbody > tr:nth-child(5) {
  animation-delay: 430ms;
}
.reveal-rows .aether-dt tbody > tr:nth-child(6) {
  animation-delay: 456ms;
}
.reveal-rows .aether-dt tbody > tr:nth-child(7) {
  animation-delay: 482ms;
}
.reveal-rows .aether-dt tbody > tr:nth-child(8) {
  animation-delay: 508ms;
}
.reveal-rows .aether-dt tbody > tr:nth-child(9) {
  animation-delay: 534ms;
}
.reveal-rows .aether-dt tbody > tr:nth-child(10) {
  animation-delay: 560ms;
}
.reveal-rows .aether-dt tbody > tr:nth-child(11) {
  animation-delay: 586ms;
}
.reveal-rows .aether-dt tbody > tr:nth-child(12) {
  animation-delay: 612ms;
}
.reveal-rows .aether-dt tbody > tr:nth-child(13) {
  animation-delay: 638ms;
}
.reveal-rows .aether-dt tbody > tr:nth-child(14) {
  animation-delay: 664ms;
}
.reveal-rows .aether-dt tbody > tr:nth-child(15) {
  animation-delay: 690ms;
}
.reveal-rows .aether-dt tbody > tr:nth-child(16) {
  animation-delay: 716ms;
}

@keyframes row-in {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: none;
  }
}

@media (prefers-reduced-motion: reduce) {
  .reveal-rows .aether-dt tbody > tr {
    animation: none;
  }
}

.aether-row-best {
  background-color: color-mix(in oklab, var(--p-primary-500) 6%, transparent) !important;
}

.aether-dt .p-datatable-tbody > tr > td {
  padding-block: 0.6rem;
}
.aether-dt .p-datatable-thead > tr > th.text-right\! .p-datatable-column-header-content {
  justify-content: flex-end;
}
.aether-dt .p-datatable-thead > tr > th .p-datatable-column-title {
  white-space: nowrap;
}
.aether-dt .p-datatable-thead > tr > th {
  padding-block: 0.55rem;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.01em;
}

.aether-results .p-paginator {
  background: transparent;
  padding: 0.5rem 0.75rem;
  gap: 2px;
}
.aether-results .p-paginator .p-paginator-page,
.aether-results .p-paginator .p-paginator-first,
.aether-results .p-paginator .p-paginator-prev,
.aether-results .p-paginator .p-paginator-next,
.aether-results .p-paginator .p-paginator-last {
  min-width: 2rem;
  height: 2rem;
  margin: 0;
  border-radius: 0.5rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.aether-results .p-paginator .p-paginator-page.p-paginator-page-selected {
  background: var(--p-primary-color);
  color: var(--p-primary-contrast-color);
}
</style>
