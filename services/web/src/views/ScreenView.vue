<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { useToast } from "primevue/usetoast";
import Message from "primevue/message";
import Skeleton from "primevue/skeleton";
import ProgressBar from "primevue/progressbar";

import PageHeader from "@/components/PageHeader.vue";
import StatPill from "@/components/StatPill.vue";
import SectionCard from "@/components/SectionCard.vue";
import ScreenForm from "@/components/ScreenForm.vue";
import ResultsTable from "@/components/ResultsTable.vue";
import ParityScatter from "@/components/ParityScatter.vue";
import EmptyState from "@/components/EmptyState.vue";
import SectionLabel from "@/components/SectionLabel.vue";
import { useScreen, useStats } from "@/composables";
import { useScreenStore } from "@/stores/screen";

const toast = useToast();
const store = useScreenStore();
const { lastResult, hasResult, form } = storeToRefs(store);

const screenMutation = useScreen();
const stats = useStats();

const submitting = computed(() => screenMutation.isPending.value);
const errorMsg = computed(() => screenMutation.error.value?.message);

function onSubmit() {
  screenMutation.mutate(
    { ...form.value },
    {
      onSuccess: (data) => {
        toast.add({
          severity: "success",
          summary: `${data.n_candidates} materiais encontrados`,
          detail:
            data.n_candidates === 0
              ? "Nenhum material contém todos esses elementos"
              : `Mostrando os ${data.rows.length} mais promissores`,
          life: 3500,
        });
      },
      onError: (err) => {
        toast.add({
          severity: "error",
          summary: "Não foi possível fazer a busca",
          detail: err.message,
          life: 6000,
        });
      },
    },
  );
}

function onReset() {
  store.resetForm();
  toast.add({ severity: "info", summary: "Configurações restauradas", life: 2000 });
}
</script>

<template>
  <section class="mx-auto max-w-7xl space-y-6 px-4 py-6 sm:space-y-8 sm:px-6 sm:py-10">
    <PageHeader
      icon="pi-search"
      title="Triagem de catalisadores"
      subtitle="Selecione os elementos da composição e ranqueie os candidatos pela proximidade do ótimo de Sabatier (ΔGₕ ≈ 0)."
    >
      <template v-if="stats.data.value">
        <StatPill
          icon="pi-database"
          :value="stats.data.value.n_structures.toLocaleString('pt-BR')"
          label="estruturas"
        />
        <StatPill
          icon="pi-check-square"
          :value="stats.data.value.n_test_canonical.toLocaleString('pt-BR')"
          label="conjunto de teste"
        />
        <StatPill
          icon="pi-table"
          :value="stats.data.value.available_elements.length"
          label="elementos"
        />
        <StatPill
          icon="pi-microchip-ai"
          :value="stats.data.value.available_models.length"
          label="modelos"
        />
      </template>
      <Skeleton v-else width="20rem" height="2rem" />
    </PageHeader>

    <ScreenForm :submitting="submitting" @submit="onSubmit" @reset="onReset" />

    <Message v-if="errorMsg" severity="error" :closable="false">
      {{ errorMsg }}
    </Message>

    <section v-if="submitting">
      <div
        class="space-y-4 rounded-2xl border border-surface-200 bg-surface-0 p-5 dark:border-surface-800 dark:bg-surface-950"
      >
        <div
          class="flex items-center gap-2 text-sm font-medium text-surface-700 dark:text-surface-200"
        >
          <i class="pi pi-spin pi-spinner text-primary-500" />
          Buscando materiais…
        </div>
        <ProgressBar mode="indeterminate" style="height: 6px" />
        <div class="space-y-3 pt-1">
          <Skeleton height="2.5rem" />
          <Skeleton v-for="i in 8" :key="i" height="2.25rem" />
        </div>
      </div>
    </section>

    <template v-else-if="hasResult && lastResult">
      <ResultsTable :result="lastResult" />

      <section class="space-y-4">
        <SectionLabel icon="pi-chart-scatter" title="Paridade entre predição e referência de DFT" />
        <ParityScatter
          :y-true="lastResult.rows.map((r) => r.dG_dft)"
          :y-pred="lastResult.rows.map((r) => r.dG_pred)"
          :title="`ΔGₕ previsto vs. DFT (${lastResult.rows.length} estruturas)`"
          color="#10b981"
        />
      </section>
    </template>

    <SectionCard v-else :padded="false">
      <EmptyState
        icon="pi-sparkles"
        title="Nenhuma triagem ainda"
        description="Selecione os elementos na tabela periódica, defina o número de candidatos e execute a triagem para obter o ranqueamento."
      />
    </SectionCard>
  </section>
</template>
