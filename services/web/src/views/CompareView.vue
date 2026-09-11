<script setup lang="ts">
import { computed } from "vue";
import Skeleton from "primevue/skeleton";
import Message from "primevue/message";
import Button from "primevue/button";

import ModelTable from "@/components/ModelTable.vue";
import AppFooter from "@/components/AppFooter.vue";
import SectionLabel from "@/components/SectionLabel.vue";
import CumulativeErrorChart from "@/components/CumulativeErrorChart.vue";
import ParityScatter from "@/components/ParityScatter.vue";
import { useComparison, useComparisonPredictions } from "@/composables";
import { modelColor } from "@/charts/palette";
import { useDark } from "@vueuse/core";

const isDark = useDark();
const { data, isLoading, error, refetch } = useComparison();
const { data: preds, isLoading: loadingPreds } = useComparisonPredictions();

/** Domínio comum aos quatro gráficos de paridade, com as caudas recortadas: no
 *  domínio completo um outlier de 5 eV do SchNet achataria os outros três. */
const PARITY_DOMAIN: [number, number] = [-1.2, 1.2];

const outliers = computed(() =>
  (preds.value?.models ?? []).map((m) => ({
    display: m.display,
    n: m.y_true.filter(
      (v, i) =>
        v < PARITY_DOMAIN[0] ||
        v > PARITY_DOMAIN[1] ||
        (m.y_pred[i] ?? 0) < PARITY_DOMAIN[0] ||
        (m.y_pred[i] ?? 0) > PARITY_DOMAIN[1],
    ).length,
  })),
);
</script>

<template>
  <section class="mx-auto max-w-7xl space-y-6 px-4 py-6 sm:px-6">
    <header>
      <h1 class="text-base font-semibold tracking-tight text-surface-900 dark:text-surface-0">
        Modelos
      </h1>
      <p class="mt-1 max-w-3xl text-xs text-surface-500">
        Os quatro modelos usam as mesmas 5.860 estruturas do Catalysis Hub: 4.220 para treino, 468
        para ajuste e 1.172 para teste. As métricas abaixo são do conjunto de teste. O limiar de
        acurácia química adotado é 43 meV.
      </p>
    </header>

    <div v-if="isLoading" class="space-y-4">
      <Skeleton height="3rem" />
      <Skeleton v-for="i in 4" :key="i" height="3.5rem" />
    </div>

    <Message v-else-if="error" severity="error" :closable="false">
      <div class="flex items-center justify-between gap-3">
        <span>Não foi possível carregar a comparação: {{ error.message }}</span>
        <Button label="Tentar de novo" icon="pi pi-refresh" size="small" @click="refetch()" />
      </div>
    </Message>

    <template v-else-if="data">
      <ModelTable :models="data.models" />

      <section class="space-y-3">
        <SectionLabel title="Distribuição do erro" />
        <CumulativeErrorChart
          v-if="preds"
          :models="preds.models"
          :chemical-accuracy-ev="preds.chemical_accuracy_eV"
        />
        <Skeleton v-else-if="loadingPreds" height="20rem" />
      </section>

      <section v-if="preds" class="space-y-3">
        <SectionLabel title="Predito vs. referência, mesma escala" />
        <p class="-mt-1 text-xs text-surface-500">
          Um seed por modelo, não a média dos cinco, então o MAE aqui difere do da tabela para
          SchNet e MACE, que são multi-seed.
        </p>
        <div class="grid gap-4 lg:grid-cols-2">
          <div v-for="(m, i) in preds.models" :key="m.display" class="min-w-0">
            <ParityScatter
              :y-true="m.y_true"
              :y-pred="m.y_pred"
              :title="m.display"
              :color="modelColor(i, isDark)"
              :domain="PARITY_DOMAIN"
            />
            <p v-if="outliers[i]?.n" class="mt-1 text-2xs text-surface-500">
              {{ outliers[i]!.n }} fora de ±1,2 eV
            </p>
          </div>
        </div>
      </section>
    </template>

    <AppFooter />
  </section>
</template>
