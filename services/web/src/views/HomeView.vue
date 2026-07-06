<script setup lang="ts">
import { RouterLink } from "vue-router";
import Button from "primevue/button";
import Skeleton from "primevue/skeleton";

import PageHeader from "@/components/PageHeader.vue";
import StatPill from "@/components/StatPill.vue";
import SectionLabel from "@/components/SectionLabel.vue";
import { useStats } from "@/composables";

const stats = useStats();

const results = [
  {
    name: "ETR + embeddings",
    value: "R² 0,96",
    meta: "árvores sobre a representação do MACE-MP-0",
    best: true,
  },
  {
    name: "MACE (cabeça)",
    value: "R² 0,96",
    meta: "cabeça de regressão sobre o MACE-MP-0 congelado",
    best: false,
  },
  {
    name: "ETR + descritores",
    value: "R² 0,93",
    meta: "dez descritores físico-químicos",
    best: false,
  },
  { name: "SchNet", value: "R² 0,91", meta: "GNN treinada do zero", best: false },
];

const features = [
  {
    icon: "pi-search",
    title: "Triagem",
    text: "Selecione os elementos da composição e ranqueie os candidatos pela proximidade do ótimo de Sabatier.",
    to: "/screen",
    cta: "Iniciar triagem",
  },
  {
    icon: "pi-chart-bar",
    title: "Comparação de modelos",
    text: "Confronte os quatro modelos sob o mesmo conjunto de teste e protocolo de avaliação.",
    to: "/compare",
    cta: "Ver comparação",
  },
  {
    icon: "pi-info-circle",
    title: "Sobre",
    text: "Conheça a base curada de 5.860 estruturas e a metodologia do estudo.",
    to: "/about",
    cta: "Saiba mais",
  },
];
</script>

<template>
  <section class="mx-auto max-w-7xl space-y-6 px-4 py-6 sm:space-y-8 sm:px-6 sm:py-10">
    <PageHeader
      stack
      icon="pi-bolt"
      title="AETHER — predição de catalisadores para a HER"
      subtitle="Sistema de apoio à triagem de catalisadores para a reação de evolução de hidrogênio (HER), com redes neurais de grafos e aprendizado por transferência. Os melhores candidatos têm energia livre de adsorção de hidrogênio próxima de zero (critério de Sabatier)."
    >
      <template #actions>
        <RouterLink to="/screen" class="block w-full sm:w-auto">
          <Button
            label="Iniciar triagem"
            icon="pi pi-arrow-right"
            icon-pos="right"
            class="w-full sm:w-auto"
          />
        </RouterLink>
      </template>
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
      <Skeleton v-else width="24rem" height="2rem" />
    </PageHeader>

    <div class="grid gap-4 md:grid-cols-3">
      <RouterLink
        v-for="f in features"
        :key="f.to"
        :to="f.to"
        class="group relative flex h-full flex-col overflow-hidden rounded-2xl border border-surface-200 bg-surface-0 p-5 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md dark:border-surface-800 dark:bg-surface-950"
      >
        <span
          class="relative mb-4 grid h-11 w-11 place-items-center rounded-xl bg-primary-500/10 text-primary-600 ring-1 ring-primary-500/20 dark:text-primary-400"
        >
          <i :class="['pi', f.icon, 'text-lg']" />
        </span>
        <h3 class="relative text-base font-semibold text-surface-900 dark:text-surface-0">
          {{ f.title }}
        </h3>
        <p
          class="relative mt-2 flex-1 text-sm leading-relaxed text-surface-600 dark:text-surface-300"
        >
          {{ f.text }}
        </p>
        <div
          class="relative mt-4 flex items-center gap-1.5 text-sm font-medium text-primary-600 dark:text-primary-400"
        >
          {{ f.cta }}
          <i class="pi pi-arrow-right text-xs transition-transform group-hover:translate-x-1" />
        </div>
      </RouterLink>
    </div>

    <section
      class="rounded-2xl border border-surface-200 bg-surface-0 p-6 shadow-sm dark:border-surface-800 dark:bg-surface-950"
    >
      <SectionLabel icon="pi-globe" title="Por que isso importa" class="mb-5" />
      <div class="grid gap-5 md:grid-cols-2">
        <div class="flex gap-3">
          <span
            class="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-primary-500/10 text-primary-600 dark:text-primary-400"
          >
            <i class="pi pi-bolt text-base" />
          </span>
          <div>
            <h3 class="text-sm font-semibold text-surface-900 dark:text-surface-0">
              Hidrogênio verde
            </h3>
            <p class="mt-1 text-sm leading-relaxed text-surface-600 dark:text-surface-300">
              É o hidrogênio produzido a partir da água com energia renovável, sem emitir carbono.
              Pode substituir combustíveis fósseis na indústria e no transporte. O maior gargalo é o
              custo dos catalisadores que aceleram a reação, e é aí que o AETHER atua: identificando
              bons catalisadores mais rápido e a menor custo.
            </p>
          </div>
        </div>
        <div class="flex gap-3">
          <span
            class="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-primary-500/10 text-primary-600 dark:text-primary-400"
          >
            <i class="pi pi-sitemap text-base" />
          </span>
          <div>
            <h3 class="text-sm font-semibold text-surface-900 dark:text-surface-0">
              Nanociência e nanotecnologia
            </h3>
            <p class="mt-1 text-sm leading-relaxed text-surface-600 dark:text-surface-300">
              A reação acontece na superfície dos materiais, numa escala de bilionésimos de metro.
              Nessa escala, pequenas mudanças mudam tudo. Estudar e projetar a matéria assim é o
              trabalho da nanociência e da nanotecnologia, e é o terreno onde o AETHER atua para
              tornar a produção de hidrogênio mais eficiente.
            </p>
          </div>
        </div>
      </div>
    </section>

    <section
      class="rounded-2xl border border-surface-200 bg-surface-0 p-6 shadow-sm dark:border-surface-800 dark:bg-surface-950"
    >
      <SectionLabel
        icon="pi-flag"
        title="Desempenho dos modelos"
        hint="R² no conjunto de teste"
        class="mb-5"
      />
      <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <div
          v-for="r in results"
          :key="r.name"
          class="relative rounded-xl border p-4 transition"
          :class="
            r.best
              ? 'border-primary-500/40 bg-primary-500/5 ring-1 ring-primary-500/20 dark:border-primary-500/30'
              : 'border-surface-200 bg-surface-50 dark:border-surface-800 dark:bg-surface-900/60'
          "
        >
          <div class="flex items-center justify-between gap-2">
            <span class="text-2xs font-medium uppercase tracking-wide text-surface-500">{{
              r.name
            }}</span>
            <i v-if="r.best" class="pi pi-star-fill text-2xs text-amber-500" title="melhor" />
          </div>
          <div
            class="mt-1.5 font-mono text-xl font-semibold tabular-nums"
            :class="
              r.best
                ? 'text-primary-600 dark:text-primary-400'
                : 'text-surface-900 dark:text-surface-0'
            "
          >
            {{ r.value }}
          </div>
          <div class="mt-0.5 text-2xs text-surface-500">{{ r.meta }}</div>
        </div>
      </div>
    </section>
  </section>
</template>
