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
    name: "Método rápido",
    value: "96%",
    meta: "o mais certeiro, resposta em segundos",
    best: true,
  },
  {
    name: "Método detalhado",
    value: "96%",
    meta: "analisa a estrutura completa do material",
    best: false,
  },
  {
    name: "Método simples",
    value: "93%",
    meta: "usa poucas características do material",
    best: false,
  },
  { name: "Método experimental", value: "91%", meta: "aprende tudo do zero", best: false },
];

const features = [
  {
    icon: "pi-search",
    title: "Encontre materiais",
    text: "Escolha os elementos químicos que te interessam e veja os materiais mais promissores, do melhor ao pior.",
    to: "/screen",
    cta: "Começar busca",
  },
  {
    icon: "pi-chart-bar",
    title: "Compare os métodos",
    text: "Veja qual método de inteligência artificial acerta mais nas previsões e por quê.",
    to: "/compare",
    cta: "Ver comparação",
  },
  {
    icon: "pi-info-circle",
    title: "Sobre o projeto",
    text: "Conheça a base com quase 6 mil materiais reais usada para fazer as previsões.",
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
      title="HIDRA: materiais para gerar hidrogênio verde"
      subtitle="Usamos inteligência artificial para prever quais materiais funcionam melhor para gerar hidrogênio, um combustível limpo. Os melhores prendem o hidrogênio na medida certa: nem forte, nem fraco demais."
    >
      <template #actions>
        <RouterLink to="/screen" class="block w-full sm:w-auto">
          <Button
            label="Começar busca"
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
          label="materiais"
        />
        <StatPill
          icon="pi-check-square"
          :value="stats.data.value.n_test_canonical.toLocaleString('pt-BR')"
          label="amostras de teste"
        />
        <StatPill
          icon="pi-table"
          :value="stats.data.value.available_elements.length"
          label="elementos"
        />
        <StatPill
          icon="pi-microchip-ai"
          :value="stats.data.value.available_models.length"
          label="métodos"
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
              custo dos materiais que aceleram a reação, e é aí que o HIDRA ajuda: encontrando bons
              materiais mais rápido e mais barato.
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
              trabalho da nanociência e da nanotecnologia, e é o terreno onde o HIDRA atua para
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
        title="O que já alcançamos"
        hint="o quanto cada método acerta"
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
