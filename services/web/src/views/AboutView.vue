<script setup lang="ts">
import PageHeader from "@/components/PageHeader.vue";
import SectionCard from "@/components/SectionCard.vue";

const tech = [
  { label: "Backend", value: "FastAPI, Python 3.14" },
  { label: "Modelos", value: "MACE-MP-0, ETR, SchNet" },
  { label: "Frontend", value: "Vue 3, Vite 7, TypeScript" },
  { label: "UI", value: "PrimeVue 4, Tailwind 4" },
  { label: "Charts", value: "ECharts 6" },
  { label: "Deploy", value: "Docker Compose" },
];

const pipeline = [
  {
    step: 1,
    title: "Seleção dos dados",
    desc: "Partimos de 60 mil reações reais e ficamos com quase 6 mil materiais bem documentados.",
  },
  {
    step: 2,
    title: "Leitura dos materiais",
    desc: "A inteligência artificial transforma cada material num conjunto de números que descreve sua estrutura.",
  },
  {
    step: 3,
    title: "Aprendizado",
    desc: "Os métodos aprendem a prever o desempenho a partir desses materiais conhecidos.",
  },
  {
    step: 4,
    title: "Busca",
    desc: "Você escolhe os elementos e recebe os materiais mais promissores, do melhor ao pior.",
  },
];
</script>

<template>
  <section class="mx-auto max-w-7xl space-y-6 px-4 py-6 sm:space-y-8 sm:px-6 sm:py-10">
    <PageHeader
      icon="pi-info-circle"
      title="Sobre o HIDRA"
      subtitle="Uma ferramenta que usa inteligência artificial para encontrar materiais que ajudam a gerar hidrogênio."
    />

    <SectionCard title="O que faz" icon="pi-bookmark">
      <p class="text-sm leading-relaxed text-surface-700 dark:text-surface-300">
        O hidrogênio é um combustível limpo, mas produzi-lo exige materiais especiais que aceleram a
        reação. O HIDRA usa inteligência artificial para prever quais materiais funcionam melhor,
        sem precisar testá-los um a um no laboratório. Os melhores materiais prendem o hidrogênio na
        medida certa: nem forte demais, nem fraco demais. Assim, em vez de meses de experimentos, a
        busca leva segundos.
      </p>
      <p class="mt-3 text-sm leading-relaxed text-surface-700 dark:text-surface-300">
        Estes materiais atuam em escala nanométrica (bilionésimos de metro), onde pequenas mudanças
        na superfície alteram completamente o comportamento. Por isso o HIDRA nasce no campo da
        <strong>Nanociência</strong>, que estuda e projeta a matéria nessa escala.
      </p>
    </SectionCard>

    <div class="grid gap-4 md:grid-cols-2">
      <SectionCard title="Tecnologias" icon="pi-code">
        <dl class="space-y-2.5 text-sm">
          <div v-for="t in tech" :key="t.label" class="flex items-baseline justify-between gap-3">
            <dt class="text-xs font-medium uppercase tracking-wide text-surface-500">
              {{ t.label }}
            </dt>
            <dd class="font-mono text-xs text-surface-700 dark:text-surface-200">{{ t.value }}</dd>
          </div>
        </dl>
      </SectionCard>

      <SectionCard title="Base de dados" icon="pi-database">
        <dl class="space-y-2.5 text-sm">
          <div class="flex items-baseline justify-between">
            <dt class="text-xs font-medium uppercase tracking-wide text-surface-500">Origem</dt>
            <dd class="font-mono text-xs">Catalysis Hub</dd>
          </div>
          <div class="flex items-baseline justify-between">
            <dt class="text-xs font-medium uppercase tracking-wide text-surface-500">Materiais</dt>
            <dd class="font-mono text-xs">5.860</dd>
          </div>
          <div class="flex items-baseline justify-between">
            <dt class="text-xs font-medium uppercase tracking-wide text-surface-500">
              Treino / ajuste / teste
            </dt>
            <dd class="font-mono text-xs">4.220 / 468 / 1.172</dd>
          </div>
          <div class="flex items-baseline justify-between">
            <dt class="text-xs font-medium uppercase tracking-wide text-surface-500">O que faz</dt>
            <dd class="font-mono text-xs">capturar hidrogênio</dd>
          </div>
        </dl>
      </SectionCard>
    </div>

    <SectionCard title="Como funciona" icon="pi-sitemap" :padded="false">
      <ol class="grid gap-3 p-5 md:grid-cols-2 lg:grid-cols-4">
        <li
          v-for="p in pipeline"
          :key="p.step"
          class="relative rounded-xl border border-surface-200 bg-surface-50 p-4 dark:border-surface-800 dark:bg-surface-900/60"
        >
          <span
            class="absolute -top-2.5 left-3 rounded-md bg-primary-500 px-2 py-0.5 text-2xs font-bold text-white"
          >
            {{ p.step }}
          </span>
          <h4 class="mt-1 text-sm font-semibold text-surface-900 dark:text-surface-0">
            {{ p.title }}
          </h4>
          <p class="mt-1 text-xs text-surface-500">{{ p.desc }}</p>
        </li>
      </ol>
    </SectionCard>

    <SectionCard title="Projeto acadêmico" icon="pi-graduation-cap">
      <p class="text-sm leading-relaxed text-surface-700 dark:text-surface-300">
        O HIDRA é fruto de uma dissertação de <strong>mestrado em Nanociências</strong>, unindo
        química de materiais, simulação atômica e inteligência artificial para acelerar a descoberta
        de catalisadores para a produção de hidrogênio.
      </p>
    </SectionCard>
  </section>
</template>
