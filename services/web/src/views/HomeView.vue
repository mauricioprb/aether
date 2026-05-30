<script setup lang="ts">
import { RouterLink } from 'vue-router'
import Card from 'primevue/card'
import Button from 'primevue/button'

const features = [
  {
    icon: 'pi-search',
    title: 'Triagem por composição',
    text: 'Defina metais e quantidade. Receba os top-N catalisadores ordenados por |ΔG_H|.',
    to: '/screen',
    cta: 'Abrir triagem',
  },
  {
    icon: 'pi-chart-bar',
    title: 'Comparação de modelos',
    text: 'ETR + MACE embeddings, MACE Stage A, SchNet e baseline handcrafted lado-a-lado.',
    to: '/compare',
    cta: 'Comparar',
  },
  {
    icon: 'pi-info-circle',
    title: 'Sobre o dataset',
    text: '5.860 estruturas curadas do Catalysis Hub. Split canônico 4.220/468/1.172.',
    to: '/about',
    cta: 'Detalhes',
  },
]
</script>

<template>
  <section class="mx-auto max-w-6xl px-6 py-12">
    <div class="mb-10 max-w-3xl">
      <h1 class="text-3xl font-semibold tracking-tight text-surface-900 dark:text-surface-0">
        Triagem de catalisadores HER
      </h1>
      <p class="mt-3 text-surface-600 dark:text-surface-300">
        Recomendação de catalisadores para a reação de evolução de hidrogênio (HER) usando representações
        aprendidas pelo MACE-MP-0 + Extra Trees Regressor. Princípio de Sabatier: ΔG_H ≈ 0 é ótimo.
      </p>
      <div class="mt-6 flex gap-3">
        <RouterLink to="/screen">
          <Button label="Começar triagem" icon="pi pi-arrow-right" icon-pos="right" />
        </RouterLink>
        <a href="/docs" target="_blank" rel="noopener">
          <Button label="Documentação da API" icon="pi pi-external-link" severity="secondary" outlined />
        </a>
      </div>
    </div>

    <div class="grid gap-5 md:grid-cols-3">
      <Card v-for="f in features" :key="f.to" class="h-full">
        <template #title>
          <div class="flex items-center gap-3">
            <span
              class="grid h-9 w-9 place-items-center rounded-md bg-primary-50 text-primary-600 dark:bg-primary-950 dark:text-primary-300"
            >
              <i :class="['pi', f.icon]" />
            </span>
            <span class="text-base font-semibold">{{ f.title }}</span>
          </div>
        </template>
        <template #content>
          <p class="text-sm leading-relaxed text-surface-600 dark:text-surface-300">{{ f.text }}</p>
        </template>
        <template #footer>
          <RouterLink :to="f.to">
            <Button :label="f.cta" text size="small" icon="pi pi-arrow-right" icon-pos="right" />
          </RouterLink>
        </template>
      </Card>
    </div>
  </section>
</template>
