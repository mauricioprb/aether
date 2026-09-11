<script setup lang="ts">
import { computed, onUnmounted, ref } from "vue";
import ProgressBar from "primevue/progressbar";

const props = withDefaults(
  defineProps<{
    elements: string[];
    candidates: number | null;
    excludeTrain: boolean;
    overlay?: boolean;
  }>(),
  { overlay: false },
);

/** ~28s na primeira triagem (carrega o checkpoint do MACE), 2 a 5s nas
 *  seguintes. Passados 8s a espera deixa de parecer normal. */
const slow = ref(false);
const timer = window.setTimeout(() => (slow.value = true), 8000);
onUnmounted(() => window.clearTimeout(timer));

/** Etapas reais, não progresso simulado: a filtragem já terminou de fato, e foi
 *  dela que veio a contagem. As outras duas rodam dentro da mesma requisição,
 *  sem como cronometrar sem inventar número. */
const steps = computed(() => [
  {
    label: "Filtrando candidatas",
    detail: props.candidates != null ? `${props.candidates} estruturas` : null,
    state: "done" as const,
  },
  { label: "Prevendo ΔG_H com ETR e MACE", detail: null, state: "doing" as const },
  { label: "Ranqueando pelo ótimo de Sabatier", detail: null, state: "todo" as const },
]);
</script>

<template>
  <!-- Com resultado anterior, faixa de status acima dele: atualizar a view não é
       modal, então sem sombra, vidro ou camada flutuante. -->
  <div
    v-if="overlay"
    class="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-surface-200 bg-surface-50 px-4 py-3 dark:border-surface-800 dark:bg-surface-900/60"
  >
    <span class="flex items-center gap-2.5 text-sm text-surface-800 dark:text-surface-100">
      <i class="pi pi-spin pi-spinner text-xs text-surface-500" />
      Prevendo ΔG_H com ETR e MACE
    </span>
    <span v-if="candidates" class="font-mono text-xs text-surface-500">
      {{ candidates }} estruturas, {{ elements.join(", ") }}
    </span>
    <span v-if="slow" class="text-xs text-surface-500">
      carregando o modelo na memória, cerca de meio minuto
    </span>
    <ProgressBar mode="indeterminate" style="height: 2px" class="w-full" />
  </div>

  <!-- Sem cartão: na primeira triagem não há nada atrás para separar, então a
       moldura seria contorno em volta de nada. O conteúdo é a área. -->
  <div v-else class="flex min-h-[calc(100vh-12rem)] items-center justify-center px-6">
    <div class="w-full max-w-xl">
      <header class="mb-7">
        <h2 class="text-base font-semibold text-surface-900 dark:text-surface-0">Triando</h2>
        <p class="mt-1.5 text-sm text-surface-500">
          <span class="font-mono">{{ elements.join(", ") }}</span>
          <template v-if="excludeTrain">, só no conjunto de teste</template>
        </p>
      </header>

      <ol class="space-y-5">
        <li v-for="st in steps" :key="st.label" class="flex items-start gap-3.5 text-sm">
          <span class="mt-px grid h-5 w-5 shrink-0 place-items-center">
            <i
              v-if="st.state === 'done'"
              class="pi pi-check text-sm text-primary-600 dark:text-primary-400"
            />
            <i
              v-else-if="st.state === 'doing'"
              class="pi pi-spin pi-spinner text-sm text-surface-500"
            />
            <span v-else class="h-2 w-2 rounded-full bg-surface-300 dark:bg-surface-700"></span>
          </span>
          <span
            class="min-w-0 flex-1"
            :class="
              st.state === 'todo'
                ? 'text-surface-400 dark:text-surface-600'
                : 'text-surface-800 dark:text-surface-100'
            "
          >
            {{ st.label }}
            <span v-if="st.detail" class="font-mono text-xs text-surface-500">
              , {{ st.detail }}
            </span>
          </span>
        </li>
      </ol>

      <ProgressBar mode="indeterminate" style="height: 2px" class="mt-8" />

      <p v-if="slow" class="mt-5 max-w-md text-xs leading-relaxed text-surface-500">
        A primeira triagem depois de iniciar o serviço carrega o modelo MACE na memória e leva cerca
        de meio minuto. As seguintes levam poucos segundos.
      </p>
    </div>
  </div>
</template>
