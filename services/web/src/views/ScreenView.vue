<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { useRoute, useRouter } from "vue-router";
import { useStorage } from "@vueuse/core";
import { useToast } from "primevue/usetoast";
import Message from "primevue/message";

import ScreenForm from "@/components/ScreenForm.vue";
import ResultsTable from "@/components/ResultsTable.vue";
import ScreeningProgress from "@/components/ScreeningProgress.vue";
import AppFooter from "@/components/AppFooter.vue";
import VolcanoPlot from "@/components/VolcanoPlot.vue";
import { useCount, useScreen } from "@/composables";
import { useScreenStore } from "@/stores/screen";
import type { ModelName, ScreenRequest } from "@/api";

const toast = useToast();
const route = useRoute();
const router = useRouter();
const store = useScreenStore();
const { lastResult, form } = storeToRefs(store);

const selectedElements = computed(() => form.value.elements);
const excludeTrain = computed(() => form.value.exclude_train);
const { data: count } = useCount(selectedElements, excludeTrain);

const screenMutation = useScreen();

const submitting = computed(() => screenMutation.isPending.value);
const errorMsg = computed(() => screenMutation.error.value?.message);
const result = computed(() => lastResult.value);
const hasRows = computed(() => (result.value?.rows.length ?? 0) > 0);

/** Sem resultado a consulta é a tela inteira, centralizada. Ao triar, ela é
 *  promovida para a lateral e o ranqueamento ocupa o resto. Uma transição só
 *  governa as duas coisas: as colunas do grid. */
const docked = computed(() => submitting.value || result.value !== null);

// Persiste a escolha do usuário entre sessões, como o formulário já faz.
const sidebarOpen = useStorage("aether.sidebar.open", true);

/** A entrada da sidebar é disparo único, e só na promoção causada por um clique
 *  em Triar. Recarregar a página com um link já triado não é uma promoção: a
 *  consulta já estava aberta, então ela apenas aparece. */
const entering = ref(false);
const skipEntrance = ref(false);

watch(docked, (now) => {
  if (!now) return;
  if (skipEntrance.value) {
    skipEntrance.value = false;
    return;
  }
  entering.value = true;
  window.setTimeout(() => (entering.value = false), 560);
});

const MODELS: ModelName[] = ["etr_emb", "stagea", "ensemble"];

function queryFromForm(f: ScreenRequest) {
  return {
    el: f.elements.join(","),
    modelo: f.model,
    top: String(f.top),
    teste: f.exclude_train ? "1" : "0",
  };
}

/** A URL é entrada não confiável: valida antes de virar estado do formulário. */
function formFromQuery(): Partial<ScreenRequest> | null {
  const raw = route.query.el;
  if (typeof raw !== "string" || !raw.trim()) return null;
  const elements = raw
    .split(",")
    .map((e) => e.trim())
    .filter(Boolean)
    .slice(0, 20);
  if (elements.length === 0) return null;

  const model = MODELS.find((m) => m === route.query.modelo);
  const top = Number(route.query.top);
  return {
    elements,
    ...(model ? { model } : {}),
    ...(Number.isFinite(top) ? { top: Math.min(100, Math.max(5, Math.trunc(top))) } : {}),
    exclude_train: route.query.teste !== "0",
  };
}

function onSubmit() {
  // Só o erro vira toast: o sucesso já aparece na tabela ao lado.
  screenMutation.mutate(
    { ...form.value },
    {
      onSuccess: () => router.replace({ query: queryFromForm(form.value) }),
      onError: (err) => {
        toast.add({
          severity: "error",
          summary: "Não foi possível concluir a triagem",
          detail: err.message,
          life: 6000,
        });
      },
    },
  );
}

onMounted(() => {
  // Só link compartilhado entra triando. Sem ele a tela abre no estado herói,
  // que já é um estado de trabalho: o mapa é a informação.
  const fromUrl = formFromQuery();
  if (fromUrl) {
    skipEntrance.value = true;
    store.setForm(fromUrl);
    onSubmit();
  }
});
</script>

<template>
  <div
    class="stage h-full"
    :class="[
      docked ? 'is-docked' : 'is-hero',
      docked && !sidebarOpen ? 'is-collapsed' : '',
      entering ? 'is-entering' : '',
    ]"
  >
    <aside
      id="painel-consulta"
      class="panel min-w-0"
      :class="docked ? 'border-r border-surface-200 dark:border-surface-800' : ''"
    >
      <ScreenForm
        :submitting="submitting"
        :variant="docked ? 'docked' : 'hero'"
        @submit="onSubmit"
        @reset="store.resetForm()"
      />
    </aside>

    <section class="results min-w-0">
      <button
        v-if="docked"
        type="button"
        class="mb-3 flex items-center gap-2 rounded px-2 py-1 text-xs text-surface-500 transition hover:bg-surface-100 hover:text-surface-800 dark:hover:bg-surface-800 dark:hover:text-surface-200"
        :aria-expanded="sidebarOpen"
        aria-controls="painel-consulta"
        @click="sidebarOpen = !sidebarOpen"
      >
        <i
          :class="['pi text-xs', sidebarOpen ? 'pi-angle-double-left' : 'pi-angle-double-right']"
        />
        {{ sidebarOpen ? "Ocultar consulta" : "Mostrar consulta" }}
      </button>

      <Message v-if="errorMsg" severity="error" :closable="false" class="mb-4">
        {{ errorMsg }}
      </Message>

      <!-- A faixa de status entra no fluxo, acima do resultado anterior, que
           continua legível: nada de camada flutuante nem desfoque. -->
      <ScreeningProgress
        v-if="submitting"
        class="mb-5"
        :elements="form.elements"
        :candidates="count?.n_candidates ?? result?.n_candidates ?? null"
        :exclude-train="form.exclude_train"
        :overlay="Boolean(result && hasRows)"
      />

      <div
        v-if="result && hasRows"
        class="space-y-5 transition-opacity duration-200"
        :class="submitting ? 'pointer-events-none opacity-55' : 'reveal'"
      >
        <ResultsTable :result="result" />
        <VolcanoPlot :rows="result.rows" :pool="result.pool_dG_pred" />
      </div>

      <div
        v-else-if="!submitting"
        class="flex min-h-[22rem] items-center justify-center px-6 text-center"
      >
        <p class="max-w-xs text-sm text-surface-500">
          Nenhuma estrutura contém todos os elementos selecionados.
        </p>
      </div>

      <AppFooter v-if="!submitting" />
    </section>
  </div>
</template>

<style scoped>
/* Flex, não grid: recolher a sidebar é margem negativa numa largura fixa, então
   o conteúdo interno nunca reflowa durante o movimento. Animar faixa de grid
   falha aqui porque o mapa tem min-width e se reorganiza enquanto a faixa
   cresce. */
.stage {
  display: flex;
  overflow: hidden;
  min-height: 0;
}

.panel {
  flex: none;
  overflow-y: auto;
  min-height: 0;
  padding: 1rem 1.25rem;
  transition: margin-left 340ms cubic-bezier(0.22, 1, 0.36, 1);
}

.results {
  flex: 1 1 0;
  min-width: 0;
  overflow-y: auto;
  padding: 1.25rem;
}

/* Herói: a consulta ocupa o centro e o resultado ainda não existe. */
.is-hero {
  justify-content: center;
}

.is-hero .panel {
  width: min(62rem, 100%);
  /* Centralizado no eixo vertical. O salto que isto causava na troca deixou de
     ser visível quando a entrada virou sidebar-in: ela começa em opacity 0 e
     fora da tela, então o reposicionamento acontece com o painel invisível.
     max-height com a rolagem própria do painel evita corte no topo quando o
     conteúdo passa da altura da viewport. */
  align-self: center;
  max-height: 100%;
  padding-block: 2rem;
}

.is-hero .results {
  display: none;
}

/* Acoplado: largura fixa de sidebar. */
.is-docked .panel {
  width: 41rem;
}

.is-collapsed .panel {
  margin-left: -41rem;
}

/* Entrada, disparo único na promoção herói → acoplado. */
.is-entering .panel {
  animation: sidebar-in 520ms cubic-bezier(0.22, 1, 0.36, 1) both;
}

.is-entering .results {
  animation: chrome 380ms ease 220ms both;
}

@keyframes sidebar-in {
  from {
    transform: translateX(-100%);
    opacity: 0;
  }
  to {
    transform: none;
    opacity: 1;
  }
}

@keyframes chrome {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.reveal {
  animation: reveal 620ms cubic-bezier(0.22, 1, 0.36, 1) 200ms both;
}

@keyframes reveal {
  from {
    opacity: 0;
    transform: translateY(14px);
    filter: blur(6px);
  }
  to {
    opacity: 1;
    transform: none;
    filter: blur(0);
  }
}

@media (max-width: 1023px) {
  .stage {
    flex-direction: column;
    overflow-y: auto;
  }
  .panel,
  .results {
    overflow: visible;
  }
  .is-hero .panel,
  .is-docked .panel {
    width: 100%;
    padding-block: 1rem;
  }
  .is-collapsed .panel {
    margin-left: 0;
    display: none;
  }
}

@media (prefers-reduced-motion: reduce) {
  .panel {
    transition: none;
  }
  .is-entering .panel,
  .is-entering .results,
  .reveal {
    animation: none;
  }
}
</style>
