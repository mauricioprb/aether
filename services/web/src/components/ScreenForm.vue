<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { useEventListener } from "@vueuse/core";
import Slider from "primevue/slider";
import InputNumber from "primevue/inputnumber";
import ToggleSwitch from "primevue/toggleswitch";
import Button from "primevue/button";
import Skeleton from "primevue/skeleton";

import PeriodicTable from "@/components/PeriodicTable.vue";
import { useScreenStore } from "@/stores/screen";
import { useCount, useElementStats, useElements } from "@/composables";
import { DG_BANDS, dgBucket, dgRamp } from "@/charts/palette";
import { num } from "@/format";
import { useDark } from "@vueuse/core";
import type { ModelName } from "@/api";

const emit = defineEmits<{ submit: []; reset: [] }>();
const props = withDefaults(defineProps<{ submitting?: boolean; variant?: "hero" | "docked" }>(), {
  variant: "docked",
});

const hero = computed(() => props.variant === "hero");

const store = useScreenStore();
const { form } = storeToRefs(store);

const { data: elements, isLoading: loadingElements } = useElements();
const { data: elementStats } = useElementStats();

const selectedElements = computed(() => form.value.elements);
const excludeTrain = computed(() => form.value.exclude_train);
const { data: count, isFetching: counting } = useCount(selectedElements, excludeTrain);

const nCandidates = computed(() => count.value?.n_candidates ?? null);
const noMatches = computed(() => nCandidates.value === 0);

const submitLabel = computed(() => {
  if (props.submitting) return "Triando…";
  if (form.value.elements.length === 0) return "Triar";
  if (counting.value || nCandidates.value === null) return "Triar";
  if (nCandidates.value === 0) return "Nenhuma candidata";
  const shown = Math.min(form.value.top, nCandidates.value);
  return `Triar ${shown} de ${nCandidates.value}`;
});

// Sem seletor: as métricas de ETR e MACE são indistinguíveis (R² 0,961 vs
// 0,957), então a escolha não mudava o resultado. Roda-se sempre os dois, e a
// divergência entre eles vira o sinal de confiança por linha.
const FIXED_MODEL: ModelName = "ensemble";

// Garante o ensemble mesmo vindo de URL antiga com ?modelo=etr_emb.
if (form.value.model !== FIXED_MODEL) store.setForm({ model: FIXED_MODEL });

const TOP_MIN = 5;
const TOP_MAX = 100;
// Marcas da régua posicionadas pela posição REAL no slider (domínio 5..100),
// e não em intervalos iguais: senão 25/50/75 não caem sob o cursor.
const topTicks = [5, 25, 50, 75, 100].map((value) => ({
  value,
  pos: ((value - TOP_MIN) / (TOP_MAX - TOP_MIN)) * 100,
}));

const canSubmit = computed(
  () => form.value.elements.length > 0 && form.value.top > 0 && !noMatches.value,
);

const isDark = useDark();

/** Leitura dos elementos escolhidos. Sem seleção, mostra os mais próximos do
 *  ótimo de Sabatier, que é por onde se começa a procurar. */
const readout = computed(() => {
  const all = elementStats.value?.elements ?? [];
  const minN = elementStats.value?.min_sample ?? 10;
  const usable = all.filter((e) => e.n >= minN);
  const picked = form.value.elements.length
    ? usable.filter((e) => form.value.elements.includes(e.symbol))
    : [...usable].sort((a, b) => Math.abs(a.median_dG) - Math.abs(b.median_dG)).slice(0, 5);
  return picked.map((e) => ({
    ...e,
    band: DG_BANDS[dgBucket(e.median_dG)]!.label,
    color: dgRamp(e.median_dG, isDark.value),
  }));
});

function toggleElement(el: string) {
  const has = form.value.elements.includes(el);
  store.setForm({
    elements: has ? form.value.elements.filter((e) => e !== el) : [...form.value.elements, el],
  });
}

function removeElement(el: string) {
  store.setForm({ elements: form.value.elements.filter((e) => e !== el) });
}

function onSubmit() {
  if (!canSubmit.value) return;
  emit("submit");
}

useEventListener(window, "keydown", (e: KeyboardEvent) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    onSubmit();
  }
});
</script>

<template>
  <form :class="hero ? 'flex flex-col gap-6' : 'flex flex-col gap-6'" @submit.prevent="onSubmit">
    <header v-if="hero" class="chrome-in flex flex-wrap items-baseline justify-between gap-3">
      <h1 class="text-base font-semibold tracking-tight text-surface-900 dark:text-surface-0">
        Onde procurar catalisadores para a HER
      </h1>
      <p v-if="!form.elements.length" class="text-sm text-surface-500">
        Comece por
        <button
          v-for="(e, i) in readout"
          :key="e.symbol"
          type="button"
          class="font-mono font-semibold text-surface-800 underline underline-offset-2 dark:text-surface-100"
          @click="toggleElement(e.symbol)"
        >
          {{ e.symbol
          }}<span v-if="i < readout.length - 1" class="font-sans font-normal text-surface-500"
            >,
          </span>
        </button>
      </p>
      <p v-else class="font-mono text-sm text-surface-500">
        {{ form.elements.join(", ") }}
      </p>
    </header>

    <section>
      <div v-if="!hero" class="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 class="text-base font-semibold text-surface-900 dark:text-surface-0">Composição</h2>
        <div v-if="form.elements.length" class="flex flex-wrap items-center gap-1.5">
          <button
            v-for="el in form.elements"
            :key="el"
            type="button"
            class="flex items-center gap-1 rounded border border-surface-300 bg-surface-0 py-0.5 pr-1 pl-2 font-mono text-sm text-surface-800 transition hover:border-surface-400 dark:border-surface-600 dark:bg-surface-800 dark:text-surface-100"
            :aria-label="`Remover ${el}`"
            @click="removeElement(el)"
          >
            {{ el }}
            <i class="pi pi-times text-2xs text-surface-400" />
          </button>
          <button
            type="button"
            class="ml-1 text-xs text-surface-500 underline underline-offset-2 hover:text-surface-800 dark:hover:text-surface-200"
            @click="store.setForm({ elements: [] })"
          >
            Limpar
          </button>
        </div>
        <span v-else class="text-xs text-surface-500">nenhum elemento selecionado</span>
      </div>

      <div>
        <Skeleton v-if="loadingElements" height="17rem" class="w-full!" />
        <PeriodicTable
          v-else
          :model-value="form.elements"
          :available="elements ?? []"
          :stats="elementStats?.elements"
          :min-sample="elementStats?.min_sample"
          @update:model-value="store.setForm({ elements: $event })"
        />
      </div>
    </section>

    <!-- Herói: controles numa fileira só, sob o mapa. A tela abre larga e baixa,
         cabendo na viewport, em vez de virar uma coluna de formulário. -->
    <section
      v-if="hero"
      class="chrome-in flex flex-col gap-3 border-t border-surface-200 pt-5 dark:border-surface-800"
    >
      <div class="flex flex-wrap items-end gap-x-8 gap-y-4">
        <div>
          <label class="mb-1.5 block text-xs font-medium text-surface-500">
            Quantas candidatas mostrar
          </label>
          <div class="flex items-center gap-2">
            <InputNumber
              v-model="form.top"
              :min="TOP_MIN"
              :max="TOP_MAX"
              show-buttons
              button-layout="horizontal"
              :step="5"
              :input-style="{ width: '3rem', textAlign: 'center' }"
              @update:model-value="store.setForm({ top: form.top })"
            />
            <span class="font-mono text-xs tabular-nums text-surface-500">
              {{ nCandidates !== null ? `de ${nCandidates}` : "" }}
            </span>
          </div>
        </div>

        <label class="flex max-w-[22rem] cursor-pointer items-start gap-2.5">
          <ToggleSwitch
            v-model="form.exclude_train"
            class="mt-0.5 shrink-0"
            @change="store.setForm({ exclude_train: form.exclude_train })"
          />
          <span class="text-sm leading-snug text-surface-700 dark:text-surface-200">
            Só o conjunto de teste
            <span class="mt-0.5 block text-2xs leading-snug text-surface-500">
              As 1.172 estruturas que o modelo nunca viu. Nas de treino a predição é otimista,
              porque o modelo pode ter memorizado a resposta.
            </span>
          </span>
        </label>

        <Button
          type="submit"
          :label="submitLabel"
          :loading="submitting"
          :disabled="!canSubmit || submitting"
          class="ml-auto min-w-[13rem]"
        />
      </div>

      <p v-if="noMatches" class="text-sm text-surface-600 dark:text-surface-300">
        Nenhuma estrutura contém {{ form.elements.join(", ") }} ao mesmo tempo.
        <button
          v-if="form.elements.length > 1"
          type="button"
          class="underline underline-offset-2"
          @click="removeElement(form.elements[form.elements.length - 1]!)"
        >
          Remover {{ form.elements[form.elements.length - 1] }}
        </button>
        <button
          v-else-if="form.exclude_train"
          type="button"
          class="underline underline-offset-2"
          @click="store.setForm({ exclude_train: false })"
        >
          Incluir o conjunto de treino
        </button>
      </p>
    </section>

    <!-- Acoplado: os mesmos controles empilhados na lateral. -->
    <template v-else>
      <p v-if="noMatches" class="text-sm text-surface-600 dark:text-surface-300">
        Nenhuma estrutura contém {{ form.elements.join(", ") }} ao mesmo tempo.
        <button
          v-if="form.elements.length > 1"
          type="button"
          class="underline underline-offset-2"
          @click="removeElement(form.elements[form.elements.length - 1]!)"
        >
          Remover {{ form.elements[form.elements.length - 1] }}
        </button>
        <button
          v-else-if="form.exclude_train"
          type="button"
          class="underline underline-offset-2"
          @click="store.setForm({ exclude_train: false })"
        >
          Incluir o conjunto de treino
        </button>
      </p>

      <section class="flex flex-col gap-5 border-t border-surface-200 pt-5 dark:border-surface-800">
        <div>
          <div class="mb-2 flex items-baseline justify-between gap-2">
            <label class="text-xs font-medium text-surface-500">Candidatas a mostrar</label>
            <span class="font-mono text-xs tabular-nums text-surface-500">
              {{ nCandidates !== null ? `de ${nCandidates} candidatas` : `máx. ${TOP_MAX}` }}
            </span>
          </div>
          <div class="flex items-center gap-4">
            <!-- A régua fica DENTRO deste bloco flex-1 para ter exatamente a mesma
                 largura do slider; se ficar fora, as marcas não coincidem. -->
            <div class="flex-1">
              <Slider
                v-model="form.top"
                :min="TOP_MIN"
                :max="TOP_MAX"
                :step="1"
                class="w-full"
                @change="store.setForm({ top: form.top })"
              />
              <div class="relative mt-2 h-3 font-mono text-2xs tabular-nums text-surface-400">
                <span
                  v-for="tick in topTicks"
                  :key="tick.value"
                  class="absolute"
                  :class="
                    tick.pos === 0
                      ? 'translate-x-0'
                      : tick.pos === 100
                        ? '-translate-x-full'
                        : '-translate-x-1/2'
                  "
                  :style="{ left: `${tick.pos}%` }"
                  >{{ tick.value }}</span
                >
              </div>
            </div>
            <InputNumber
              v-model="form.top"
              :min="TOP_MIN"
              :max="TOP_MAX"
              show-buttons
              button-layout="horizontal"
              :step="1"
              :input-style="{ width: '2.75rem', textAlign: 'center' }"
              size="small"
              @update:model-value="store.setForm({ top: form.top })"
            />
          </div>
        </div>

        <label class="flex cursor-pointer items-center gap-3">
          <ToggleSwitch
            v-model="form.exclude_train"
            class="shrink-0"
            @change="store.setForm({ exclude_train: form.exclude_train })"
          />
          <span class="text-sm leading-snug text-surface-800 dark:text-surface-100">
            Só o conjunto de teste
            <span class="mt-0.5 block text-2xs leading-snug text-surface-500">
              As 1.172 estruturas que o modelo nunca viu. Nas de treino a predição é otimista,
              porque o modelo pode ter memorizado a resposta.
            </span>
          </span>
        </label>

        <Button
          type="submit"
          :label="submitLabel"
          :loading="submitting"
          :disabled="!canSubmit || submitting"
          class="w-full"
        />

        <div class="flex items-center justify-between">
          <span class="font-mono text-2xs text-surface-400">Ctrl + Enter</span>
          <button
            type="button"
            class="text-xs text-surface-500 underline underline-offset-2 hover:text-surface-800 dark:hover:text-surface-200"
            @click="emit('reset')"
          >
            Restaurar padrões
          </button>
        </div>
      </section>

      <section
        v-if="readout.length"
        class="border-t border-surface-200 pt-5 dark:border-surface-800"
      >
        <h3 class="mb-2.5 text-xs font-medium text-surface-500">
          {{ form.elements.length ? "Elementos selecionados" : "Mais perto do ótimo" }}
        </h3>
        <ul class="divide-y divide-surface-200 dark:divide-surface-800">
          <li v-for="e in readout" :key="e.symbol" class="flex items-center gap-3 py-2 text-sm">
            <span
              class="h-2.5 w-2.5 shrink-0 rounded-full"
              :style="{ backgroundColor: e.color }"
            ></span>
            <button
              type="button"
              class="w-9 shrink-0 text-left font-mono font-semibold text-surface-900 hover:underline dark:text-surface-0"
              @click="toggleElement(e.symbol)"
            >
              {{ e.symbol }}
            </button>
            <span class="flex-1 truncate text-xs text-surface-500">{{ e.band }}</span>
            <span class="shrink-0 font-mono text-xs tabular-nums text-surface-500"
              >{{ e.n }} est.</span
            >
            <span
              class="w-16 shrink-0 text-right font-mono tabular-nums text-surface-800 dark:text-surface-100"
            >
              {{ num(e.median_dG, 3) }}
            </span>
          </li>
        </ul>
      </section>
    </template>
  </form>
</template>

<style scoped>
/* Só o cromo que é substituído faz fade. O mapa nunca some: é ele que viaja. */
.chrome-in {
  animation: chrome-in 380ms cubic-bezier(0.22, 1, 0.36, 1) 200ms both;
}
@keyframes chrome-in {
  from {
    opacity: 0;
    transform: translateY(6px);
  }
  to {
    opacity: 1;
    transform: none;
  }
}
@media (prefers-reduced-motion: reduce) {
  .chrome-in {
    animation: none;
  }
}
</style>

<style>
/* A trilha do slider tem poucos pixels de altura, então só acerta quem clica
   exatamente em cima. O pseudo-elemento amplia a área de acerto sem engordar a
   linha: ele conta como acerto no próprio .p-slider, e o handle, por ser filho,
   continua pintando acima. */
.p-slider.p-slider-horizontal {
  position: relative;
}
.p-slider.p-slider-horizontal::before {
  content: "";
  position: absolute;
  inset-block: -14px;
  inset-inline: 0;
}

.model-picker .p-selectbutton {
  display: flex;
  width: 100%;
  gap: 2px;
  padding: 2px;
  border: 1px solid var(--p-surface-200);
  border-radius: 0.375rem;
  background: var(--p-surface-50);
}
.dark .model-picker .p-selectbutton {
  border-color: var(--p-surface-700);
  background: var(--p-surface-900);
}
.model-picker .p-selectbutton .p-togglebutton {
  flex: 1 1 0;
  border: 0 !important;
  border-radius: 0.25rem !important;
  background: transparent !important;
  padding-block: 0.4rem;
  font-size: 0.8125rem;
}
.model-picker .p-selectbutton .p-togglebutton::before {
  display: none;
}
.model-picker .p-selectbutton .p-togglebutton.p-togglebutton-checked {
  background: var(--p-surface-0) !important;
  color: var(--p-primary-color) !important;
  font-weight: 600;
}
.dark .model-picker .p-selectbutton .p-togglebutton.p-togglebutton-checked {
  background: var(--p-surface-800) !important;
}
</style>
