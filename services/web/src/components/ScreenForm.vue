<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { useEventListener } from "@vueuse/core";
import SelectButton from "primevue/selectbutton";
import Slider from "primevue/slider";
import InputNumber from "primevue/inputnumber";
import ToggleSwitch from "primevue/toggleswitch";
import Button from "primevue/button";
import Skeleton from "primevue/skeleton";

import PeriodicTable from "@/components/PeriodicTable.vue";
import { useScreenStore } from "@/stores/screen";
import { useElements } from "@/composables";
import type { ModelName } from "@/api";

const emit = defineEmits<{ submit: []; reset: [] }>();
defineProps<{ submitting?: boolean }>();

const store = useScreenStore();
const { form } = storeToRefs(store);

const { data: elements, isLoading: loadingElements } = useElements();

const modelOptions: { label: string; value: ModelName; r2: string; hint: string }[] = [
  {
    label: "ETR + emb.",
    value: "etr_emb",
    r2: "R² = 0,96",
    hint: "ETR sobre os embeddings do MACE-MP-0",
  },
  {
    label: "MACE",
    value: "stagea",
    r2: "R² = 0,96",
    hint: "cabeça de regressão sobre o MACE-MP-0 congelado",
  },
  {
    label: "Combinado",
    value: "ensemble",
    r2: "média",
    hint: "média das predições dos dois modelos",
  },
];

const TOP_MIN = 5;
const TOP_MAX = 100;
// Marcas da régua posicionadas pela posição REAL no slider (domínio 5..100),
// e não em intervalos iguais: senão 25/50/75 não caem sob o cursor.
const topTicks = [5, 25, 50, 75, 100].map((value) => ({
  value,
  pos: ((value - TOP_MIN) / (TOP_MAX - TOP_MIN)) * 100,
}));

const canSubmit = computed(() => form.value.elements.length > 0 && form.value.top > 0);
const currentModelHint = computed(() => modelOptions.find((m) => m.value === form.value.model));

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
  <form
    class="rounded-2xl border border-surface-200 bg-surface-0 shadow-sm dark:border-surface-800 dark:bg-surface-950"
    @submit.prevent="onSubmit"
  >
    <div class="border-b border-surface-200 p-5 dark:border-surface-800">
      <div class="mb-3 flex items-center justify-between gap-2">
        <div class="flex items-center gap-2">
          <span
            class="grid h-6 w-6 place-items-center rounded-md bg-primary-500/10 text-primary-600 dark:text-primary-400"
          >
            <i class="pi pi-tag text-2xs" />
          </span>
          <h3
            class="text-xs font-semibold uppercase tracking-wider text-surface-700 dark:text-surface-200"
          >
            Elementos obrigatórios
          </h3>
        </div>
        <span v-if="form.elements.length" class="text-xs text-surface-500">
          {{ form.elements.join(", ") }}
        </span>
      </div>
      <Skeleton v-if="loadingElements" height="14rem" class="w-full!" />
      <PeriodicTable
        v-else
        :model-value="form.elements"
        :available="elements ?? []"
        @update:model-value="store.setForm({ elements: $event })"
      />
      <p class="mt-2 text-xs text-surface-500">
        A estrutura deve conter <em>todos</em> os elementos selecionados.
      </p>
    </div>

    <div class="border-b border-surface-200 p-5 dark:border-surface-800">
      <div class="mb-3 flex items-center gap-2">
        <span
          class="grid h-6 w-6 place-items-center rounded-md bg-primary-500/10 text-primary-600 dark:text-primary-400"
        >
          <i class="pi pi-microchip-ai text-2xs" />
        </span>
        <h3
          class="text-xs font-semibold uppercase tracking-wider text-surface-700 dark:text-surface-200"
        >
          Modelo preditivo
        </h3>
      </div>
      <div class="model-picker">
        <SelectButton
          v-model="form.model"
          :options="modelOptions"
          option-label="label"
          option-value="value"
          :allow-empty="false"
          @change="store.setForm({ model: form.model })"
        />
      </div>
      <div class="mt-2.5 flex items-center gap-2 text-xs">
        <span class="text-surface-500">{{ currentModelHint?.hint }}</span>
        <span
          class="ml-auto shrink-0 rounded-full bg-primary-500/10 px-2 py-0.5 text-2xs font-semibold text-primary-700 dark:text-primary-400"
        >
          {{ currentModelHint?.r2 }}
        </span>
      </div>
    </div>

    <div class="grid gap-0 md:grid-cols-2">
      <div
        class="border-b border-surface-200 p-5 dark:border-surface-800 md:border-b-0 md:border-r"
      >
        <div class="mb-3 flex items-center gap-2">
          <span
            class="grid h-6 w-6 place-items-center rounded-md bg-primary-500/10 text-primary-600 dark:text-primary-400"
          >
            <i class="pi pi-sort-amount-up text-2xs" />
          </span>
          <h3
            class="text-xs font-semibold uppercase tracking-wider text-surface-700 dark:text-surface-200"
          >
            Número de candidatos
          </h3>
        </div>
        <div class="flex items-center gap-4">
          <!-- A régua fica DENTRO deste bloco flex-1 para ter exatamente a
               mesma largura do slider; se ficar fora, ela se estende sob o
               campo numérico e as marcas deixam de coincidir com o cursor. -->
          <div class="flex-1">
            <Slider
              v-model="form.top"
              :min="TOP_MIN"
              :max="TOP_MAX"
              :step="1"
              class="w-full"
              @change="store.setForm({ top: form.top })"
            />
            <div class="relative mt-1.5 h-3 text-2xs tabular-nums text-surface-400">
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
            :input-style="{ width: '3.5rem', textAlign: 'center' }"
            @update:model-value="store.setForm({ top: form.top })"
          />
        </div>
      </div>

      <div class="p-5">
        <div class="mb-3 flex items-center gap-2">
          <span
            class="grid h-6 w-6 place-items-center rounded-md bg-primary-500/10 text-primary-600 dark:text-primary-400"
          >
            <i class="pi pi-filter text-2xs" />
          </span>
          <h3
            class="text-xs font-semibold uppercase tracking-wider text-surface-700 dark:text-surface-200"
          >
            Opções
          </h3>
        </div>
        <label class="flex cursor-pointer items-start gap-3 rounded-lg">
          <ToggleSwitch
            v-model="form.exclude_train"
            class="shrink-0"
            @change="store.setForm({ exclude_train: form.exclude_train })"
          />
          <span class="text-xs leading-snug">
            <span class="block font-medium text-surface-700 dark:text-surface-200"
              >Restringir ao conjunto de teste</span
            >
            <span class="mt-0.5 block text-2xs text-surface-500">
              Exclui as estruturas vistas no treinamento, evitando predições memorizadas.
            </span>
          </span>
        </label>
      </div>
    </div>

    <footer
      class="flex items-center justify-between gap-3 rounded-b-2xl border-t border-surface-200 bg-surface-50 px-4 py-3 sm:px-5 dark:border-surface-800 dark:bg-surface-900/60"
    >
      <div class="hidden items-center gap-3 text-2xs text-surface-500 sm:flex">
        <span>
          Atalho:
          <kbd
            class="rounded border border-surface-300 bg-surface-0 px-1.5 py-0.5 text-2xs font-mono dark:border-surface-700 dark:bg-surface-950"
            >Ctrl</kbd
          >
          +
          <kbd
            class="rounded border border-surface-300 bg-surface-0 px-1.5 py-0.5 text-2xs font-mono dark:border-surface-700 dark:bg-surface-950"
            >Enter</kbd
          >
        </span>
      </div>
      <div class="flex w-full flex-col-reverse gap-2 sm:w-auto sm:flex-row sm:items-center">
        <Button
          type="button"
          label="Restaurar"
          icon="pi pi-refresh"
          severity="secondary"
          text
          size="small"
          class="w-full sm:w-auto"
          @click="emit('reset')"
        />
        <Button
          type="submit"
          :label="submitting ? 'Triando…' : 'Triar candidatos'"
          icon="pi pi-search"
          icon-pos="right"
          :loading="submitting"
          :disabled="!canSubmit || submitting"
          class="w-full sm:w-auto"
        />
      </div>
    </footer>
  </form>
</template>

<style>
.model-picker .p-selectbutton {
  display: flex;
  width: 100%;
  gap: 4px;
  padding: 4px;
  border: 0;
  border-radius: 0.6rem;
  background: var(--p-surface-100);
}
.dark .model-picker .p-selectbutton {
  background: var(--p-surface-800);
}
.model-picker .p-selectbutton .p-togglebutton {
  flex: 1 1 0;
  border: 0 !important;
  border-radius: 0.45rem !important;
  background: transparent !important;
}
.model-picker .p-selectbutton .p-togglebutton::before {
  display: none;
}
.model-picker .p-selectbutton .p-togglebutton.p-togglebutton-checked {
  background: color-mix(in oklab, var(--p-primary-color) 16%, transparent) !important;
  color: var(--p-primary-color) !important;
  font-weight: 600;
}
</style>
