<script setup lang="ts">
import { computed } from 'vue'
import { storeToRefs } from 'pinia'
import MultiSelect from 'primevue/multiselect'
import SelectButton from 'primevue/selectbutton'
import Slider from 'primevue/slider'
import InputNumber from 'primevue/inputnumber'
import ToggleSwitch from 'primevue/toggleswitch'
import Button from 'primevue/button'
import Skeleton from 'primevue/skeleton'

import { useScreenStore } from '@/stores/screen'
import { useElements } from '@/composables'
import type { ModelName } from '@/api'

const emit = defineEmits<{ submit: []; reset: [] }>()
defineProps<{ submitting?: boolean }>()

const store = useScreenStore()
const { form } = storeToRefs(store)

const { data: elements, isLoading: loadingElements } = useElements()

const modelOptions: { label: string; value: ModelName; hint: string }[] = [
  { label: 'ETR + emb',   value: 'etr_emb',  hint: 'rápido · CPU · R²=0.961' },
  { label: 'MACE Stage A', value: 'stagea',   hint: 'GNN · R²=0.956 ± 0.002' },
  { label: 'Ensemble',     value: 'ensemble', hint: 'média dos dois' },
]

const canSubmit = computed(() => form.value.elements.length > 0 && form.value.top > 0)

function onSubmit() {
  if (!canSubmit.value) return
  emit('submit')
}

function onKeydown(e: KeyboardEvent) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') onSubmit()
}
</script>

<template>
  <form
    class="rounded-xl border border-surface-200 bg-surface-0 p-5 shadow-sm dark:border-surface-800 dark:bg-surface-950"
    @submit.prevent="onSubmit"
    @keydown="onKeydown"
  >
    <div class="grid gap-5 md:grid-cols-2 lg:grid-cols-[1.4fr_1fr_auto]">
      <div>
        <label class="mb-1.5 block text-xs font-medium uppercase tracking-wide text-surface-500">
          Metais obrigatórios
        </label>
        <Skeleton v-if="loadingElements" height="2.5rem" class="!w-full" />
        <MultiSelect
          v-else
          v-model="form.elements"
          :options="elements ?? []"
          display="chip"
          filter
          :placeholder="form.elements.length ? '' : 'Selecione elementos (ex: Pt, Ni)'"
          class="w-full"
          :max-selected-labels="10"
          @change="store.setForm({ elements: form.elements })"
        />
        <p class="mt-1.5 text-xs text-surface-500">
          Estruturas devem conter <em>todos</em> os metais escolhidos (H ignorado).
        </p>
      </div>

      <div>
        <label class="mb-1.5 block text-xs font-medium uppercase tracking-wide text-surface-500">
          Modelo
        </label>
        <SelectButton
          v-model="form.model"
          :options="modelOptions"
          option-label="label"
          option-value="value"
          :allow-empty="false"
          class="w-full"
          @change="store.setForm({ model: form.model })"
        />
        <p class="mt-1.5 text-xs text-surface-500">
          {{ modelOptions.find((m) => m.value === form.model)?.hint }}
        </p>
      </div>

      <div class="flex flex-col items-stretch justify-end gap-2">
        <Button
          type="submit"
          :label="submitting ? 'Processando…' : 'Triar candidatos'"
          icon="pi pi-search"
          icon-pos="right"
          :loading="submitting"
          :disabled="!canSubmit || submitting"
          class="w-full"
        />
        <button
          type="button"
          class="text-xs text-surface-500 hover:text-surface-700 dark:hover:text-surface-200"
          @click="emit('reset')"
        >
          Restaurar padrões
        </button>
      </div>
    </div>

    <div class="mt-5 grid gap-5 md:grid-cols-[1fr_auto]">
      <div>
        <div class="mb-1.5 flex items-baseline justify-between">
          <label class="text-xs font-medium uppercase tracking-wide text-surface-500"
            >Top N candidatos</label
          >
          <InputNumber
            v-model="form.top"
            :min="1"
            :max="500"
            show-buttons
            button-layout="horizontal"
            :step="1"
            :input-style="{ width: '4rem', textAlign: 'center' }"
            @update:model-value="store.setForm({ top: form.top })"
          />
        </div>
        <Slider
          v-model="form.top"
          :min="5"
          :max="100"
          :step="1"
          @change="store.setForm({ top: form.top })"
        />
        <div class="mt-1 flex justify-between text-[10px] text-surface-400">
          <span>5</span><span>25</span><span>50</span><span>75</span><span>100</span>
        </div>
      </div>

      <div class="flex items-end justify-between gap-3 md:flex-col md:items-start md:justify-end">
        <div>
          <div class="text-xs font-medium uppercase tracking-wide text-surface-500">
            Excluir treino
          </div>
          <p class="mt-0.5 max-w-[14rem] text-[11px] text-surface-500">
            Restringe ao test canônico (1172 IDs). Evita predições memorizadas pelo ETR.
          </p>
        </div>
        <ToggleSwitch
          v-model="form.exclude_train"
          @change="store.setForm({ exclude_train: form.exclude_train })"
        />
      </div>
    </div>

    <div class="mt-4 text-right text-[11px] text-surface-400">
      Atalho: <kbd class="rounded border border-surface-300 px-1 py-0.5 text-[10px] dark:border-surface-700">Ctrl</kbd>
      +
      <kbd class="rounded border border-surface-300 px-1 py-0.5 text-[10px] dark:border-surface-700">Enter</kbd>
    </div>
  </form>
</template>
