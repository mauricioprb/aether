<script setup lang="ts">
import { computed } from 'vue'
import { storeToRefs } from 'pinia'
import { useToast } from 'primevue/usetoast'
import Message from 'primevue/message'
import Skeleton from 'primevue/skeleton'

import ScreenForm from '@/components/ScreenForm.vue'
import ResultsTable from '@/components/ResultsTable.vue'
import EmptyState from '@/components/EmptyState.vue'
import { useScreen, useStats } from '@/composables'
import { useScreenStore } from '@/stores/screen'

const toast = useToast()
const store = useScreenStore()
const { form, lastResult, hasResult } = storeToRefs(store)

const screenMutation = useScreen()
const stats = useStats()

const submitting = computed(() => screenMutation.isPending.value)
const errorMsg = computed(() => screenMutation.error.value?.message)

function onSubmit() {
  screenMutation.mutate({ ...form.value }, {
    onSuccess: (data) => {
      toast.add({
        severity: 'success',
        summary: `${data.n_candidates} candidatos`,
        detail: data.n_candidates === 0
          ? 'Nenhum encontrado com esses elementos'
          : `Mostrando top ${data.rows.length} ordenados por |ΔG_H|`,
        life: 3500,
      })
    },
    onError: (err) => {
      toast.add({
        severity: 'error',
        summary: 'Falha na triagem',
        detail: err.message,
        life: 6000,
      })
    },
  })
}

function onReset() {
  store.resetForm()
  toast.add({ severity: 'info', summary: 'Padrões restaurados', life: 2000 })
}
</script>

<template>
  <section class="mx-auto max-w-7xl px-6 py-8">
    <header class="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 class="text-2xl font-semibold tracking-tight">Triagem de catalisadores</h1>
        <p class="mt-1 text-sm text-surface-500">
          Filtre por composição e ranqueie candidatos por |ΔG_H_pred|.
        </p>
      </div>
      <div class="flex gap-3 text-xs text-surface-500">
        <div v-if="stats.data.value">
          <span class="font-semibold text-surface-700 dark:text-surface-200">{{ stats.data.value.n_structures }}</span> estruturas
        </div>
        <div v-if="stats.data.value">
          <span class="font-semibold text-surface-700 dark:text-surface-200">{{ stats.data.value.n_test_canonical }}</span> teste
        </div>
        <div v-if="stats.data.value">
          <span class="font-semibold text-surface-700 dark:text-surface-200">{{ stats.data.value.available_elements.length }}</span> metais
        </div>
        <Skeleton v-else width="12rem" height="1rem" />
      </div>
    </header>

    <ScreenForm :submitting="submitting" @submit="onSubmit" @reset="onReset" />

    <Message v-if="errorMsg" severity="error" class="mt-4" :closable="false">
      {{ errorMsg }}
    </Message>

    <section class="mt-6">
      <div v-if="submitting" class="space-y-2">
        <Skeleton height="3rem" />
        <Skeleton v-for="i in 6" :key="i" height="2.25rem" />
      </div>

      <ResultsTable v-else-if="hasResult && lastResult" :result="lastResult" />

      <div v-else class="rounded-xl border border-dashed border-surface-300 dark:border-surface-700">
        <EmptyState
          icon="pi-search"
          title="Nenhuma triagem ainda"
          description="Escolha os metais, ajuste top N e clique em Triar candidatos."
        />
      </div>
    </section>
  </section>
</template>
