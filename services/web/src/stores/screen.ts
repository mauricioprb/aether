import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import type { CandidateRow, ModelName, ScreenRequest, ScreenResponse } from '@/api'

const DEFAULTS: ScreenRequest = {
  elements: ['Pt'],
  top: 10,
  model: 'etr_emb',
  exclude_train: true,
}

const STORAGE_KEY = 'aether.screen.form.v1'

function loadPersistedForm(): ScreenRequest {
  if (typeof window === 'undefined') return { ...DEFAULTS }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return { ...DEFAULTS }
    const parsed = JSON.parse(raw) as Partial<ScreenRequest>
    return {
      elements: parsed.elements ?? DEFAULTS.elements,
      top: parsed.top ?? DEFAULTS.top,
      model: (parsed.model as ModelName | undefined) ?? DEFAULTS.model,
      exclude_train: parsed.exclude_train ?? DEFAULTS.exclude_train,
    }
  } catch {
    return { ...DEFAULTS }
  }
}

export const useScreenStore = defineStore('screen', () => {
  const form = ref<ScreenRequest>(loadPersistedForm())
  const lastQuery = ref<ScreenRequest | null>(null)
  const lastResult = ref<ScreenResponse | null>(null)

  function persist() {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(form.value))
  }

  function setForm(partial: Partial<ScreenRequest>) {
    form.value = { ...form.value, ...partial }
    persist()
  }

  function resetForm() {
    form.value = { ...DEFAULTS }
    persist()
  }

  function setLastQuery(q: ScreenRequest) {
    lastQuery.value = { ...q }
  }

  function setLastResult(r: ScreenResponse) {
    lastResult.value = r
  }

  const rows = computed<CandidateRow[]>(() => lastResult.value?.rows ?? [])
  const hasResult = computed(() => lastResult.value !== null)

  return {
    form,
    lastQuery,
    lastResult,
    rows,
    hasResult,
    setForm,
    resetForm,
    setLastQuery,
    setLastResult,
  }
})
