<script setup lang="ts">
import { computed, ref } from "vue";
import { useDark } from "@vueuse/core";
import Button from "primevue/button";
import Message from "primevue/message";
import SelectButton from "primevue/selectbutton";

import PageHeader from "@/components/PageHeader.vue";
import SectionCard from "@/components/SectionCard.vue";
import AppFooter from "@/components/AppFooter.vue";
import { usePredict } from "@/composables";
import { formatOf, type ModelName } from "@/api";
import { DG_BANDS, dgBucket, dgRamp } from "@/charts/palette";
import { num } from "@/format";

const isDark = useDark();
const predict = usePredict();

const file = ref<File | null>(null);
const dragging = ref(false);
const model = ref<ModelName>("ensemble");
const localError = ref<string | null>(null);

const MODELS: { label: string; value: ModelName }[] = [
  { label: "Ensemble", value: "ensemble" },
  { label: "ETR", value: "etr_emb" },
  { label: "MACE", value: "stagea" },
];

const result = computed(() => predict.data.value ?? null);
// O 400 do backend carrega a razao da recusa (atomos sobrepostos, sem H, formato
// ilegivel). E a parte util da resposta: mostrar "erro" perderia o motivo.
const errorMsg = computed(() => localError.value ?? predict.error.value?.message ?? null);

function take(f: File | undefined) {
  if (!f) return;
  predict.reset();
  if (!formatOf(f.name)) {
    localError.value = `"${f.name}" não é um formato aceito. Use .cif, .xyz/.extxyz ou POSCAR.`;
    file.value = null;
    return;
  }
  localError.value = null;
  file.value = f;
}

function onDrop(e: DragEvent) {
  dragging.value = false;
  take(e.dataTransfer?.files?.[0]);
}

function onPick(e: Event) {
  take((e.target as HTMLInputElement).files?.[0]);
}

function submit() {
  if (file.value) predict.mutate({ file: file.value, model: model.value });
}

function clear() {
  file.value = null;
  localError.value = null;
  predict.reset();
}
</script>

<template>
  <div class="mx-auto flex h-full w-full max-w-3xl flex-col gap-6 px-4 py-6">
    <PageHeader title="Prever ΔG_H de uma estrutura">
      <p class="text-sm text-surface-600 dark:text-surface-300">
        Envie a estrutura com o H adsorvido. A predição sai dos mesmos pesos da triagem, mas aqui
        ela não vem do conjunto: é calculada para a geometria que você enviou.
      </p>
    </PageHeader>

    <SectionCard>
      <label
        class="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-6 py-10 text-center transition"
        :class="
          dragging
            ? 'border-primary-500 bg-primary-50/60 dark:bg-primary-950/30'
            : 'border-surface-300 hover:border-surface-400 dark:border-surface-700 dark:hover:border-surface-600'
        "
        @dragover.prevent="dragging = true"
        @dragleave.prevent="dragging = false"
        @drop.prevent="onDrop"
      >
        <input
          type="file"
          class="sr-only"
          accept=".cif,.xyz,.extxyz,.vasp,POSCAR,CONTCAR"
          @change="onPick"
        />
        <i class="pi pi-upload text-lg text-surface-400" aria-hidden="true" />
        <span class="text-sm font-medium text-surface-800 dark:text-surface-100">
          {{ file ? file.name : "Arraste um arquivo ou clique para escolher" }}
        </span>
        <span class="text-2xs text-surface-500">CIF, extended XYZ ou POSCAR, até 1 MB</span>
      </label>

      <div class="mt-4 flex flex-wrap items-center gap-3">
        <SelectButton
          v-model="model"
          :options="MODELS"
          option-label="label"
          option-value="value"
          :allow-empty="false"
          size="small"
          aria-label="Modelo"
        />
        <Button
          v-if="file || result"
          label="Limpar"
          severity="secondary"
          text
          size="small"
          @click="clear"
        />
        <Button
          label="Prever"
          class="ml-auto min-w-[9rem]"
          :loading="predict.isPending.value"
          :disabled="!file || predict.isPending.value"
          @click="submit"
        />
      </div>
    </SectionCard>

    <Message v-if="errorMsg" severity="error" :closable="false">{{ errorMsg }}</Message>

    <SectionCard v-if="result" :title="result.chemical_formula">
      <div class="flex flex-wrap items-end gap-x-10 gap-y-4">
        <div>
          <div class="text-2xs tracking-wide text-surface-500 uppercase">ΔG_H previsto</div>
          <div
            class="font-mono text-3xl font-semibold tabular-nums"
            :style="{ color: dgRamp(result.dG_pred, isDark) }"
          >
            {{ result.dG_pred >= 0 ? "+" : "−" }}{{ num(Math.abs(result.dG_pred)) }}
            <span class="text-base font-normal text-surface-500">eV</span>
          </div>
          <div class="mt-1 text-xs text-surface-600 dark:text-surface-300">
            {{ DG_BANDS[dgBucket(result.dG_pred)]?.label }}, a
            <span class="font-mono tabular-nums">{{ num(result.abs_dG_pred) }}</span> eV do ótimo de
            Sabatier
          </div>
        </div>

        <dl class="flex flex-wrap gap-x-8 gap-y-2 text-xs">
          <div>
            <dt class="text-surface-500">Átomos</dt>
            <dd class="font-mono tabular-nums text-surface-900 dark:text-surface-0">
              {{ result.n_atoms }}
            </dd>
          </div>
          <div>
            <dt class="text-surface-500">ΔE_H</dt>
            <dd class="font-mono tabular-nums text-surface-900 dark:text-surface-0">
              {{ num(result.dE_pred) }} eV
            </dd>
          </div>
          <div>
            <dt class="text-surface-500">Modelo</dt>
            <dd class="text-surface-900 dark:text-surface-0">{{ result.model }}</dd>
          </div>
        </dl>
      </div>

      <!-- Fora do treino a barra de erro publicada nao vale, e o numero acima
           continua saindo com a mesma cara de resposta. O aviso e o que separa
           extrapolar de extrapolar calado. -->
      <Message v-if="!result.elements_in_training" severity="warn" :closable="false" class="mt-4">
        Algum elemento desta estrutura não aparece no conjunto de treino. A predição é extrapolação
        e não herda o erro de 70 meV medido no teste.
      </Message>
    </SectionCard>

    <AppFooter class="mt-auto" />
  </div>
</template>
