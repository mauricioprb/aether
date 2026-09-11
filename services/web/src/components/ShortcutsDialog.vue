<script setup lang="ts">
import Dialog from "primevue/dialog";
import { showShortcutsHelp } from "@/composables";

const groups: { label: string; items: { keys: string[]; desc: string }[] }[] = [
  {
    label: "Navegação",
    items: [
      { keys: ["g", "h"], desc: "Ir para Início" },
      { keys: ["g", "s"], desc: "Ir para Buscar" },
      { keys: ["g", "c"], desc: "Ir para Comparar" },
    ],
  },
  {
    label: "Busca",
    items: [{ keys: ["Ctrl", "Enter"], desc: "Executar a triagem" }],
  },
  {
    label: "Geral",
    items: [
      { keys: ["?"], desc: "Abrir esta ajuda" },
      { keys: ["Esc"], desc: "Fechar diálogos" },
    ],
  },
];
</script>

<template>
  <Dialog
    v-model:visible="showShortcutsHelp"
    modal
    header="Atalhos de teclado"
    :style="{ width: '32rem' }"
    :draggable="false"
    dismissable-mask
  >
    <div class="space-y-5">
      <section v-for="g in groups" :key="g.label">
        <h4 class="mb-2 text-xs font-semibold text-surface-500">
          {{ g.label }}
        </h4>
        <ul class="space-y-1.5">
          <li
            v-for="item in g.items"
            :key="item.desc"
            class="flex items-center justify-between text-sm"
          >
            <span class="text-surface-700 dark:text-surface-200">{{ item.desc }}</span>
            <span class="flex items-center gap-1">
              <template v-for="(k, i) in item.keys" :key="k">
                <kbd
                  class="min-w-6 rounded border border-surface-300 bg-surface-50 px-1.5 py-0.5 text-center text-2xs font-mono text-surface-700 shadow-sm dark:border-surface-700 dark:bg-surface-900 dark:text-surface-200"
                  >{{ k }}</kbd
                >
                <span v-if="i < item.keys.length - 1" class="text-2xs text-surface-400">
                  então
                </span>
              </template>
            </span>
          </li>
        </ul>
      </section>
    </div>
  </Dialog>
</template>
