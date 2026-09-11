<script setup lang="ts">
import { useDark, useToggle } from "@vueuse/core";
import { showShortcutsHelp } from "@/composables";
import BrandMark from "@/components/BrandMark.vue";
import BrandWordmark from "@/components/BrandWordmark.vue";

const isDark = useDark({ selector: "html", attribute: "class", valueDark: "dark", valueLight: "" });
const toggleDark = useToggle(isDark);

const links = [
  { to: "/screen", label: "Triagem" },
  { to: "/predict", label: "Prever" },
  { to: "/compare", label: "Modelos" },
];
</script>

<template>
  <header
    class="flex h-14 shrink-0 items-center gap-3 border-b border-surface-200 bg-surface-0 px-3 sm:gap-8 sm:px-5 dark:border-surface-800 dark:bg-surface-950"
  >
    <RouterLink
      to="/screen"
      aria-label="AETHER — Triagem"
      class="flex h-10 shrink-0 items-center gap-2 rounded-sm text-surface-900 dark:text-surface-0"
    >
      <BrandMark :size="32" class="text-primary-600 dark:text-primary-400" />
      <BrandWordmark :height="15" class="hidden sm:block" />
    </RouterLink>

    <nav aria-label="Navegação principal" class="flex h-full items-stretch gap-1 sm:gap-2">
      <RouterLink
        v-for="r in links"
        :key="r.to"
        :to="r.to"
        class="nav-link inline-flex items-center border-b-2 border-transparent px-2 pt-0.5 text-sm font-medium whitespace-nowrap text-surface-600 hover:text-surface-900 focus-visible:-outline-offset-4 sm:px-3 dark:text-surface-400 dark:hover:text-surface-0"
        active-class=""
      >
        {{ r.label }}
      </RouterLink>
    </nav>

    <div class="ml-auto flex shrink-0 items-center">
      <button
        type="button"
        class="grid h-10 w-9 place-items-center rounded-sm text-surface-600 hover:bg-surface-100 hover:text-surface-900 sm:w-10 dark:text-surface-400 dark:hover:bg-surface-800 dark:hover:text-surface-0"
        aria-label="Atalhos de teclado"
        title="Atalhos de teclado"
        @click="showShortcutsHelp = true"
      >
        <i class="pi pi-question-circle text-sm" aria-hidden="true" />
      </button>
      <button
        type="button"
        class="grid h-10 w-9 place-items-center rounded-sm text-surface-600 hover:bg-surface-100 hover:text-surface-900 sm:w-10 dark:text-surface-400 dark:hover:bg-surface-800 dark:hover:text-surface-0"
        :aria-label="isDark ? 'Ativar tema claro' : 'Ativar tema escuro'"
        :title="isDark ? 'Tema claro' : 'Tema escuro'"
        @click="toggleDark()"
      >
        <i :class="['pi text-sm', isDark ? 'pi-sun' : 'pi-moon']" aria-hidden="true" />
      </button>
    </div>
  </header>
</template>

<style scoped>
.nav-link[aria-current="page"] {
  border-bottom-color: var(--p-primary-color);
  color: var(--p-text-color);
}
</style>
