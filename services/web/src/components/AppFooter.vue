<script setup lang="ts">
import BrandMark from "@/components/BrandMark.vue";
import BrandWordmark from "@/components/BrandWordmark.vue";
import LogoLattes from "@/components/LogoLattes.vue";

// Perfil, nao o repositorio: o codigo esta sob registro no INPI e continua privado.
const links = [
  { href: "https://github.com/mauricioprb", icon: "pi-github", label: "GitHub" },
  { href: "http://lattes.cnpq.br/4826637586256637", icon: "lattes", label: "Currículo Lattes" },
];
</script>

<template>
  <!-- Fecha o conteúdo rolável, não a viewport: na casca do app ele ficaria
       preso na tela e nunca sairia do caminho. -->
  <footer
    class="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-surface-200 px-1 pt-4 text-2xs text-surface-500 dark:border-surface-800"
  >
    <span class="flex items-center gap-2 text-surface-700 dark:text-surface-300">
      <BrandMark :size="16" class="text-primary-600 dark:text-primary-400" />
      <BrandWordmark :height="9" />
    </span>

    <span class="ml-auto flex items-center gap-1">
      <a
        v-for="l in links"
        :key="l.href"
        :href="l.href"
        target="_blank"
        rel="noopener"
        :aria-label="l.label"
        :title="l.label"
        class="grid h-7 w-7 place-items-center rounded-sm text-surface-500 hover:bg-surface-100 hover:text-surface-900 dark:hover:bg-surface-800 dark:hover:text-surface-0"
      >
        <LogoLattes v-if="l.icon === 'lattes'" class="h-3.5 w-3.5" />
        <i v-else :class="['pi text-sm', l.icon]" aria-hidden="true" />
      </a>
    </span>
  </footer>
</template>
