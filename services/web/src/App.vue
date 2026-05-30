<script setup lang="ts">
import { useDark, useToggle } from '@vueuse/core'
import Toast from 'primevue/toast'
import ConfirmDialog from 'primevue/confirmdialog'

const isDark = useDark({
  selector: 'html',
  attribute: 'class',
  valueDark: 'dark',
  valueLight: '',
})
const toggleDark = useToggle(isDark)
</script>

<template>
  <div class="flex h-full flex-col">
    <header
      class="flex items-center justify-between border-b border-surface-200 dark:border-surface-800 bg-surface-0 dark:bg-surface-950 px-6 py-3"
    >
      <RouterLink to="/" class="flex items-center gap-3">
        <span class="grid h-9 w-9 place-items-center rounded-lg bg-primary-500 text-white font-bold">A</span>
        <div class="leading-tight">
          <div class="text-sm font-semibold tracking-wide">AETHER</div>
          <div class="text-xs text-surface-500">HER catalyst screening</div>
        </div>
      </RouterLink>

      <nav class="flex items-center gap-1">
        <RouterLink
          v-for="r in [
            { to: '/screen', label: 'Triagem', icon: 'pi-search' },
            { to: '/compare', label: 'Comparar', icon: 'pi-chart-bar' },
            { to: '/about', label: 'Sobre', icon: 'pi-info-circle' },
          ]"
          :key="r.to"
          :to="r.to"
          class="flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium text-surface-600 hover:bg-surface-100 hover:text-surface-900 dark:text-surface-300 dark:hover:bg-surface-800 dark:hover:text-surface-0"
          active-class="!bg-primary-50 !text-primary-700 dark:!bg-primary-950 dark:!text-primary-300"
        >
          <i :class="['pi', r.icon, 'text-xs']" />
          {{ r.label }}
        </RouterLink>
        <button
          type="button"
          class="ml-2 grid h-9 w-9 place-items-center rounded-md text-surface-600 hover:bg-surface-100 dark:text-surface-300 dark:hover:bg-surface-800"
          :title="isDark ? 'Tema claro' : 'Tema escuro'"
          @click="toggleDark()"
        >
          <i :class="['pi', isDark ? 'pi-sun' : 'pi-moon']" />
        </button>
      </nav>
    </header>

    <main class="flex-1 overflow-auto bg-surface-50 dark:bg-surface-900">
      <RouterView />
    </main>

    <Toast position="bottom-right" />
    <ConfirmDialog />
  </div>
</template>
