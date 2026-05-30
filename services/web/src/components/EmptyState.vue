<script setup lang="ts">
defineProps<{
  icon?: string
  title: string
  description?: string
}>()
</script>

<template>
  <div class="flex flex-col items-center justify-center px-6 py-16 text-center">
    <div
      v-if="icon"
      class="mb-4 grid h-14 w-14 place-items-center rounded-full bg-surface-100 text-surface-400 dark:bg-surface-800"
    >
      <i :class="['pi', icon, 'text-xl']" />
    </div>
    <h3 class="text-base font-semibold text-surface-700 dark:text-surface-200">{{ title }}</h3>
    <p v-if="description" class="mt-1 max-w-sm text-sm text-surface-500">{{ description }}</p>
    <slot />
  </div>
</template>
