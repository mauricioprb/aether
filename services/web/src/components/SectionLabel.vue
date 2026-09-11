<script setup lang="ts">
defineProps<{ title: string }>();
</script>

<template>
  <div
    class="flex flex-wrap items-baseline justify-between gap-3 border-b border-surface-200 pb-2 dark:border-surface-800"
  >
    <h2 class="text-sm font-semibold text-surface-900 dark:text-surface-0">{{ title }}</h2>
    <div v-if="$slots.actions" class="flex items-center gap-2">
      <slot name="actions" />
    </div>
  </div>
</template>
