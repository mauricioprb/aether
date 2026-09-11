<script setup lang="ts">
defineProps<{ title: string }>();
</script>

<template>
  <header class="flex flex-col gap-3">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <h1 class="text-base font-semibold tracking-tight text-surface-900 dark:text-surface-0">
        {{ title }}
      </h1>
      <div v-if="$slots.actions" class="flex items-center gap-2">
        <slot name="actions" />
      </div>
    </div>
    <div
      v-if="$slots.default"
      class="flex flex-wrap items-baseline gap-x-6 gap-y-2 border-y border-surface-200 py-2.5 dark:border-surface-800"
    >
      <slot />
    </div>
  </header>
</template>
