<script setup lang="ts">
withDefaults(defineProps<{ title?: string; padded?: boolean }>(), { padded: true });
</script>

<template>
  <section
    class="rounded-lg border border-surface-200 bg-surface-0 dark:border-surface-800 dark:bg-surface-950"
  >
    <header
      v-if="title || $slots.actions"
      class="flex flex-wrap items-center justify-between gap-3 border-b border-surface-200 px-4 py-2.5 dark:border-surface-800"
    >
      <h2 class="text-sm font-semibold text-surface-900 dark:text-surface-0">{{ title }}</h2>
      <div class="flex items-center gap-2">
        <slot name="actions" />
      </div>
    </header>
    <div :class="padded ? 'p-4' : ''">
      <slot />
    </div>
  </section>
</template>
