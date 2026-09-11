<script setup lang="ts">
defineProps<{ label: string; value: string | number }>();
</script>

<template>
  <div class="flex items-baseline gap-1.5">
    <span class="font-mono text-sm font-medium tabular-nums text-surface-900 dark:text-surface-0">
      {{ value }}
    </span>
    <span class="text-xs text-surface-500">{{ label }}</span>
  </div>
</template>
