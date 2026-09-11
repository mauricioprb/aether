<script setup lang="ts">
defineProps<{ title: string }>();
</script>

<template>
  <div class="flex flex-col items-center gap-3 px-6 py-14 text-center">
    <p class="text-sm text-surface-500">{{ title }}</p>
    <slot />
  </div>
</template>
