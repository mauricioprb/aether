<script setup lang="ts">
import Toast from "primevue/toast";
import ConfirmDialog from "primevue/confirmdialog";
import AppTopBar from "@/components/AppTopBar.vue";
import ShortcutsDialog from "@/components/ShortcutsDialog.vue";
import { useGlobalShortcuts } from "@/composables";

useGlobalShortcuts();
</script>

<template>
  <div class="flex h-full flex-col bg-surface-50 dark:bg-surface-900">
    <AppTopBar />

    <main class="min-w-0 flex-1 overflow-auto">
      <RouterView />
    </main>

    <Toast position="bottom-right" />
    <ConfirmDialog />
    <ShortcutsDialog />
  </div>
</template>
