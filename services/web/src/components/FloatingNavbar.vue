<script setup lang="ts">
import { ref, computed, onMounted } from "vue";
import { useDark, useToggle, useScroll, useMouse, useMediaQuery } from "@vueuse/core";
import { showShortcutsHelp } from "@/composables";

defineOptions({ name: "FloatingNavbar" });

const isDark = useDark({
  selector: "html",
  attribute: "class",
  valueDark: "dark",
  valueLight: "",
});
const toggleDark = useToggle(isDark);

const mobileOpen = ref(false);

const links = [
  { to: "/screen", label: "Buscar", icon: "pi-search" },
  { to: "/compare", label: "Comparar", icon: "pi-chart-bar" },
  { to: "/about", label: "Sobre", icon: "pi-info-circle" },
];

function closeMobile() {
  mobileOpen.value = false;
}

const scrollTarget = ref<HTMLElement | null>(null);
onMounted(() => {
  scrollTarget.value = document.querySelector("main");
});
const { y: scrollY } = useScroll(scrollTarget);
const { y: mouseY } = useMouse({ type: "client" });

const isDesktop = useMediaQuery("(min-width: 768px)");

const TOP_THRESHOLD = 80;
const MOUSE_THRESHOLD = 90;

const navbarVisible = computed(
  () => !isDesktop.value || scrollY.value < TOP_THRESHOLD || mouseY.value < MOUSE_THRESHOLD,
);
</script>

<template>
  <nav
    class="fixed top-0 left-0 right-0 z-50 flex justify-center px-4 pt-4 transition-transform duration-300 ease-out sm:px-6"
    :class="navbarVisible ? 'translate-y-0' : '-translate-y-full'"
  >
    <div
      class="flex w-full max-w-5xl items-center justify-between rounded-2xl border border-surface-200/60 bg-surface-0/80 px-5 py-3 shadow-lg shadow-surface-900/5 backdrop-blur-lg dark:border-surface-700/50 dark:bg-surface-900/75 dark:shadow-surface-950/40"
    >
      <RouterLink to="/" class="flex shrink-0 items-center gap-3" @click="closeMobile">
        <img src="/logo.svg" alt="HIDRA" class="h-9 w-9 shrink-0" />
        <div class="leading-tight hidden sm:block">
          <div class="text-sm font-semibold tracking-wide">HIDRA</div>
        </div>
      </RouterLink>

      <div class="hidden md:flex items-center gap-1">
        <RouterLink
          v-for="r in links"
          :key="r.to"
          :to="r.to"
          class="flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-surface-600 transition-colors hover:bg-surface-100 hover:text-surface-900 dark:text-surface-300 dark:hover:bg-surface-800 dark:hover:text-surface-0"
          active-class="bg-primary-50! text-primary-700! dark:bg-primary-950! dark:text-primary-300!"
        >
          <i :class="['pi', r.icon, 'text-xs']" />
          {{ r.label }}
        </RouterLink>
      </div>

      <div class="flex items-center gap-1">
        <button
          type="button"
          class="grid h-9 w-9 place-items-center rounded-lg text-surface-500 transition-colors hover:bg-surface-100 hover:text-surface-700 dark:hover:bg-surface-800 dark:hover:text-surface-200"
          title="Atalhos (?)"
          @click="showShortcutsHelp = true"
        >
          <i class="pi pi-question-circle text-sm" />
        </button>
        <button
          type="button"
          class="grid h-9 w-9 place-items-center rounded-lg text-surface-500 transition-colors hover:bg-surface-100 hover:text-surface-700 dark:hover:bg-surface-800 dark:hover:text-surface-200"
          :title="isDark ? 'Tema claro' : 'Tema escuro'"
          @click="toggleDark()"
        >
          <i :class="['pi text-sm', isDark ? 'pi-sun' : 'pi-moon']" />
        </button>

        <button
          type="button"
          class="grid h-9 w-9 place-items-center rounded-lg text-surface-600 transition-colors hover:bg-surface-100 dark:text-surface-300 dark:hover:bg-surface-800 md:hidden"
          :aria-label="mobileOpen ? 'Fechar menu' : 'Abrir menu'"
          @click="mobileOpen = !mobileOpen"
        >
          <i :class="['pi text-sm', mobileOpen ? 'pi-times' : 'pi-bars']" />
        </button>
      </div>
    </div>
  </nav>

  <Transition name="navbar-slide">
    <div
      v-if="mobileOpen"
      class="fixed top-24 left-4 right-4 z-40 rounded-2xl border border-surface-200/60 bg-surface-0/95 p-4 shadow-xl backdrop-blur-lg dark:border-surface-700/50 dark:bg-surface-900/95 md:hidden"
    >
      <RouterLink
        v-for="r in links"
        :key="r.to"
        :to="r.to"
        class="flex items-center gap-3 rounded-lg px-4 py-3 text-sm font-medium text-surface-700 transition-colors hover:bg-surface-100 dark:text-surface-200 dark:hover:bg-surface-800"
        active-class="bg-primary-50! text-primary-700! dark:bg-primary-950! dark:text-primary-300!"
        @click="closeMobile"
      >
        <i :class="['pi', r.icon, 'text-sm']" />
        {{ r.label }}
      </RouterLink>
    </div>
  </Transition>
</template>

<style scoped>
.navbar-slide-enter-active,
.navbar-slide-leave-active {
  transition: all 0.25s ease;
}
.navbar-slide-enter-from,
.navbar-slide-leave-to {
  opacity: 0;
  transform: translateY(-8px) scale(0.97);
}
</style>
