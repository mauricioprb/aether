import { ref } from "vue";
import { useRouter } from "vue-router";
import { useEventListener } from "@vueuse/core";

export const showShortcutsHelp = ref(false);

export function useGlobalShortcuts() {
  const router = useRouter();
  let lastG = 0;

  useEventListener("keydown", (e: KeyboardEvent) => {
    const target = e.target as HTMLElement | null;
    const tag = target?.tagName?.toLowerCase();
    const editable =
      tag === "input" || tag === "textarea" || tag === "select" || target?.isContentEditable;
    if (editable) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;

    if (e.key === "?") {
      e.preventDefault();
      showShortcutsHelp.value = true;
      return;
    }
    if (e.key === "Escape" && showShortcutsHelp.value) {
      showShortcutsHelp.value = false;
      return;
    }
    if (e.key === "g") {
      lastG = Date.now();
      return;
    }
    if (Date.now() - lastG < 800) {
      if (e.key === "h") router.push("/");
      if (e.key === "s") router.push("/screen");
      if (e.key === "c") router.push("/compare");
      lastG = 0;
    }
  });
}
