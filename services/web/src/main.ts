import { createApp } from "vue";
import { createPinia } from "pinia";
import { VueQueryPlugin } from "@tanstack/vue-query";
import PrimeVue from "primevue/config";
import Aura from "@primeuix/themes/aura";
import ToastService from "primevue/toastservice";
import ConfirmationService from "primevue/confirmationservice";
import { definePreset, palette } from "@primeuix/themes";

import App from "@/App.vue";
import { router } from "@/router";
import "@/main.css";
import "@/charts/setup";

const THEME_COLOR = "#38C649";

const SURFACE_LIGHT = "{neutral}";
const SURFACE_DARK = "{neutral}";

const surface = (color: string) => ({
  0: "#ffffff",
  ...(palette(color) as Record<string, string>),
});

const HidraPreset = definePreset(Aura, {
  semantic: {
    primary: palette(THEME_COLOR),
    colorScheme: {
      light: { surface: surface(SURFACE_LIGHT) },
      dark: { surface: surface(SURFACE_DARK) },
    },
  },
});

const app = createApp(App);

app.use(createPinia());
app.use(router);
app.use(VueQueryPlugin, {
  queryClientConfig: {
    defaultOptions: {
      queries: {
        retry: 1,
        staleTime: 60_000,
        refetchOnWindowFocus: false,
      },
    },
  },
});
app.use(PrimeVue, {
  theme: {
    preset: HidraPreset,
    options: {
      darkModeSelector: ".dark",
      cssLayer: {
        name: "primevue",
        order: "theme, base, primevue, components, utilities",
      },
    },
  },
  ripple: true,
});
app.use(ToastService);
app.use(ConfirmationService);

app.mount("#app");
