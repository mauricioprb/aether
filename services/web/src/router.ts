import { createRouter, createWebHistory, type RouteRecordRaw } from "vue-router";

const routes: RouteRecordRaw[] = [
  { path: "/", redirect: "/screen" },
  {
    path: "/screen",
    name: "screen",
    component: () => import("@/views/ScreenView.vue"),
    meta: { title: "Triagem" },
  },
  {
    path: "/predict",
    name: "predict",
    component: () => import("@/views/PredictView.vue"),
    meta: { title: "Prever" },
  },
  {
    path: "/compare",
    name: "compare",
    component: () => import("@/views/CompareView.vue"),
    meta: { title: "Modelos" },
  },
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
});

router.afterEach((to) => {
  const base = "AETHER";
  const title = to.meta.title as string | undefined;
  document.title = title ? `${title} - ${base}` : base;
});
