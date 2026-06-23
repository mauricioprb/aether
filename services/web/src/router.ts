import { createRouter, createWebHistory, type RouteRecordRaw } from "vue-router";

const routes: RouteRecordRaw[] = [
  {
    path: "/",
    name: "home",
    component: () => import("@/views/HomeView.vue"),
    meta: { title: "Início" },
  },
  {
    path: "/screen",
    name: "screen",
    component: () => import("@/views/ScreenView.vue"),
    meta: { title: "Buscar materiais" },
  },
  {
    path: "/compare",
    name: "compare",
    component: () => import("@/views/CompareView.vue"),
    meta: { title: "Comparar métodos" },
  },
  {
    path: "/about",
    name: "about",
    component: () => import("@/views/AboutView.vue"),
    meta: { title: "Sobre" },
  },
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
});

router.afterEach((to) => {
  const base = "HIDRA";
  const title = to.meta.title as string | undefined;
  document.title = title ? `${title} - ${base}` : base;
});
