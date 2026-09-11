<script setup lang="ts">
import { computed, ref } from "vue";
import { useDark, useMediaQuery } from "@vueuse/core";
import { DG_BANDS, bandColor, dgInk, dgRamp } from "@/charts/palette";
import { num } from "@/format";
import type { ElementStat } from "@/api";

const selected = defineModel<string[]>({ required: true });

const props = defineProps<{
  available: string[];
  stats?: ElementStat[];
  minSample?: number;
}>();

type El = [number, string, number, number];

const MAIN: El[] = [
  [1, "H", 1, 1],
  [2, "He", 18, 1],
  [3, "Li", 1, 2],
  [4, "Be", 2, 2],
  [5, "B", 13, 2],
  [6, "C", 14, 2],
  [7, "N", 15, 2],
  [8, "O", 16, 2],
  [9, "F", 17, 2],
  [10, "Ne", 18, 2],
  [11, "Na", 1, 3],
  [12, "Mg", 2, 3],
  [13, "Al", 13, 3],
  [14, "Si", 14, 3],
  [15, "P", 15, 3],
  [16, "S", 16, 3],
  [17, "Cl", 17, 3],
  [18, "Ar", 18, 3],
  [19, "K", 1, 4],
  [20, "Ca", 2, 4],
  [21, "Sc", 3, 4],
  [22, "Ti", 4, 4],
  [23, "V", 5, 4],
  [24, "Cr", 6, 4],
  [25, "Mn", 7, 4],
  [26, "Fe", 8, 4],
  [27, "Co", 9, 4],
  [28, "Ni", 10, 4],
  [29, "Cu", 11, 4],
  [30, "Zn", 12, 4],
  [31, "Ga", 13, 4],
  [32, "Ge", 14, 4],
  [33, "As", 15, 4],
  [34, "Se", 16, 4],
  [35, "Br", 17, 4],
  [36, "Kr", 18, 4],
  [37, "Rb", 1, 5],
  [38, "Sr", 2, 5],
  [39, "Y", 3, 5],
  [40, "Zr", 4, 5],
  [41, "Nb", 5, 5],
  [42, "Mo", 6, 5],
  [43, "Tc", 7, 5],
  [44, "Ru", 8, 5],
  [45, "Rh", 9, 5],
  [46, "Pd", 10, 5],
  [47, "Ag", 11, 5],
  [48, "Cd", 12, 5],
  [49, "In", 13, 5],
  [50, "Sn", 14, 5],
  [51, "Sb", 15, 5],
  [52, "Te", 16, 5],
  [53, "I", 17, 5],
  [54, "Xe", 18, 5],
  [55, "Cs", 1, 6],
  [56, "Ba", 2, 6],
  [72, "Hf", 4, 6],
  [73, "Ta", 5, 6],
  [74, "W", 6, 6],
  [75, "Re", 7, 6],
  [76, "Os", 8, 6],
  [77, "Ir", 9, 6],
  [78, "Pt", 10, 6],
  [79, "Au", 11, 6],
  [80, "Hg", 12, 6],
  [81, "Tl", 13, 6],
  [82, "Pb", 14, 6],
  [83, "Bi", 15, 6],
  [84, "Po", 16, 6],
  [85, "At", 17, 6],
  [86, "Rn", 18, 6],
  [87, "Fr", 1, 7],
  [88, "Ra", 2, 7],
  [104, "Rf", 4, 7],
  [105, "Db", 5, 7],
  [106, "Sg", 6, 7],
  [107, "Bh", 7, 7],
  [108, "Hs", 8, 7],
  [109, "Mt", 9, 7],
  [110, "Ds", 10, 7],
  [111, "Rg", 11, 7],
  [112, "Cn", 12, 7],
  [113, "Nh", 13, 7],
  [114, "Fl", 14, 7],
  [115, "Mc", 15, 7],
  [116, "Lv", 16, 7],
  [117, "Ts", 17, 7],
  [118, "Og", 18, 7],
];

const FBLOCK: El[] = [
  [57, "La", 3, 9],
  [58, "Ce", 4, 9],
  [59, "Pr", 5, 9],
  [60, "Nd", 6, 9],
  [61, "Pm", 7, 9],
  [62, "Sm", 8, 9],
  [63, "Eu", 9, 9],
  [64, "Gd", 10, 9],
  [65, "Tb", 11, 9],
  [66, "Dy", 12, 9],
  [67, "Ho", 13, 9],
  [68, "Er", 14, 9],
  [69, "Tm", 15, 9],
  [70, "Yb", 16, 9],
  [71, "Lu", 17, 9],
  [89, "Ac", 3, 10],
  [90, "Th", 4, 10],
  [91, "Pa", 5, 10],
  [92, "U", 6, 10],
  [93, "Np", 7, 10],
  [94, "Pu", 8, 10],
  [95, "Am", 9, 10],
  [96, "Cm", 10, 10],
  [97, "Bk", 11, 10],
  [98, "Cf", 12, 10],
  [99, "Es", 13, 10],
  [100, "Fm", 14, 10],
  [101, "Md", 15, 10],
  [102, "No", 16, 10],
  [103, "Lr", 17, 10],
];

const NAMES: Record<string, string> = {
  H: "Hidrogênio",
  He: "Hélio",
  Li: "Lítio",
  Be: "Berílio",
  B: "Boro",
  C: "Carbono",
  N: "Nitrogênio",
  O: "Oxigênio",
  F: "Flúor",
  Ne: "Neônio",
  Na: "Sódio",
  Mg: "Magnésio",
  Al: "Alumínio",
  Si: "Silício",
  P: "Fósforo",
  S: "Enxofre",
  Cl: "Cloro",
  Ar: "Argônio",
  K: "Potássio",
  Ca: "Cálcio",
  Sc: "Escândio",
  Ti: "Titânio",
  V: "Vanádio",
  Cr: "Cromo",
  Mn: "Manganês",
  Fe: "Ferro",
  Co: "Cobalto",
  Ni: "Níquel",
  Cu: "Cobre",
  Zn: "Zinco",
  Ga: "Gálio",
  Ge: "Germânio",
  As: "Arsênio",
  Se: "Selênio",
  Br: "Bromo",
  Kr: "Criptônio",
  Rb: "Rubídio",
  Sr: "Estrôncio",
  Y: "Ítrio",
  Zr: "Zircônio",
  Nb: "Nióbio",
  Mo: "Molibdênio",
  Tc: "Tecnécio",
  Ru: "Rutênio",
  Rh: "Ródio",
  Pd: "Paládio",
  Ag: "Prata",
  Cd: "Cádmio",
  In: "Índio",
  Sn: "Estanho",
  Sb: "Antimônio",
  Te: "Telúrio",
  I: "Iodo",
  Xe: "Xenônio",
  Cs: "Césio",
  Ba: "Bário",
  La: "Lantânio",
  Ce: "Cério",
  Pr: "Praseodímio",
  Nd: "Neodímio",
  Pm: "Promécio",
  Sm: "Samário",
  Eu: "Európio",
  Gd: "Gadolínio",
  Tb: "Térbio",
  Dy: "Disprósio",
  Ho: "Hólmio",
  Er: "Érbio",
  Tm: "Túlio",
  Yb: "Itérbio",
  Lu: "Lutécio",
  Hf: "Háfnio",
  Ta: "Tântalo",
  W: "Tungstênio",
  Re: "Rênio",
  Os: "Ósmio",
  Ir: "Irídio",
  Pt: "Platina",
  Au: "Ouro",
  Hg: "Mercúrio",
  Tl: "Tálio",
  Pb: "Chumbo",
  Bi: "Bismuto",
  Po: "Polônio",
  At: "Astato",
  Rn: "Radônio",
  Fr: "Frâncio",
  Ra: "Rádio",
  Ac: "Actínio",
  Th: "Tório",
  Pa: "Protactínio",
  U: "Urânio",
  Np: "Netúnio",
  Pu: "Plutônio",
  Am: "Amerício",
  Cm: "Cúrio",
  Bk: "Berquélio",
  Cf: "Califórnio",
  Es: "Einstênio",
  Fm: "Férmio",
  Md: "Mendelévio",
  No: "Nobélio",
  Lr: "Laurêncio",
  Rf: "Rutherfórdio",
  Db: "Dúbnio",
  Sg: "Seabórgio",
  Bh: "Bóhrio",
  Hs: "Hássio",
  Mt: "Meitnério",
  Ds: "Darmstácio",
  Rg: "Roentgênio",
  Cn: "Copernício",
  Nh: "Nihônio",
  Fl: "Fleróvio",
  Mc: "Moscóvio",
  Lv: "Livermório",
  Ts: "Tennesso",
  Og: "Oganessônio",
};

const isDark = useDark();

const statBySymbol = computed(() => new Map((props.stats ?? []).map((s) => [s.symbol, s])));

/** Só entra na escala de cor com amostra suficiente: mediana de n=1 é ruído
 *  colorido, e C tem 1 estrutura na base, O tem 3. */
function statFor(sym: string): ElementStat | null {
  const st = statBySymbol.value.get(sym);
  if (!st || st.n < (props.minSample ?? 10)) return null;
  return st;
}

function cellStyle(sym: string) {
  const st = statFor(sym);
  const selected = isSelected(sym);
  // Seleção é moldura na borda da célula, não um retângulo flutuando dentro
  // dela: o anel interno lia como adesivo e brigava com o preenchimento.
  const ink = isDark.value ? "#f4f7f6" : "#12191a";
  const mark = selected ? { borderColor: ink, borderWidth: "2px", fontWeight: "700" } : {};
  if (!st) return mark;
  const c = dgRamp(st.median_dG, isDark.value);
  return {
    backgroundColor: c,
    borderColor: c,
    color: dgInk(st.median_dG, isDark.value),
    ...mark,
  };
}

function cellTitle(sym: string) {
  const name = NAMES[sym] ?? sym;
  if (!availableSet.value.has(sym)) return `${name}, ausente na base`;
  const st = statBySymbol.value.get(sym);
  if (!st) return name;
  const suffix =
    st.n < (props.minSample ?? 10)
      ? `${st.n} estrutura${st.n === 1 ? "" : "s"}, amostra insuficiente`
      : `${st.n} estruturas, mediana ΔG_H ${num(st.median_dG, 3)} eV`;
  return `${name}: ${suffix}`;
}

const availableSet = computed(() => new Set(props.available));
const selectedSet = computed(() => new Set(selected.value));

const isDesktop = useMediaQuery("(min-width: 768px)");
const search = ref("");

const availableList = computed(() =>
  [...MAIN, ...FBLOCK]
    .filter(([, sym]) => availableSet.value.has(sym))
    .sort((a, b) => a[0] - b[0])
    .map(([z, sym]) => ({ z, sym, name: NAMES[sym] ?? sym })),
);

const filteredList = computed(() => {
  const q = search.value.trim().toLowerCase();
  if (!q) return availableList.value;
  return availableList.value.filter(
    (e) => e.sym.toLowerCase().includes(q) || e.name.toLowerCase().includes(q),
  );
});

function isAvailable(sym: string) {
  return availableSet.value.has(sym);
}
function isSelected(sym: string) {
  return selectedSet.value.has(sym);
}

function toggle(sym: string) {
  if (!isAvailable(sym)) return;
  selected.value = isSelected(sym)
    ? selected.value.filter((s) => s !== sym)
    : [...selected.value, sym];
}

function clearAll() {
  selected.value = [];
}
</script>

<template>
  <div>
    <div v-if="isDesktop">
      <div class="overflow-x-auto pb-1">
        <div
          class="grid min-w-[34rem] gap-1"
          style="grid-template-columns: repeat(18, minmax(0, 1fr))"
        >
          <button
            v-for="[, sym, col, row] in MAIN"
            :key="sym"
            type="button"
            :disabled="!isAvailable(sym)"
            :title="cellTitle(sym)"
            :style="{ gridColumn: col, gridRow: row, ...cellStyle(sym) }"
            class="relative aspect-square rounded border text-center transition select-none focus-visible:ring-2 focus-visible:ring-surface-900 focus-visible:outline-none dark:focus-visible:ring-surface-0"
            :class="[
              isAvailable(sym)
                ? 'cursor-pointer hover:brightness-110'
                : 'cursor-default border-transparent bg-surface-100 text-surface-300 dark:bg-surface-800/50 dark:text-surface-600',
              !statFor(sym) && isAvailable(sym)
                ? 'border-transparent bg-surface-200 text-surface-500 dark:bg-surface-700/50 dark:text-surface-400'
                : '',
            ]"
            @click="toggle(sym)"
          >
            <span class="grid h-full place-items-center text-xs font-semibold">{{ sym }}</span>
          </button>
        </div>
      </div>

      <div class="mt-1 overflow-x-auto pb-1">
        <div
          class="grid min-w-[34rem] gap-1"
          style="grid-template-columns: repeat(18, minmax(0, 1fr))"
        >
          <button
            v-for="[, sym, col, row] in FBLOCK"
            :key="sym"
            type="button"
            :disabled="!isAvailable(sym)"
            :title="cellTitle(sym)"
            :style="{ gridColumn: col, gridRow: row - 8, ...cellStyle(sym) }"
            class="relative aspect-square rounded border text-center transition select-none focus-visible:ring-2 focus-visible:ring-surface-900 focus-visible:outline-none dark:focus-visible:ring-surface-0"
            :class="[
              isAvailable(sym)
                ? 'cursor-pointer hover:brightness-110'
                : 'cursor-default border-transparent bg-surface-100 text-surface-300 dark:bg-surface-800/50 dark:text-surface-600',
              !statFor(sym) && isAvailable(sym)
                ? 'border-transparent bg-surface-200 text-surface-500 dark:bg-surface-700/50 dark:text-surface-400'
                : '',
            ]"
            @click="toggle(sym)"
          >
            <span class="grid h-full place-items-center text-xs font-semibold">{{ sym }}</span>
          </button>
        </div>
      </div>

      <div class="mt-3">
        <div class="mb-1.5 text-xs font-medium text-surface-500">mediana de ΔG_H (eV)</div>
        <div
          class="flex"
          role="img"
          aria-label="Escala divergente de ΔG_H, de ligação forte a fraca"
        >
          <span
            v-for="(b, i) in DG_BANDS"
            :key="b.label"
            class="h-2 flex-1"
            :style="{ backgroundColor: bandColor(i, isDark) }"
            :title="`${b.label}, ${b.hint}`"
          ></span>
        </div>
        <div class="mt-1 flex justify-between font-mono text-2xs tabular-nums text-surface-400">
          <span>−0,20</span>
          <span>0</span>
          <span>+0,20</span>
        </div>
        <div class="flex justify-between text-2xs text-surface-500">
          <span>liga forte</span>
          <span>ótimo</span>
          <span>liga fraca</span>
        </div>
        <p class="mt-2 text-2xs text-surface-400">
          Sem cor: elemento fora da base, ou com menos de {{ minSample ?? 10 }} estruturas.
        </p>
      </div>
    </div>

    <div v-else>
      <div class="relative">
        <i class="pi pi-search absolute left-3 top-1/2 -translate-y-1/2 text-xs text-surface-400" />
        <input
          v-model="search"
          type="text"
          inputmode="search"
          placeholder="Buscar elemento (ex.: platina, Pt)"
          class="w-full rounded-lg border border-surface-300 bg-surface-0 py-2.5 pl-9 pr-3 text-sm outline-none transition placeholder:text-surface-400 focus:border-primary-500 dark:border-surface-700 dark:bg-surface-900"
        />
      </div>

      <ul
        class="mt-2 max-h-72 overflow-y-auto rounded-lg border border-surface-200 dark:border-surface-800"
      >
        <li v-if="!filteredList.length" class="px-4 py-6 text-center text-sm text-surface-500">
          Nenhum elemento encontrado.
        </li>
        <li v-for="e in filteredList" :key="e.sym">
          <button
            type="button"
            class="flex w-full items-center gap-3 border-b border-surface-100 px-3 py-2.5 text-left transition last:border-b-0 dark:border-surface-800/60"
            :class="
              isSelected(e.sym)
                ? 'bg-primary-50 dark:bg-primary-950/40'
                : 'hover:bg-surface-50 dark:hover:bg-surface-800/50'
            "
            @click="toggle(e.sym)"
          >
            <span
              class="grid h-5 w-5 shrink-0 place-items-center rounded border transition"
              :class="
                isSelected(e.sym)
                  ? 'border-primary-500 bg-primary-500 text-white'
                  : 'border-surface-300 dark:border-surface-600'
              "
            >
              <i v-if="isSelected(e.sym)" class="pi pi-check text-2xs" />
            </span>
            <span
              class="w-9 shrink-0 font-mono text-sm font-semibold text-surface-800 dark:text-surface-100"
              >{{ e.sym }}</span
            >
            <span class="flex-1 text-sm text-surface-700 dark:text-surface-200">{{ e.name }}</span>
            <span class="text-2xs tabular-nums text-surface-400">{{ e.z }}</span>
          </button>
        </li>
      </ul>

      <div v-if="selected.length" class="mt-2 flex items-center justify-between gap-2 text-xs">
        <span class="text-surface-500">{{ selected.length }} selecionado(s)</span>
        <button
          type="button"
          class="font-medium text-primary-600 hover:underline dark:text-primary-400"
          @click="clearAll"
        >
          Limpar
        </button>
      </div>
    </div>
  </div>
</template>
