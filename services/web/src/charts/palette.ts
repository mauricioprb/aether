/** Cores de dado. Separadas do verde de marca: o acento da UI nunca codifica valor. */
export const chartAccent = (isDark: boolean) => (isDark ? "#4FB593" : "#0D6B52");
export const chartRef = (isDark: boolean) => (isDark ? "#D99C4E" : "#B5792A");

/** Rampa divergente em torno de ΔG_H = 0 (ótimo de Sabatier).
 *  Negativo: adsorção forte demais. Positivo: fraca demais. */
export function dgColor(dg: number, chemAccEv = 0.043, isDark = false): string {
  if (Math.abs(dg) < chemAccEv) return isDark ? "#4FB593" : "#0D6B52";
  if (dg < 0) return isDark ? "#7E95D4" : "#3F5A9C";
  return isDark ? "#D99C4E" : "#B5792A";
}

/** Rampa divergente de 5 degraus para o mapa periódico. O zero de Sabatier é o
 *  ótimo, então a escala diverge dele: azul liga forte demais, âmbar liga fraca
 *  demais. Cinco degraus mostram o gradiente do vulcão sem virar borrão. */
const RAMP_LIGHT = ["#2f4a8c", "#7d92c4", "#0D6B52", "#c99a52", "#9c6415"];
const RAMP_DARK = ["#6B82C4", "#9FB0DC", "#4FB593", "#D99C4E", "#B87A28"];

export function dgBucket(dg: number, chemAccEv = 0.043): number {
  if (Math.abs(dg) < chemAccEv) return 2;
  if (dg < -0.2) return 0;
  if (dg < 0) return 1;
  if (dg > 0.2) return 4;
  return 3;
}

export const DG_BANDS = [
  { label: "liga muito forte", hint: "ΔG_H < −0,20 eV" },
  { label: "liga forte", hint: "−0,20 a −0,043 eV" },
  { label: "ótimo", hint: "|ΔG_H| < 43 meV" },
  { label: "liga fraca", hint: "0,043 a 0,20 eV" },
  { label: "liga muito fraca", hint: "ΔG_H > 0,20 eV" },
];

export function dgRamp(dg: number, isDark = false, chemAccEv = 0.043): string {
  return (isDark ? RAMP_DARK : RAMP_LIGHT)[dgBucket(dg, chemAccEv)]!;
}

/** Tinta legível sobre cada degrau da rampa. Escolhida por razão de contraste
 *  medida, não por tema: branco sobre a rampa escura dá 2,2:1 a 3,7:1, abaixo
 *  do mínimo AA de 4,5:1. Com estas tintas o pior caso é 4,94:1. */
const INK_LIGHT = ["#ffffff", "#111111", "#ffffff", "#111111", "#ffffff"];
const INK_DARK = ["#111111", "#111111", "#111111", "#111111", "#111111"];

export function dgInk(dg: number, isDark = false, chemAccEv = 0.043): string {
  return (isDark ? INK_DARK : INK_LIGHT)[dgBucket(dg, chemAccEv)]!;
}

export function bandColor(i: number, isDark = false): string {
  return (isDark ? RAMP_DARK : RAMP_LIGHT)[i]!;
}

/** Identidade de modelo é categórica, não valor: por isso fica fora do eixo
 *  azul→verde→âmbar da rampa de ΔG. Sem isso, azul e verde significavam "liga
 *  forte" e "ótimo" numa tela e "SchNet" e "ETR" na outra. */
// O quarto é neutro de propósito: âmbar colidiria com a linha de acurácia
// química no mesmo gráfico, e verde com o "ótimo" da rampa de ΔG.
const MODEL_LIGHT = ["#6D42A8", "#B03A63", "#2D6E80", "#3A4A54"];
const MODEL_DARK = ["#A98BDB", "#E08BA6", "#6FB3C4", "#B4C4CB"];

export function modelColor(index: number, isDark = false): string {
  const set = isDark ? MODEL_DARK : MODEL_LIGHT;
  return set[index % set.length]!;
}

/** Neutros e superfícies dos gráficos, num lugar só. Antes cada componente
 *  escolhia o seu: dois usavam slate do Tailwind (azulado) e um usava o neutro
 *  do app (esverdeado), então grade, texto e tooltip não batiam entre telas. */
export const chartTheme = (isDark: boolean) => ({
  text: isDark ? "#b9c3c2" : "#4a5658",
  grid: isDark ? "#2a3234" : "#e2e6e5",
  surface: isDark ? "#171c1d" : "#ffffff",
  faint: isDark ? "#4a5658" : "#aab4b4",
});
