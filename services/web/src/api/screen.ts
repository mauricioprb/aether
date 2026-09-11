import { http } from "./client";
import type {
  ComparisonPredictionsResponse,
  ComparisonResponse,
  CountResponse,
  ElementStatsResponse,
  ModelName,
  PredictResponse,
  StructureFormat,
  ScreenRequest,
  ScreenResponse,
  StatsResponse,
} from "./types";

export async function fetchStats(): Promise<StatsResponse> {
  const { data } = await http.get<StatsResponse>("/stats");
  return data;
}

export async function fetchElements(): Promise<string[]> {
  const { data } = await http.get<{ elements: string[] }>("/elements");
  return data.elements;
}

/** Quantas estruturas casam com o filtro, sem rodar modelo. */
export async function fetchCount(
  elements: string[],
  excludeTrain: boolean,
): Promise<CountResponse> {
  const params = new URLSearchParams();
  elements.forEach((e) => params.append("elements", e));
  params.set("exclude_train", String(excludeTrain));
  const { data } = await http.get<CountResponse>(`/count?${params}`);
  return data;
}

export async function runScreen(req: ScreenRequest): Promise<ScreenResponse> {
  const { data } = await http.post<ScreenResponse>("/screen", req);
  return data;
}

export async function fetchElementStats(): Promise<ElementStatsResponse> {
  const { data } = await http.get<ElementStatsResponse>("/elements/stats");
  return data;
}

export async function fetchComparison(): Promise<ComparisonResponse> {
  const { data } = await http.get<ComparisonResponse>("/comparison");
  return data;
}

export async function fetchComparisonPredictions(): Promise<ComparisonPredictionsResponse> {
  const { data } = await http.get<ComparisonPredictionsResponse>("/comparison/predictions");
  return data;
}

/** Extensao -> formato que a API aceita. Fechada de proposito: o backend so
 *  habilita estes tres leitores, entao errar aqui viraria 422 sem explicacao. */
const FORMATS: [RegExp, StructureFormat][] = [
  [/\.cif$/i, "cif"],
  [/\.(xyz|extxyz)$/i, "extxyz"],
  [/(\.vasp$|^poscar|^contcar)/i, "vasp"],
];

export function formatOf(filename: string): StructureFormat | null {
  return FORMATS.find(([re]) => re.test(filename))?.[1] ?? null;
}

/** Predicao de estrutura arbitraria. O corpo e o arquivo cru: a API nao usa
 *  multipart, entao nao ha FormData nem dependencia extra. */
export async function runPredict(file: File, model: ModelName): Promise<PredictResponse> {
  const format = formatOf(file.name);
  if (!format) throw { status: 0, message: "Formato não reconhecido: use .cif, .xyz ou POSCAR" };
  const { data } = await http.post<PredictResponse>(
    `/predict?format=${format}&model=${model}`,
    await file.text(),
    { headers: { "Content-Type": "text/plain" } },
  );
  return data;
}
