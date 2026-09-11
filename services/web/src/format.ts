/** Número para leitor pt-BR: decimal com vírgula.
 *  Sem isto a tela mistura "5.860" (milhar) com "0.087" (decimal) e o ponto
 *  passa a significar duas coisas na mesma linha. */
export const num = (v: number, digits = 3) =>
  v.toLocaleString("pt-BR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
