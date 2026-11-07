/**
 * Standard normal functions at double precision: positive-term series for erf on |x| < 3,
 * Lentz continued fraction for erfc beyond. Quantile: Acklam + Newton refinement.
 */
const SQRT_PI = Math.sqrt(Math.PI);

function erfSeries(x: number): number {
  let term = x;
  let sum = x;
  const x2 = x * x;
  for (let n = 1; n < 500; n++) {
    term *= (2 * x2) / (2 * n + 1);
    sum += term;
    if (Math.abs(term) < 1e-17 * Math.abs(sum)) break;
  }
  return (2 / SQRT_PI) * Math.exp(-x2) * sum;
}

function erfcContinuedFraction(x: number): number {
  const tiny = 1e-300;
  let f = x;
  let C = f;
  let D = 0;
  for (let k = 1; k < 1000; k++) {
    const a = k / 2;
    D = x + a * D;
    D = Math.abs(D) < tiny ? 1 / tiny : 1 / D;
    C = x + a / C;
    if (Math.abs(C) < tiny) C = tiny;
    const delta = C * D;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-16) break;
  }
  return Math.exp(-x * x) / SQRT_PI / f;
}

export function erfc(x: number): number {
  if (x < 0) return 2 - erfc(-x);
  return x < 3 ? 1 - erfSeries(x) : erfcContinuedFraction(x);
}

export const normalCdf = (z: number) => 0.5 * erfc(-z / Math.SQRT2);
export const normalSf = (z: number) => 0.5 * erfc(z / Math.SQRT2);
export const normalPdf = (z: number) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

export function normalQuantile(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1,
    2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1,
    -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968,
    2.938163982698783,
  ];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  let x: number;
  if (p < 0.02425) {
    const q = Math.sqrt(-2 * Math.log(p));
    x =
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= 1 - 0.02425) {
    const q = p - 0.5;
    const r = q * q;
    x =
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x =
      -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  for (let i = 0; i < 2; i++) x -= (normalCdf(x) - p) / normalPdf(x);
  return x;
}
