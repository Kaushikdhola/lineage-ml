/**
 * Summary statistics for simulators: mean and a Student-t confidence interval across seeds.
 * Checked against scipy.stats in python/oracle.py.
 */

/** ln Γ(x) for x > 0 (Lanczos, g = 7, n = 9): ~15 significant digits. */
export function logGamma(x: number): number {
  if (!(x > 0)) throw new Error("logGamma: x must be positive");
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  const z = x - 1;
  let a = c[0];
  const t = z + 7.5;
  for (let i = 1; i < 9; i++) a += c[i] / (z + i);
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Continued fraction for the incomplete beta function (modified Lentz). */
function betaContinuedFraction(a: number, b: number, x: number): number {
  const tiny = 1e-300;
  let c = 1;
  let d = 1 - ((a + b) * x) / (a + 1);
  if (Math.abs(d) < tiny) d = tiny;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 500; m++) {
    const m2 = 2 * m;
    let num = (m * (b - m) * x) / ((a + m2 - 1) * (a + m2));
    d = 1 + num * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + num / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    h *= d * c;
    num = (-(a + m) * (a + b + m) * x) / ((a + m2) * (a + m2 + 1));
    d = 1 + num * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + num / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const step = d * c;
    h *= step;
    if (Math.abs(step - 1) < 1e-15) return h;
  }
  throw new Error("betaContinuedFraction: no convergence");
}

/** Regularized incomplete beta I_x(a, b). */
export function regularizedBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lnFront = logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x);
  // use the symmetry relation where the continued fraction converges fastest
  if (x < (a + 1) / (a + b + 2)) return (Math.exp(lnFront) * betaContinuedFraction(a, b, x)) / a;
  return 1 - (Math.exp(lnFront) * betaContinuedFraction(b, a, 1 - x)) / b;
}

/** CDF of Student's t with `df` degrees of freedom. */
export function studentTCdf(t: number, df: number): number {
  const x = df / (df + t * t);
  const tail = 0.5 * regularizedBeta(x, df / 2, 0.5);
  return t >= 0 ? 1 - tail : tail;
}

/** Quantile of Student's t: the t with P(T ≤ t) = p. Bisection on the CDF to machine precision. */
export function studentTQuantile(p: number, df: number): number {
  if (!(p > 0 && p < 1)) throw new Error("studentTQuantile: p must be in (0, 1)");
  if (p === 0.5) return 0;
  let lo = -1;
  let hi = 1;
  while (studentTCdf(lo, df) > p) lo *= 2;
  while (studentTCdf(hi, df) < p) hi *= 2;
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    if (studentTCdf(mid, df) < p) lo = mid;
    else hi = mid;
    if (hi - lo <= 1e-15 * Math.max(1, Math.abs(mid))) break;
  }
  return (lo + hi) / 2;
}

export interface MeanCI {
  n: number;
  mean: number;
  /** Sample standard deviation (n − 1). */
  sd: number;
  lo: number;
  hi: number;
}

/** Mean and two-sided t confidence interval (default 95%) of independent samples, e.g. one per seed. */
export function meanCI(xs: number[], level = 0.95): MeanCI {
  const n = xs.length;
  if (n < 2) throw new Error("meanCI: need at least 2 samples");
  if (xs.some((x) => !Number.isFinite(x))) throw new Error("meanCI: non-finite sample");
  const mean = xs.reduce((s, x) => s + x, 0) / n;
  const sd = Math.sqrt(xs.reduce((s, x) => s + (x - mean) ** 2, 0) / (n - 1));
  const half = studentTQuantile(0.5 + level / 2, n - 1) * (sd / Math.sqrt(n));
  return { n, mean, sd, lo: mean - half, hi: mean + half };
}
