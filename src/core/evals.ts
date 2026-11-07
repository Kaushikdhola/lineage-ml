import type { Mat, Vec } from "./linalg";
import { normalQuantile } from "./normal";

/** Wilson (1927) score interval for a binomial proportion. */
export function wilsonInterval(successes: number, n: number, alpha = 0.05): [number, number] {
  if (n <= 0) throw new Error("wilsonInterval: n must be > 0");
  const z = normalQuantile(1 - alpha / 2);
  const p = successes / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const center = (p + z2 / (2 * n)) / denom;
  const half = (z / denom) * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n));
  return [Math.max(0, center - half), Math.min(1, center + half)];
}

/** Exact McNemar (1947) test on discordant pairs b (only A correct) and c (only B correct). */
export function mcnemarExact(b: number, c: number): number {
  const n = b + c;
  if (n === 0) return 1;
  const k = Math.min(b, c);
  const logHalfN = n * Math.log(0.5);
  let logC = 0;
  const terms: number[] = [logHalfN];
  for (let i = 1; i <= k; i++) {
    logC += Math.log(n - i + 1) - Math.log(i);
    terms.push(logC + logHalfN);
  }
  const m = Math.max(...terms);
  const cdf = Math.exp(m) * terms.reduce((s, t) => s + Math.exp(t - m), 0);
  return Math.min(1, 2 * cdf);
}

/**
 * Unbiased pass@k (Chen et al., 2021): 1 − C(n−c, k) / C(n, k), from n samples with c correct.
 * Numerically stable product form.
 */
export function passAtK(n: number, c: number, k: number): number {
  if (k > n || c > n || c < 0 || k < 1) throw new Error("passAtK: need 0 ≤ c ≤ n and 1 ≤ k ≤ n");
  if (n - c < k) return 1;
  let prod = 1;
  for (let i = n - c + 1; i <= n; i++) prod *= 1 - k / i;
  return 1 - prod;
}

/** pass^k (Yao et al., 2024, τ-bench): probability that k i.i.d. trials ALL succeed, C(c, k) / C(n, k). */
export function passHatK(n: number, c: number, k: number): number {
  if (k > n || c > n || c < 0 || k < 1) throw new Error("passHatK: need 0 ≤ c ≤ n and 1 ≤ k ≤ n");
  if (c < k) return 0;
  let prod = 1;
  for (let i = 0; i < k; i++) prod *= (c - i) / (n - i);
  return prod;
}

/** Bradley–Terry (1952): P(i beats j) = π_i / (π_i + π_j) = σ(β_i − β_j) with β = log π. */
export function bradleyTerryProb(betaI: number, betaJ: number): number {
  const d = betaI - betaJ;
  return d >= 0 ? 1 / (1 + Math.exp(-d)) : Math.exp(d) / (1 + Math.exp(d));
}

/**
 * Maximum-likelihood Bradley–Terry strengths from a wins matrix (wins[i][j] = times i beat j),
 * via the MM algorithm (Hunter, 2004). Returns log-strengths β centered to mean 0.
 */
export function bradleyTerryFit(wins: Mat, tol = 1e-12, maxIter = 100_000): Vec {
  const n = wins.length;
  let p = new Array<number>(n).fill(1);
  const W = wins.map((row) => row.reduce((s, x) => s + x, 0));
  if (W.some((w) => w === 0))
    throw new Error("bradleyTerryFit: every player needs at least one win (MLE otherwise diverges)");
  for (let it = 0; it < maxIter; it++) {
    const next = p.map((pi, i) => {
      let denom = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const nij = wins[i][j] + wins[j][i];
        if (nij > 0) denom += nij / (pi + p[j]);
      }
      return W[i] / denom;
    });
    const g = Math.exp(next.reduce((s, x) => s + Math.log(x), 0) / n);
    const normed = next.map((x) => x / g);
    const delta = Math.max(...normed.map((x, i) => Math.abs(Math.log(x) - Math.log(p[i]))));
    p = normed;
    if (delta < tol) break;
  }
  return p.map(Math.log);
}

/** Elo (1978): expected score E_A = 1 / (1 + 10^((R_B − R_A)/400)). */
export function eloExpected(ra: number, rb: number): number {
  return 1 / (1 + Math.pow(10, (rb - ra) / 400));
}

/** Elo update for A after scoring S ∈ {0, 0.5, 1} against B. */
export function eloUpdate(ra: number, rb: number, score: number, k = 32): number {
  return ra + k * (score - eloExpected(ra, rb));
}
