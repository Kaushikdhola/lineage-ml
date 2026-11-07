import { dot, type Mat, type Vec } from "./linalg";
import { entropy, softmax } from "./softmax";

/**
 * Why divide by √d_k? If q and k have independent zero-mean, unit-variance entries,
 * Var(q·k) = d_k. Large-variance logits push softmax toward one-hot (low entropy,
 * vanishing gradients). Scaling by 1/√d_k restores unit variance.
 */
export function dotProductMoments(qs: Mat, ks: Mat): { mean: number; variance: number } {
  if (qs.length !== ks.length || qs.length < 2) throw new Error("need ≥ 2 paired samples");
  const xs = qs.map((q, i) => dot(q, ks[i]));
  const mean = xs.reduce((s, x) => s + x, 0) / xs.length;
  const variance = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / (xs.length - 1); // unbiased
  return { mean, variance };
}

/** Softmax over one query's scores against keys K, with or without 1/√d scaling. */
export function softmaxSaturation(q: Vec, K: Mat, scaled: boolean): { entropy: number; maxProb: number } {
  const s = scaled ? 1 / Math.sqrt(q.length) : 1;
  const p = softmax(K.map((k) => dot(q, k) * s));
  return { entropy: entropy(p), maxProb: Math.max(...p) };
}
