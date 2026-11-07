import type { Vec } from "./linalg";

/** log Σ exp(z), stable. Entries may be -Infinity (masked); at least one must be finite. */
export function logSumExp(z: Vec): number {
  let max = -Infinity;
  for (const x of z) if (x > max) max = x;
  if (max === -Infinity) throw new Error("logSumExp: all entries are -Infinity");
  let s = 0;
  for (const x of z) s += Math.exp(x - max);
  return max + Math.log(s);
}

/** softmax(z / τ)_i = exp(z_i/τ) / Σ_j exp(z_j/τ). τ = 1 is the plain softmax. */
export function softmax(z: Vec, temperature = 1): Vec {
  if (!(temperature > 0)) throw new Error("softmax: temperature must be > 0");
  const scaled = z.map((x) => x / temperature);
  const lse = logSumExp(scaled);
  return scaled.map((x) => Math.exp(x - lse));
}

export function logSoftmax(z: Vec, temperature = 1): Vec {
  const scaled = z.map((x) => x / temperature);
  const lse = logSumExp(scaled);
  return scaled.map((x) => x - lse);
}

/** Shannon entropy in nats; 0·log 0 = 0. */
export function entropy(p: Vec): number {
  let h = 0;
  for (const x of p) if (x > 0) h -= x * Math.log(x);
  return h;
}

export function sigmoid(x: number): number {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

/** log σ(x), stable for large |x|. */
export function logSigmoid(x: number): number {
  return x >= 0 ? -Math.log1p(Math.exp(-x)) : x - Math.log1p(Math.exp(x));
}
