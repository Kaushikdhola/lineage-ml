/**
 * A tiny attention model that has to learn: key–value retrieval through one attention layer.
 *
 *   memory   K ∈ ℝ^{n×d_k} (keys), V ∈ ℝ^{n×d_v} (values), fixed
 *   input    x = k_t + noise, a corrupted copy of the target key t
 *   query    q = W x, the only learned parameter is W ∈ ℝ^{d_k×d_k}
 *   attend   a = softmax(c · q Kᵀ),  out = a V,  with c = 1/√d_k (intact) or 1 (ablated)
 *   loss     ½ ‖out − v_t‖², averaged over the batch
 *
 * The gradient is written out by hand (checked against finite differences in the oracle) so the
 * simulator can train in the browser. Everything is seeded with mulberry32 so runs reproduce
 * exactly, here and in python/oracle.py.
 */
import { dot } from "./linalg";
import { gaussian, mulberry32 } from "./random";
import { softmax } from "./softmax";

export interface RetrievalConfig {
  dk: number;
  nKeys: number;
  dv: number;
  batch: number;
  steps: number;
  /** Standard deviation of the noise added to the target key to form the input. */
  noise: number;
  lr: number;
  /** Multiplier on the logits q·k: 1/√d_k for the intact model, 1 for the ablated one. */
  scale: number;
}

export interface RetrievalProblem {
  K: number[][];
  V: number[][];
  /** Initial query projection, row-major d_k × d_k, entries N(0, 1/d_k). */
  W0: number[];
}

export interface RetrievalBatch {
  X: number[][];
  targets: number[];
}

export interface RetrievalStepResult {
  loss: number;
  /** dL/dW, row-major d_k × d_k. */
  grad: number[];
  /** Mean attention entropy over the batch, nats. */
  entropy: number;
  /** Mean Frobenius norm of the softmax Jacobian diag(a) − a aᵀ over the batch. */
  jacobianNorm: number;
}

/** Shannon entropy of a probability vector, nats (0 log 0 = 0). */
function entropyOf(a: number[]): number {
  let h = 0;
  for (const p of a) if (p > 0) h -= p * Math.log(p);
  return h;
}

/** ‖diag(a) − a aᵀ‖_F: how much the softmax output can still move. Zero when a is one-hot. */
export function softmaxJacobianNorm(a: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < a.length; j++) {
      const v = (i === j ? a[i] : 0) - a[i] * a[j];
      s += v * v;
    }
  return Math.sqrt(s);
}

/** Loss, gradient and diagnostics for one batch. Pure: does not modify W. */
export function retrievalLossGrad(
  W: number[],
  K: number[][],
  V: number[][],
  batch: RetrievalBatch,
  scale: number,
): RetrievalStepResult {
  const d = K[0].length;
  const n = K.length;
  const B = batch.X.length;
  const grad = new Array<number>(d * d).fill(0);
  let loss = 0;
  let ent = 0;
  let jac = 0;
  const q = new Array<number>(d);
  const dq = new Array<number>(d);
  for (let b = 0; b < B; b++) {
    const x = batch.X[b];
    const t = batch.targets[b];
    for (let i = 0; i < d; i++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += W[i * d + j] * x[j];
      q[i] = s;
    }
    const logits = K.map((k) => scale * dot(q, k));
    const a = softmax(logits);
    const out = V[0].map((_, c) => a.reduce((s, ak, k) => s + ak * V[k][c], 0));
    const r = out.map((o, c) => o - V[t][c]);
    loss += 0.5 * dot(r, r);
    ent += entropyOf(a);
    jac += softmaxJacobianNorm(a);
    // back through out = a V, then through the softmax
    const g = V.map((v) => dot(r, v));
    const ag = dot(a, g);
    const dS = a.map((ak, k) => ak * (g[k] - ag));
    dq.fill(0);
    for (let k = 0; k < n; k++) {
      const coef = scale * dS[k];
      for (let i = 0; i < d; i++) dq[i] += coef * K[k][i];
    }
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) grad[i * d + j] += (dq[i] * x[j]) / B;
  }
  return { loss: loss / B, grad, entropy: ent / B, jacobianNorm: jac / B };
}

/** The fixed memory and the initial weights for a seed. Draw order is part of the contract. */
export function retrievalProblem(
  rng: () => number,
  cfg: Pick<RetrievalConfig, "dk" | "nKeys" | "dv">,
): RetrievalProblem {
  const g = gaussian(rng);
  const K = Array.from({ length: cfg.nKeys }, () => Array.from({ length: cfg.dk }, g));
  const V = Array.from({ length: cfg.nKeys }, () => Array.from({ length: cfg.dv }, g));
  const s = 1 / Math.sqrt(cfg.dk);
  const W0 = Array.from({ length: cfg.dk * cfg.dk }, () => g() * s);
  return { K, V, W0 };
}

/** One training batch: for each example, a uniform target, then its noisy key. */
export function retrievalBatch(
  rng: () => number,
  K: number[][],
  batch: number,
  noise: number,
): RetrievalBatch {
  const g = gaussian(rng);
  const targets: number[] = [];
  const X: number[][] = [];
  for (let b = 0; b < batch; b++) {
    const t = Math.floor(rng() * K.length);
    targets.push(t);
    X.push(K[t].map((k) => k + noise * g()));
  }
  return { X, targets };
}

export interface RetrievalRun {
  /** Batch loss before each update, one per step. */
  loss: number[];
  initEntropy: number;
  initJacobianNorm: number;
  /** Mean batch loss over the last `tail` steps. */
  finalLoss: number;
  diverged: boolean;
}

/** Trains W with plain SGD from a seed. Deterministic. */
export function trainRetrieval(seed: number, cfg: RetrievalConfig, tail = 20): RetrievalRun {
  const rng = mulberry32(seed);
  const { K, V, W0 } = retrievalProblem(rng, cfg);
  const W = W0.slice();
  const loss: number[] = [];
  let initEntropy = NaN;
  let initJacobianNorm = NaN;
  let diverged = false;
  for (let step = 0; step < cfg.steps; step++) {
    const batch = retrievalBatch(rng, K, cfg.batch, cfg.noise);
    const r = retrievalLossGrad(W, K, V, batch, cfg.scale);
    if (step === 0) {
      initEntropy = r.entropy;
      initJacobianNorm = r.jacobianNorm;
    }
    loss.push(r.loss);
    if (!Number.isFinite(r.loss)) {
      diverged = true;
      break;
    }
    for (let i = 0; i < W.length; i++) W[i] -= cfg.lr * r.grad[i];
  }
  const last = loss.slice(-tail);
  return {
    loss,
    initEntropy,
    initJacobianNorm,
    finalLoss: last.reduce((s, x) => s + x, 0) / last.length,
    diverged,
  };
}
