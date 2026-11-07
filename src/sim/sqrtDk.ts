/**
 * Ablation: delete √d_k from scaled dot-product attention.
 *
 * Experiment: one attention layer learns key–value retrieval (src/core/retrieval.ts) with d_k = 64.
 * Intact divides the logits by √d_k; ablated does not. Everything else is identical: data, init,
 * optimizer (plain SGD), steps. To keep the comparison fair, each version uses its own best learning
 * rate, chosen on a 21-point log grid with held-out tuning seeds (100–103), and is then evaluated on
 * seeds 0–19. tests/sim.sqrtDk.test.ts re-derives both learning rates, so they cannot drift.
 */
import { meanCI, type MeanCI } from "../core/stats";
import { trainRetrieval, type RetrievalConfig, type RetrievalRun } from "../core/retrieval";
import { yieldToHost, type Band, type RunOptions, type SimReport, type Simulator } from "./types";

export const SQRT_DK_BASE: Omit<RetrievalConfig, "lr" | "scale"> = {
  dk: 64,
  nKeys: 16,
  dv: 16,
  batch: 8,
  steps: 200,
  noise: 0.5,
};

/** Learning-rate grid: 10^-4 … 10^1 in quarter decades. */
export const SQRT_DK_LR_GRID = Array.from({ length: 21 }, (_, i) => 10 ** (-4 + i * 0.25));
export const SQRT_DK_TUNING_SEEDS = [100, 101, 102, 103];

/** Best grid learning rates on the tuning seeds (verified by tests). */
export const SQRT_DK_LR = { intact: SQRT_DK_LR_GRID[13], ablated: SQRT_DK_LR_GRID[7] };

export const sqrtDkConfig = (ablated: boolean): RetrievalConfig => ({
  ...SQRT_DK_BASE,
  scale: ablated ? 1 : 1 / Math.sqrt(SQRT_DK_BASE.dk),
  lr: ablated ? SQRT_DK_LR.ablated : SQRT_DK_LR.intact,
});

function band(runs: RetrievalRun[], steps: number): Band {
  const mean: number[] = [];
  const lo: number[] = [];
  const hi: number[] = [];
  for (let t = 0; t < steps; t++) {
    const ci = meanCI(runs.map((r) => r.loss[t]));
    mean.push(ci.mean);
    lo.push(ci.lo);
    hi.push(ci.hi);
  }
  return { mean, lo, hi };
}

export interface SqrtDkRaw {
  intact: RetrievalRun[];
  ablated: RetrievalRun[];
}

/** Runs both versions over seeds 0…N−1. */
export async function runSqrtDkRaw(options: RunOptions = {}): Promise<SqrtDkRaw> {
  const n = options.seeds ?? 20;
  const intact: RetrievalRun[] = [];
  const ablated: RetrievalRun[] = [];
  for (let seed = 0; seed < n; seed++) {
    if (options.signal?.aborted) throw new DOMException("simulation stopped", "AbortError");
    intact.push(trainRetrieval(seed, sqrtDkConfig(false)));
    ablated.push(trainRetrieval(seed, sqrtDkConfig(true)));
    options.onProgress?.(seed + 1, n);
    await yieldToHost();
  }
  return { intact, ablated };
}

const ci = (runs: RetrievalRun[], f: (r: RetrievalRun) => number): MeanCI => meanCI(runs.map(f));

export function sqrtDkReport(raw: SqrtDkRaw): SimReport {
  const steps = SQRT_DK_BASE.steps;
  const n = raw.intact.length;
  return {
    id: "sqrt-dk",
    seeds: n,
    chart: {
      title: "Training loss",
      xLabel: "SGD step",
      yLabel: "retrieval loss",
      xs: Array.from({ length: steps }, (_, i) => i),
      intact: band(raw.intact, steps),
      ablated: band(raw.ablated, steps),
    },
    stats: [
      {
        label: "Final loss",
        hint: "Mean batch loss over the last 20 steps. Lower is better; 0 is perfect retrieval.",
        intact: ci(raw.intact, (r) => r.finalLoss),
        ablated: ci(raw.ablated, (r) => r.finalLoss),
        digits: 3,
      },
      {
        label: "Attention entropy at step 0",
        hint: `Nats; uniform over ${SQRT_DK_BASE.nKeys} keys is ln ${SQRT_DK_BASE.nKeys} = ${Math.log(SQRT_DK_BASE.nKeys).toFixed(2)}. Near 0 means each query has locked onto one key before learning anything.`,
        intact: ci(raw.intact, (r) => r.initEntropy),
        ablated: ci(raw.ablated, (r) => r.initEntropy),
        digits: 2,
      },
      {
        label: "Softmax slope at step 0",
        hint: "‖diag(a) − aaᵀ‖_F, how much the attention weights can still move. Saturated softmax ≈ 0.",
        intact: ci(raw.intact, (r) => r.initJacobianNorm),
        ablated: ci(raw.ablated, (r) => r.initJacobianNorm),
        digits: 3,
      },
    ],
    labels: { intact: "with √d_k", ablated: "without √d_k" },
    claim:
      "Without √d_k, the logits have variance d_k = 64, so the softmax starts saturated: each query locks onto one key at random. Even at its own best learning rate, training stalls far above the intact model.",
    method: `One attention layer learns key–value retrieval (d_k = ${SQRT_DK_BASE.dk}, ${SQRT_DK_BASE.nKeys} keys, plain SGD, ${steps} steps, batch ${SQRT_DK_BASE.batch}). Same data and initialization for both. Each version uses its own best learning rate from a 21-point grid, tuned on separate seeds (${SQRT_DK_LR.intact.toPrecision(2)} with √d_k, ${SQRT_DK_LR.ablated.toPrecision(2)} without). Bands are 95% t-intervals over ${n} seeds.`,
  };
}

export const sqrtDkSimulator: Simulator = {
  id: "sqrt-dk",
  run: async (options) => sqrtDkReport(await runSqrtDkRaw(options)),
};
