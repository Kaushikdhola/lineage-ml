/**
 * How a batch of per-token losses becomes one gradient: the weight each token receives.
 *
 * All functions return, per sequence, the weight on each target token (n_i = L_i − 1 targets).
 * For language-model losses the weights are multiplied by N (total targets) so that a global token
 * mean is exactly 1.0. For RL objectives they are the raw coefficients ∂L/∂ log π(token), with the
 * sign convention of a loss to MINIMISE (negative = the token is reinforced).
 */
export type Weights = number[][];

const targets = (lengths: number[]) => lengths.map((L) => L - 1);

export function tokenMean(lengths: number[]): Weights {
  return targets(lengths).map((n) => new Array<number>(n).fill(1));
}

export function sequenceMean(lengths: number[]): Weights {
  const n = targets(lengths);
  const N = n.reduce((s, x) => s + x, 0);
  return n.map((ni) => new Array<number>(ni).fill(N / (n.length * ni)));
}

export function meanOfMicrobatchMeans(lengths: number[], micro: number): Weights {
  const n = targets(lengths);
  const N = n.reduce((s, x) => s + x, 0);
  const k = Math.ceil(n.length / micro);
  const out: Weights = [];
  for (let i = 0; i < n.length; i += micro) {
    const g = n.slice(i, i + micro);
    const Ng = g.reduce((s, x) => s + x, 0);
    for (const ni of g) out.push(new Array<number>(ni).fill(N / (k * Ng)));
  }
  return out;
}

export function firstMicrobatchOnly(lengths: number[], micro: number): Weights {
  return targets(lengths).map((ni, i) => new Array<number>(ni).fill(i < micro ? 1 : 0));
}

/** Sample standard deviation (ddof = 1), as torch.std() computes by default. */
function std(x: number[]): number {
  const m = x.reduce((s, v) => s + v, 0) / x.length;
  return Math.sqrt(x.reduce((s, v) => s + (v - m) ** 2, 0) / (x.length - 1));
}

export type GroupObjective = "grpo" | "dr-grpo" | "dapo";

export interface GroupWeights {
  advantages: number[];
  weights: Weights; // per response, per token: ∂L/∂ log π at ratio = 1 (no clipping active)
}

/**
 * Per-token gradient coefficients for one group of G responses to the same prompt, on-policy
 * (ratio = 1, so clipping is inactive and KL is omitted).
 *
 *  GRPO (Shao et al., 2024):     A_i = (r_i − mean r) / (std r + eps),  L = −1/G Σ_i 1/|o_i| Σ_t A_i ρ_it
 *  Dr. GRPO (Liu et al., 2025):  A_i = r_i − mean r,                    L = −1/(G·Lmax) Σ_i Σ_t A_i ρ_it
 *  DAPO (Yu et al., 2025):       A_i as GRPO,                           L = −1/Σ_j|o_j| Σ_i Σ_t A_i ρ_it
 */
export function groupWeights(
  lengths: number[],
  rewards: number[],
  objective: GroupObjective,
  opts: { eps?: number; maxLength?: number } = {},
): GroupWeights {
  const G = lengths.length;
  if (rewards.length !== G) throw new Error("groupWeights: one reward per response");
  const eps = opts.eps ?? 1e-4;
  const mean = rewards.reduce((s, r) => s + r, 0) / G;
  const centred = rewards.map((r) => r - mean);
  const advantages = objective === "dr-grpo" ? centred : centred.map((c) => c / (std(rewards) + eps));
  const total = lengths.reduce((s, L) => s + L, 0);
  const Lmax = opts.maxLength ?? Math.max(...lengths);
  const weights = lengths.map((L, i) => {
    const norm = objective === "grpo" ? G * L : objective === "dapo" ? total : G * Lmax;
    return new Array<number>(L).fill(-advantages[i] / norm);
  });
  return { advantages, weights };
}
