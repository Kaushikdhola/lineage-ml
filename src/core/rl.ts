import type { Mat, Vec } from "./linalg";
import { logSigmoid, softmax } from "./softmax";

/** Returns G_t = Σ_{k≥0} γ^k r_{t+k} for every t. */
export function discountedReturns(rewards: Vec, gamma: number): Vec {
  const G = new Array<number>(rewards.length);
  let acc = 0;
  for (let t = rewards.length - 1; t >= 0; t--) {
    acc = rewards[t] + gamma * acc;
    G[t] = acc;
  }
  return G;
}

/** A finite MDP: P[s][a][s'] transition probabilities, R[s][a] expected reward. */
export interface MDP {
  P: number[][][];
  R: Mat;
}

/** Bellman expectation backup: V(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]. */
export function bellmanExpectationBackup(mdp: MDP, pi: Mat, V: Vec, gamma: number): Vec {
  return mdp.R.map((Rs, s) =>
    Rs.reduce(
      (acc, r, a) => acc + pi[s][a] * (r + gamma * mdp.P[s][a].reduce((x, p, s2) => x + p * V[s2], 0)),
      0,
    ),
  );
}

/** Iterative policy evaluation until the max change is below tol. */
export function policyEvaluation(mdp: MDP, pi: Mat, gamma: number, tol = 1e-12, maxIter = 100_000): Vec {
  let V = new Array<number>(mdp.R.length).fill(0);
  for (let it = 0; it < maxIter; it++) {
    const next = bellmanExpectationBackup(mdp, pi, V, gamma);
    const delta = Math.max(...next.map((v, i) => Math.abs(v - V[i])));
    V = next;
    if (delta < tol) return V;
  }
  throw new Error("policyEvaluation did not converge");
}

/** Q(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) V(s') */
export function actionValues(mdp: MDP, V: Vec, gamma: number): Mat {
  return mdp.R.map((Rs, s) =>
    Rs.map((r, a) => r + gamma * mdp.P[s][a].reduce((x, p, s2) => x + p * V[s2], 0)),
  );
}

/** Value iteration: V*(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V*(s')]. */
export function valueIteration(
  mdp: MDP,
  gamma: number,
  tol = 1e-12,
  maxIter = 100_000,
): { V: Vec; policy: number[] } {
  let V = new Array<number>(mdp.R.length).fill(0);
  for (let it = 0; it < maxIter; it++) {
    const Q = actionValues(mdp, V, gamma);
    const next = Q.map((row) => Math.max(...row));
    const delta = Math.max(...next.map((v, i) => Math.abs(v - V[i])));
    V = next;
    if (delta < tol) {
      const Qf = actionValues(mdp, V, gamma);
      return { V, policy: Qf.map((row) => row.indexOf(Math.max(...row))) };
    }
  }
  throw new Error("valueIteration did not converge");
}

/** Watkins Q-learning: Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]. Returns a new table. */
export function qLearningUpdate(
  Q: Mat,
  s: number,
  a: number,
  r: number,
  sNext: number,
  alpha: number,
  gamma: number,
  terminal = false,
): Mat {
  const target = r + (terminal ? 0 : gamma * Math.max(...Q[sNext]));
  return Q.map((row, i) =>
    i === s ? row.map((q, j) => (j === a ? q + alpha * (target - q) : q)) : row.slice(),
  );
}

/** Score function of a softmax policy π = softmax(θ): ∇_θ log π(a) = e_a − π. */
export function softmaxScore(theta: Vec, a: number): Vec {
  const pi = softmax(theta);
  return pi.map((p, i) => (i === a ? 1 : 0) - p);
}

/** Exact policy gradient for a bandit (REINFORCE in expectation): ∇J = Σ_a π(a) r(a) ∇ log π(a). */
export function banditPolicyGradient(theta: Vec, rewards: Vec): Vec {
  const pi = softmax(theta);
  const g = new Array<number>(theta.length).fill(0);
  pi.forEach((p, a) => {
    const score = softmaxScore(theta, a);
    for (let i = 0; i < g.length; i++) g[i] += p * rewards[a] * score[i];
  });
  return g;
}

/** PPO clipped surrogate (Schulman et al., 2017), per sample: min(ρ A, clip(ρ, 1−ε, 1+ε) A). */
export function ppoClipTerm(ratio: number, advantage: number, eps: number): number {
  const clipped = Math.min(Math.max(ratio, 1 - eps), 1 + eps);
  return Math.min(ratio * advantage, clipped * advantage);
}

export function ppoClipObjective(ratios: Vec, advantages: Vec, eps: number): number {
  return ratios.reduce((s, r, i) => s + ppoClipTerm(r, advantages[i], eps), 0) / ratios.length;
}

/** KL-penalized reward used in RLHF: R = r − β (log π(y|x) − log π_ref(y|x)). */
export function klPenalizedReward(reward: number, logProb: number, refLogProb: number, beta: number): number {
  return reward - beta * (logProb - refLogProb);
}

/** DPO loss (Rafailov et al., 2023): −log σ(β[(log π_w − log π_ref,w) − (log π_l − log π_ref,l)]). */
export function dpoLoss(
  logPw: number,
  logPl: number,
  refLogPw: number,
  refLogPl: number,
  beta: number,
): number {
  return -logSigmoid(beta * (logPw - refLogPw - (logPl - refLogPl)));
}
