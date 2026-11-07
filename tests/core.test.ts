/**
 * The engine must reproduce the oracle (NumPy / SciPy / statsmodels, plus PyTorch when installed).
 * If a test fails: do not loosen tolerances or edit fixtures. Find the bug.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import * as core from "../src/core";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const F: any = JSON.parse(readFileSync(resolve(__dirname, "fixtures/core.json"), "utf8"));

/** Restore -Infinity (encoded as null in JSON). */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const inf = (x: any): any => (x === null ? -Infinity : Array.isArray(x) ? x.map(inf) : x);

function close(got: number, want: number, rel = 1e-9, abs = 1e-12) {
  if (want === -Infinity) return expect(got).toBe(-Infinity);
  expect(Math.abs(got - want), `got ${got}, want ${want}`).toBeLessThanOrEqual(abs + rel * Math.abs(want));
}
const closeAll = (got: number[] | number[][], want: number[] | number[][], rel = 1e-9, abs = 1e-12) =>
  (got as number[]).flat().forEach((g, i) => close(g, (want as number[]).flat()[i], rel, abs));

describe("softmax family", () => {
  it("softmax, log-softmax, logsumexp, entropy across temperatures (incl. huge logits and -∞ masks)", () => {
    for (const c of F.softmax) {
      const z = inf(c.z);
      closeAll(core.softmax(z, c.tau), c.p);
      closeAll(core.logSoftmax(z, c.tau), inf(c.logp));
      close(core.logSumExp(z.map((x: number) => x / c.tau)), c.lse);
      close(core.entropy(core.softmax(z, c.tau)), c.entropy, 1e-9, 1e-12);
    }
  });
  it("sigmoid and log-sigmoid are stable at ±800", () => {
    for (const c of F.sigmoid) {
      close(core.sigmoid(c.x), c.sigmoid, 1e-12, 1e-300);
      close(core.logSigmoid(c.x), c.logSigmoid, 1e-12, 1e-300);
    }
  });
});

describe("attention lineage", () => {
  const A = F.attention;
  it("cosine similarity", () => close(core.cosineSimilarity(A.cosine.a, A.cosine.b), A.cosine.cos));
  it("Bahdanau additive attention", () => {
    const r = core.additiveAttention(A.additive.s, A.additive.H, A.additive.W, A.additive.U, A.additive.v);
    closeAll(r.scores, A.additive.scores);
    closeAll(r.weights, A.additive.weights);
    closeAll(r.context, A.additive.context);
  });
  it("Luong dot and general scores", () => {
    close(core.luongDotScore(A.luong.ht, A.luong.hs), A.luong.dot);
    close(core.luongGeneralScore(A.luong.ht, A.luong.Wa, A.luong.hs), A.luong.general);
  });
  it("scaled dot-product attention", () => {
    const r = core.scaledDotProductAttention(A.sdpa.Q, A.sdpa.K, A.sdpa.V);
    closeAll(core.attentionScores(A.sdpa.Q, A.sdpa.K), A.sdpa.scores);
    closeAll(r.scores, A.sdpa.scaled);
    closeAll(r.weights, A.sdpa.weights);
    closeAll(r.output, A.sdpa.output);
  });
  it("causal mask", () => {
    const mask = core.causalMask(5);
    closeAll(mask, inf(A.causal.mask));
    const r = core.scaledDotProductAttention(A.causal.Q, A.causal.K, A.causal.V, mask);
    closeAll(r.weights, A.causal.weights, 1e-9, 1e-15);
    closeAll(r.output, A.causal.output);
  });
  it("multi-head attention (oracle uses reshape; engine uses per-head slices)", () => {
    const m = A.multihead;
    const r = core.multiHeadAttention(m.X, m.X, m.X, {
      WQ: core.splitHeads(m.WQ, m.h),
      WK: core.splitHeads(m.WK, m.h),
      WV: core.splitHeads(m.WV, m.h),
      WO: m.WO,
    });
    closeAll(r.output, m.output);
  });
});

describe("why √d_k", () => {
  it("dot-product mean and variance from samples", () => {
    for (const c of F.scaling.moments) {
      const r = core.dotProductMoments(c.qs, c.ks);
      close(r.mean, c.mean, 1e-9, 1e-12);
      close(r.variance, c.variance);
    }
  });
  it("softmax saturation, raw vs scaled", () => {
    const s = F.scaling.saturation;
    for (const key of ["raw", "scaled"] as const) {
      const r = core.softmaxSaturation(s.q, s.K, key === "scaled");
      close(r.entropy, s[key].entropy, 1e-9, 1e-12);
      close(r.maxProb, s[key].maxProb);
    }
    expect(s.scaled.entropy).toBeGreaterThan(s.raw.entropy); // the whole point of scaling
  });
});

describe("agent lineage (RL → RLHF)", () => {
  const R = F.rl;
  it("discounted returns", () =>
    closeAll(core.discountedReturns(R.returns.rewards, R.returns.gamma), R.returns.G));
  it("Bellman policy evaluation matches an exact linear solve", () => {
    closeAll(
      core.policyEvaluation({ P: R.mdp.P, R: R.mdp.R }, R.mdp.pi, R.mdp.gamma),
      R.mdp.V_pi,
      1e-9,
      1e-10,
    );
  });
  it("value iteration: optimal values and greedy policy", () => {
    const r = core.valueIteration({ P: R.mdp.P, R: R.mdp.R }, R.mdp.gamma);
    closeAll(r.V, R.mdp.V_star, 1e-9, 1e-10);
    expect(r.policy).toEqual(R.mdp.policy);
  });
  it("Q-learning update (non-terminal and terminal)", () => {
    const q = R.qlearning;
    closeAll(core.qLearningUpdate(q.Q, q.s, q.a, q.r, q.s2, q.alpha, q.gamma), q.Q_next);
    closeAll(core.qLearningUpdate(q.Q, q.s, q.a, q.r, q.s2, q.alpha, q.gamma, true), q.Q_terminal);
  });
  it("policy gradient theorem: score-function gradient equals finite differences", () => {
    closeAll(
      core.banditPolicyGradient(R.policyGradient.theta, R.policyGradient.rewards),
      R.policyGradient.grad,
      1e-6,
      1e-9,
    );
  });
  it("PPO clipped objective", () => {
    const p = R.ppo;
    closeAll(
      p.ratios.map((r: number, i: number) => core.ppoClipTerm(r, p.advantages[i], p.eps)),
      p.terms,
    );
    close(core.ppoClipObjective(p.ratios, p.advantages, p.eps), p.objective);
  });
  it("KL-penalized reward and DPO loss (incl. extreme log-prob gaps)", () => {
    close(core.klPenalizedReward(R.kl.reward, R.kl.logp, R.kl.logref, R.kl.beta), R.kl.value);
    for (const d of R.dpo)
      close(core.dpoLoss(d.logPw, d.logPl, d.refLogPw, d.refLogPl, d.beta), d.loss, 1e-10, 1e-300);
  });
});

describe("evals lineage", () => {
  const E = F.evals;
  it("Wilson interval (statsmodels)", () => {
    for (const c of E.wilson) closeAll(core.wilsonInterval(c.x, c.n), c.ci, 1e-9, 1e-12);
  });
  it("exact McNemar (statsmodels)", () => {
    for (const c of E.mcnemar) close(core.mcnemarExact(c.b, c.c), c.p, 1e-9, 1e-15);
  });
  it("pass@k (Chen et al.) and pass^k (τ-bench) vs exact binomial coefficients", () => {
    for (const c of E.passk) {
      close(core.passAtK(c.n, c.c, c.k), c.passAtK, 1e-10, 1e-14);
      close(core.passHatK(c.n, c.c, c.k), c.passHatK, 1e-10, 1e-300);
    }
  });
  it("Bradley–Terry MLE: MM algorithm matches BFGS on the likelihood", () => {
    const beta = core.bradleyTerryFit(E.bradleyTerry.wins);
    closeAll(beta, E.bradleyTerry.beta, 1e-5, 1e-6);
    beta.forEach((bi, i) =>
      beta.forEach((bj, j) => close(core.bradleyTerryProb(bi, bj), E.bradleyTerry.probs[i][j], 1e-5, 1e-6)),
    );
  });
  it("Elo expected score and update", () => {
    for (const c of E.elo) {
      close(core.eloExpected(c.ra, c.rb), c.expected);
      close(core.eloUpdate(c.ra, c.rb, c.s, c.k), c.updated);
    }
  });
});

describe("normal distribution", () => {
  it("cdf and quantile vs scipy", () => {
    F.normal.z.forEach((z: number, i: number) => close(core.normalCdf(z), F.normal.cdf[i], 1e-9, 1e-15));
    F.normal.p.forEach((p: number, i: number) => close(core.normalQuantile(p), F.normal.ppf[i], 1e-9, 1e-9));
  });
});

describe("invariants", () => {
  it("softmax is shift-invariant and sums to 1", () => {
    const z = [0.2, -1.4, 3.3, 0.9];
    const p = core.softmax(z);
    close(
      p.reduce((s, x) => s + x, 0),
      1,
      1e-12,
      1e-12,
    );
    closeAll(core.softmax(z.map((x) => x + 500)), p, 1e-12, 1e-15);
  });
  it("temperature → 0 approaches argmax; → ∞ approaches uniform", () => {
    const z = [1, 3, 2];
    expect(core.softmax(z, 0.01)[1]).toBeGreaterThan(0.999);
    core.softmax(z, 1e6).forEach((p) => close(p, 1 / 3, 1e-5, 1e-6));
  });
  it("Bradley–Terry with two players is exactly a sigmoid of the logit gap (softmax over 2)", () => {
    const [a, b] = [0.7, -0.4];
    closeAll([core.bradleyTerryProb(a, b), core.bradleyTerryProb(b, a)], core.softmax([a, b]), 1e-12, 1e-15);
  });
  it("pass^k ≤ pass@1 ≤ pass@k", () => {
    const [n, c, k] = [40, 22, 5];
    expect(core.passHatK(n, c, k)).toBeLessThanOrEqual(c / n);
    expect(core.passAtK(n, c, k)).toBeGreaterThanOrEqual(c / n);
  });
  it("rejects bad input", () => {
    expect(() => core.softmax([1, 2], 0)).toThrow();
    expect(() => core.logSumExp([-Infinity, -Infinity])).toThrow();
    expect(() => core.matmul([[1, 2]], [[1, 2]])).toThrow();
    expect(() => core.passAtK(5, 6, 1)).toThrow();
  });
});
