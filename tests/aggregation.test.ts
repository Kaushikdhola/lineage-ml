/** Closed-form token weights must match numerical derivatives of the explicit losses (python oracle). */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import * as agg from "../src/core/aggregation";

const F = JSON.parse(readFileSync(resolve(__dirname, "fixtures/aggregation.json"), "utf8"));
const closeAll = (got: number[][], want: number[][], tol: number) =>
  got.flat().forEach((g, i) => expect(Math.abs(g - want.flat()[i]), `index ${i}`).toBeLessThan(tol));

describe("language-model aggregation", () => {
  const { lengths, micro } = F.lm;
  it("token mean", () => closeAll(agg.tokenMean(lengths), F.lm.token_mean, 1e-6));
  it("sequence mean", () => closeAll(agg.sequenceMean(lengths), F.lm.sequence_mean, 1e-6));
  it("mean of micro-batch means", () =>
    closeAll(agg.meanOfMicrobatchMeans(lengths, micro), F.lm.mean_of_means, 1e-6));
  it("first micro-batch only", () =>
    closeAll(agg.firstMicrobatchOnly(lengths, micro), F.lm.first_only, 1e-6));
});

describe("group objectives (on-policy token coefficients)", () => {
  for (const obj of ["grpo", "dr-grpo", "dapo"] as const) {
    it(obj, () => {
      const r = agg.groupWeights(F.group.lengths, F.group.rewards, obj);
      r.advantages.forEach((a, i) => expect(Math.abs(a - F.group[obj].advantages[i])).toBeLessThan(1e-12));
      closeAll(r.weights, F.group[obj].weights, 1e-8);
    });
  }
  it("GRPO's per-token weight shrinks with length; Dr. GRPO's does not", () => {
    const L = [10, 100];
    const g = agg.groupWeights(L, [0, 0.0001], "grpo");
    const d = agg.groupWeights(L, [0, 0.0001], "dr-grpo");
    const ratio = (w: agg.GroupWeights) =>
      Math.abs(w.weights[1][0] / w.advantages[1]) / Math.abs(w.weights[0][0] / w.advantages[0]);
    expect(ratio(g)).toBeCloseTo(0.1, 10);
    expect(ratio(d)).toBeCloseTo(1, 10);
  });
});

/**
 * Hand-derived from the paper equations, independent of groupWeights' code. One group of G = 2 answers,
 * |o_1| = 2 and |o_2| = 3 tokens, rewards 1 and 0, so r̄ = 1/2, r − r̄ = ±1/2 and the sample standard
 * deviation (ddof 1) is √((¼ + ¼)/1) = √½. Weights are ∂L/∂log π at ratio 1.
 *   GRPO (DeepSeekMath):  Â = ±½ / (√½ + ε),  L = −1/G Σ_i 1/|o_i| Σ_t Â_i ρ    → −Â_i / (G·|o_i|)
 *   DAPO:                 same Â,              L = −1/Σ_j|o_j| Σ_{i,t} Â_i ρ   → −Â_i / 5
 *   Dr. GRPO:             Â = ±½ (no std),     L = −1/(G·L_max) Σ_{i,t} Â_i ρ  → −Â_i / (2·L_max)
 */
describe("group objectives, hand-derived from the papers", () => {
  const A = 0.5 / (Math.SQRT1_2 + 1e-4); // the eps = 1e-4 convention shared with TRL
  const L = [2, 3];
  const R = [1, 0];
  const close = (got: number[][], want: number[][]) =>
    got.forEach((row, i) => row.forEach((v, t) => expect(v).toBeCloseTo(want[i][t], 14)));

  it("GRPO: each answer averaged over its own tokens", () =>
    close(agg.groupWeights(L, R, "grpo").weights, [
      [-A / 4, -A / 4],
      [A / 6, A / 6, A / 6],
    ]));
  it("DAPO: every token over the group's total tokens", () =>
    close(agg.groupWeights(L, R, "dapo").weights, [
      [-A / 5, -A / 5],
      [A / 5, A / 5, A / 5],
    ]));
  it("Dr. GRPO: centred rewards over the constant G·L_max", () =>
    close(agg.groupWeights(L, R, "dr-grpo", { maxLength: 8 }).weights, [
      [-0.5 / 16, -0.5 / 16],
      [0.5 / 16, 0.5 / 16, 0.5 / 16],
    ]));
  it("Dr. GRPO's normaliser does not depend on the batch; DAPO's does", () => {
    const other = [2, 7]; // the second answer gets longer
    const dr = (lengths: number[]) => agg.groupWeights(lengths, R, "dr-grpo", { maxLength: 8 }).weights[0][0];
    const dapo = (lengths: number[]) => agg.groupWeights(lengths, R, "dapo").weights[0][0];
    expect(dr(other)).toBe(dr(L));
    expect(dapo(other)).not.toBeCloseTo(dapo(L), 6);
  });
});
