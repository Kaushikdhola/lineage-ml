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
