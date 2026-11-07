/**
 * Evidence for the √d_k ablation. The UI states a claim and an answer to the guess question;
 * these tests are what make them true. No claim without evidence.
 */
import { describe, expect, it } from "vitest";
import { getFormula } from "../src/atlas/graph";
import { trainRetrieval } from "../src/core/retrieval";
import {
  runSqrtDkRaw,
  SQRT_DK_BASE,
  SQRT_DK_LR,
  SQRT_DK_LR_GRID,
  SQRT_DK_TUNING_SEEDS,
  sqrtDkConfig,
  sqrtDkReport,
} from "../src/sim/sqrtDk";

const tuningLoss = (scale: number, lr: number) =>
  SQRT_DK_TUNING_SEEDS.map((s) => trainRetrieval(s, { ...SQRT_DK_BASE, scale, lr })).reduce(
    (acc, r) => ({
      loss: acc.loss + r.finalLoss / SQRT_DK_TUNING_SEEDS.length,
      diverged: acc.diverged || r.diverged,
    }),
    { loss: 0, diverged: false },
  );

describe("sqrt-dk simulator", () => {
  it("is deterministic per seed", () => {
    const a = trainRetrieval(5, sqrtDkConfig(true));
    const b = trainRetrieval(5, sqrtDkConfig(true));
    expect(a).toEqual(b);
  });

  it("claim: across 20 seeds, deleting √d_k saturates the softmax at init and training stalls", async () => {
    const raw = await runSqrtDkRaw({ seeds: 20 });
    for (const r of [...raw.intact, ...raw.ablated]) {
      expect(r.diverged).toBe(false);
      expect(r.loss.every(Number.isFinite)).toBe(true);
    }
    const rep = sqrtDkReport(raw);
    const [final, entropy, slope] = rep.stats;
    // training stalls: intervals separated, and an order of magnitude apart
    expect(final.ablated.lo).toBeGreaterThan(final.intact.hi);
    expect(final.ablated.mean).toBeGreaterThan(10 * final.intact.mean);
    // the intact model actually solves the task, so the comparison is meaningful
    expect(final.intact.mean).toBeLessThan(0.1 * rep.chart.intact.mean[0]);
    // saturated at step 0: low entropy (uniform would be ln 16 ≈ 2.77) and a flat softmax
    expect(entropy.ablated.hi).toBeLessThan(entropy.intact.lo);
    expect(entropy.ablated.mean).toBeLessThan(1);
    expect(slope.ablated.hi).toBeLessThan(slope.intact.lo);
    expect(rep.seeds).toBe(20);
  });

  it("fair: each version's learning rate is its own best on the grid, and the ablated best still loses", () => {
    const intact = SQRT_DK_LR_GRID.map((lr) => tuningLoss(1 / Math.sqrt(SQRT_DK_BASE.dk), lr));
    const ablated = SQRT_DK_LR_GRID.map((lr) => tuningLoss(1, lr));
    const argmin = (xs: { loss: number }[]) => xs.reduce((b, x, i) => (x.loss < xs[b].loss ? i : b), 0);
    expect(SQRT_DK_LR_GRID[argmin(intact)]).toBe(SQRT_DK_LR.intact);
    expect(SQRT_DK_LR_GRID[argmin(ablated)]).toBe(SQRT_DK_LR.ablated);
    // the best learning rate is inside the grid, not on its edge
    expect(argmin(ablated)).toBeGreaterThan(0);
    expect(argmin(ablated)).toBeLessThan(SQRT_DK_LR_GRID.length - 1);
    expect(ablated[argmin(ablated)].loss).toBeGreaterThan(10 * intact[argmin(intact)].loss);
    // "training explodes" is wrong: no learning rate on the grid diverges without √d_k
    expect(ablated.some((r) => r.diverged)).toBe(false);
  });

  it("the guess question's answer is the choice the evidence supports", () => {
    const ab = getFormula("scaled-dot-product-attention").ablations!.find((a) => a.token === "sqrtdk")!;
    expect(ab.simulator).toBe("sqrt-dk");
    expect(ab.choices[ab.answer]).toMatch(/stalls/);
  });
});
