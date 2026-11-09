/**
 * The GRPO weights measured from real TRL runs (public/xray/data.json, exported by xray/) against the
 * closed forms in src/core/aggregation.ts. At step 1 every ratio is 1, so clipping is inactive and the
 * on-policy formulas apply exactly. The export stores the weight on each token's cross-entropy,
 * w = −∂L/∂log π; aggregation.ts returns ∂L/∂log π.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { groupWeights, type GroupObjective } from "../src/core/aggregation";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const D: any = JSON.parse(readFileSync(resolve(__dirname, "../public/xray/data.json"), "utf8"));

const OBJECTIVE: Record<string, GroupObjective> = { dapo: "dapo", grpo: "grpo", dr_grpo: "dr-grpo" };

describe("TRL GRPOTrainer, measured, against aggregation.ts", () => {
  const runs = D.trl.grpo as {
    id: string;
    config: { loss_type: string; scale_rewards: string; max_completion_length: number };
    lengths: number[];
    rewards: number[];
    steps: { step: number; weights: number[][]; ratios: number[][] }[];
  }[];

  it("covers the default loss type and the two alternatives", () => {
    expect(runs.find((r) => r.id === "grpo-default")!.config.loss_type).toBe("dapo");
    expect(runs.map((r) => r.config.loss_type).sort()).toEqual(["dapo", "dr_grpo", "grpo"]);
  });

  for (const id of ["grpo-default", "grpo-grpo", "grpo-dr_grpo"]) {
    it(`${id}: step-1 weights equal groupWeights`, () => {
      const run = runs.find((r) => r.id === id)!;
      const objective = OBJECTIVE[run.config.loss_type];
      // aggregation.ts's Dr. GRPO drops the std normalisation; the run must have done the same
      if (objective === "dr-grpo") expect(run.config.scale_rewards).toBe("none");
      const s1 = run.steps.find((s) => s.step === 1)!;
      expect(s1.ratios.flat().every((r) => Math.abs(r - 1) < 1e-12)).toBe(true);
      const want = groupWeights(run.lengths, run.rewards, objective, {
        maxLength: run.config.max_completion_length,
      }).weights;
      const scale = Math.max(...want.flat().map(Math.abs));
      s1.weights.forEach((row, i) =>
        row.forEach((w, t) => expect(Math.abs(-w - want[i][t])).toBeLessThan(1e-5 * scale)),
      );
    });
  }
});
