import { describe, expect, it } from "vitest";
import * as core from "../src/core";
import { EDGES, FORMULAS } from "../src/atlas/formulas";
import { ancestors, lineagePath, tokenKeys, topologicalOrder, validateAtlas } from "../src/atlas/graph";
import { renderTexToString } from "../src/math/katex";

describe("atlas", () => {
  it("has no structural issues (ids, tokens ↔ notation, compute functions, edges, acyclic)", () => {
    expect(validateAtlas(core as unknown as Record<string, unknown>)).toEqual([]);
  });

  it("every formula renders with KaTeX and exposes its tokens as data-tok spans", () => {
    for (const f of FORMULAS) {
      const html = renderTexToString(f.tex);
      for (const key of new Set(tokenKeys(f.tex)))
        expect(html, `${f.id}: ${key}`).toContain(`data-tok="${key}"`);
    }
  });

  it("covers the four families", () => {
    const fams = new Set(FORMULAS.map((f) => f.family));
    expect([...fams].sort()).toEqual(["agents", "attention", "evals", "foundations"]);
  });

  it("softmax is the hub that connects families", () => {
    const fams = new Set(
      EDGES.filter((e) => e.from === "softmax").map((e) => FORMULAS.find((f) => f.id === e.to)!.family),
    );
    expect(fams.size).toBeGreaterThanOrEqual(4);
  });

  it("lineage paths read as stories", () => {
    expect(lineagePath("multi-head-attention")).toEqual([
      "softmax",
      "bahdanau-attention",
      "luong-attention",
      "attention-scores",
      "scaled-scores",
      "scaled-dot-product-attention",
      "multi-head-attention",
    ]);
    expect(lineagePath("dpo")[0]).toBe("discounted-return");
    expect(ancestors("dpo")).toContain("bradley-terry");
    expect(topologicalOrder()).toHaveLength(FORMULAS.length);
  });
});
