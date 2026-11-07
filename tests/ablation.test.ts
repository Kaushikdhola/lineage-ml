import { describe, expect, it } from "vitest";
import { FORMULAS } from "../src/atlas/formulas";
import { tokenKeys } from "../src/atlas/graph";
import { ablateTex } from "../src/math/ablate";
import { renderTexToString } from "../src/math/katex";
import { SIMULATORS } from "../src/sim";

describe("ablation mode", () => {
  const ablations = FORMULAS.flatMap((f) => (f.ablations ?? []).map((a) => ({ f, a })));

  it("has at least one ablation and every one names a registered simulator", () => {
    expect(ablations.length).toBeGreaterThan(0);
    for (const { f, a } of ablations)
      expect(SIMULATORS[a.simulator], `${f.id}: ${a.simulator}`).toBeDefined();
  });

  it("every ablated formula still renders, without the deleted term and with every other term", () => {
    for (const { f, a } of ablations) {
      const tex = ablateTex(f.tex, a.token);
      const html = renderTexToString(tex);
      expect(html).not.toContain(`data-tok="${a.token}"`);
      for (const k of new Set(tokenKeys(f.tex)))
        if (k !== a.token) expect(html, `${f.id}: ${k}`).toContain(`data-tok="${k}"`);
    }
  });

  it("deleting √d_k leaves softmax(Q Kᵀ) V", () => {
    const f = FORMULAS.find((x) => x.id === "scaled-dot-product-attention")!;
    expect(ablateTex(f.tex, "sqrtdk")).toBe(
      String.raw`\mathrm{Attention}(\tok{q}{Q}, \tok{k}{K}, \tok{v}{V}) = \tok{softmax}{\mathrm{softmax}}\!\left({\tok{q}{Q}\, \tok{k}{K}^{\tok{T}{\top}}}\right) \tok{v}{V}`,
    );
  });

  it("tidies fractions and stranded operators", () => {
    expect(ablateTex(String.raw`\dfrac{a}{\tok{d}{d}}`, "d")).toBe("{a}");
    expect(ablateTex(String.raw`\frac{\tok{n}{n}}{b}`, "n")).toBe(String.raw`\frac{1}{b}`);
    expect(ablateTex(String.raw`\left(x + \tok{M}{M}\right)`, "M")).toBe(String.raw`\left(x\right)`);
    expect(ablateTex(String.raw`\left(\tok{a}{a} + y\right)`, "a")).toBe(String.raw`\left( y\right)`);
    // escaped braces inside the deleted group do not confuse the matcher
    expect(ablateTex(String.raw`\tok{a}{\{x\}}\, y`, "a")).toBe(String.raw`\, y`);
    expect(() => ablateTex("x", "nope")).toThrow();
  });
});
