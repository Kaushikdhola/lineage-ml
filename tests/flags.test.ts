import { describe, expect, it } from "vitest";
import { readFlags } from "../src/ui/flags";

describe("feature flags", () => {
  it("ablation mode is off unless ?ablation=1", () => {
    expect(readFlags("").ablation).toBe(false);
    expect(readFlags("?f=scaled-dot-product-attention").ablation).toBe(false);
    expect(readFlags("?f=x&ablate=sqrtdk").ablation).toBe(false);
    expect(readFlags("?ablation=0").ablation).toBe(false);
    expect(readFlags("?ablation=1").ablation).toBe(true);
    expect(readFlags("?f=x&ablation=1&ablate=sqrtdk").ablation).toBe(true);
  });
});
