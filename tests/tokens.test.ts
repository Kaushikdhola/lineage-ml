import { describe, expect, it } from "vitest";
import { getFormula } from "../src/atlas/graph";
import { tokenKeys } from "../src/atlas/graph";
import { diffTokens, tokenIds } from "../src/math/tokens";

describe("token identity and diff", () => {
  it("numbers repeated keys by occurrence", () => {
    expect(tokenIds(["q", "k", "q"])).toEqual(["q#0", "k#0", "q#1"]);
  });

  it("scores → scaled scores: everything persists, √d_k enters", () => {
    const d = diffTokens(
      tokenKeys(getFormula("attention-scores").tex),
      tokenKeys(getFormula("scaled-scores").tex),
    );
    expect(d.persist).toEqual(expect.arrayContaining(["S#0", "q#0", "k#0", "T#0"]));
    expect(d.enter).toEqual(["sqrtdk#0"]);
    expect(d.exit).toEqual([]);
  });

  it("softmax → temperature: τ is the only new term", () => {
    const d = diffTokens(
      tokenKeys(getFormula("softmax").tex),
      tokenKeys(getFormula("softmax-temperature").tex),
    );
    expect(d.enter).toEqual(["tau#0"]);
  });
});
