import { describe, expect, it } from "vitest";
import { EDGES, FORMULAS } from "../src/atlas/formulas";
import { pathBetween } from "../src/atlas/graph";
import { computeLayout, DEFAULT_LAYOUT, generations, ringRadii } from "../src/atlas/layout";
import type { Edge, Family, Formula } from "../src/atlas/schema";
import { mulberry32 } from "../src/core/random";

const dist2d = (s: { x: number; z: number }) => Math.hypot(s.x, s.z);

/** A synthetic atlas of `n` formulas across all seven families, for scale tests. */
function synthetic(n: number, seed = 7): { formulas: Formula[]; edges: Edge[] } {
  const fams: Family[] = [
    "foundations",
    "optimization",
    "architecture",
    "attention",
    "generative",
    "agents",
    "evals",
  ];
  const rng = mulberry32(seed);
  const formulas: Formula[] = [
    {
      id: "hub",
      name: "hub",
      family: "foundations",
      tex: "",
      idea: "",
      symbols: {},
      compute: null,
      hub: true,
    },
  ];
  const edges: Edge[] = [];
  for (let i = 1; i < n; i++) {
    const family = fams[Math.floor(rng() * fams.length)];
    formulas.push({ id: `f${i}`, name: `f${i}`, family, tex: "", idea: "", symbols: {}, compute: null });
    // link to 0-2 earlier formulas, keeping the graph acyclic; prefer the same family
    const k = Math.floor(rng() * 3);
    const parents = new Set<string>();
    for (let j = 0; j < k; j++) {
      const sameFam = formulas.filter((f, idx) => idx < i && f.family === family);
      const pool = rng() < 0.75 && sameFam.length ? sameFam : formulas.slice(0, i);
      parents.add(pool[Math.floor(rng() * pool.length)].id);
    }
    for (const p of parents) edges.push({ from: p, to: `f${i}`, op: "", kind: "conceptual" });
  }
  return { formulas, edges };
}

function checkInvariants(formulas: Formula[], edges: Edge[]) {
  const layout = computeLayout(formulas, edges);
  const byId = new Map(layout.stars.map((s) => [s.id, s]));
  const gen = generations(formulas, edges);

  // hub at the centre
  const hub = layout.stars.find((s) => s.hub)!;
  expect([hub.x, hub.y, hub.z]).toEqual([0, 0, 0]);

  // generation = longest path from a root; every child sits strictly outside all its parents
  for (const e of edges) {
    expect(gen.get(e.to)!).toBeGreaterThan(gen.get(e.from)!);
    expect(dist2d(byId.get(e.to)!), `${e.from} -> ${e.to}`).toBeGreaterThan(dist2d(byId.get(e.from)!));
  }

  // rings strictly increase outward
  for (let g = 1; g < layout.rings.length; g++)
    expect(layout.rings[g].radius).toBeGreaterThan(layout.rings[g - 1].radius);

  // every star sits inside its family sector
  for (const s of layout.stars) {
    if (s.hub) continue;
    const sec = layout.sectors.find((x) => x.family === s.family)!;
    expect(s.angle).toBeGreaterThanOrEqual(sec.start - 1e-9);
    expect(s.angle).toBeLessThanOrEqual(sec.end + 1e-9);
  }

  // stars on the same ring and family keep the minimum arc spacing
  for (const a of layout.stars)
    for (const b of layout.stars) {
      if (a === b || a.hub || b.hub || a.family !== b.family || a.generation !== b.generation) continue;
      const r = layout.rings[a.generation].radius;
      const arc = Math.abs(Math.atan2(Math.sin(a.angle - b.angle), Math.cos(a.angle - b.angle))) * r;
      expect(arc, `${a.id} vs ${b.id}`).toBeGreaterThanOrEqual(DEFAULT_LAYOUT.spacing - 1e-9);
    }

  // the disc has thickness, but stays a disc
  const ys = layout.stars.map((s) => Math.abs(s.y));
  expect(Math.max(...ys)).toBeGreaterThan(0.1);
  expect(Math.max(...ys)).toBeLessThan(layout.outerRadius * 0.25);
  return layout;
}

describe("sky layout", () => {
  it("is deterministic for a seed and changes with the seed", () => {
    const a = computeLayout(FORMULAS, EDGES);
    const b = computeLayout(FORMULAS, EDGES);
    expect(a).toEqual(b);
    const c = computeLayout(FORMULAS, EDGES, { seed: 1 });
    expect(c.stars.map((s) => s.y)).not.toEqual(a.stars.map((s) => s.y));
  });

  it("holds its invariants on the real atlas", () => {
    const layout = checkInvariants(FORMULAS, EDGES);
    expect(layout.stars.find((s) => s.id === "softmax")!.hub).toBe(true);
    // multi-head attention derives through six steps from a root
    expect(layout.stars.find((s) => s.id === "multi-head-attention")!.generation).toBe(6);
  });

  it("holds its invariants at ~130 formulas", () => {
    const { formulas, edges } = synthetic(130);
    const layout = checkInvariants(formulas, edges);
    expect(layout.sectors).toHaveLength(7);
  });

  it("ring radius grows with how many formulas the ring holds", () => {
    const o = DEFAULT_LAYOUT;
    const base = ringRadii([2, 4, 4], [0, 0, 0], o);
    const fuller = ringRadii([2, 9, 4], [0, 0, 0], o);
    expect(fuller[1]).toBeGreaterThan(base[1]);
    expect(fuller[2] - fuller[1]).toBeCloseTo(base[2] - base[1]); // only the fuller ring and beyond move out
    // a crowded family arc pushes its ring out far enough to keep the spacing
    const crowded = ringRadii([2, 4], [0, 12 / 0.5], o);
    expect(crowded[1]).toBeGreaterThanOrEqual((12 * o.spacing) / 0.5);
  });

  it("finds the tour chain from dot product to multi-head attention", () => {
    expect(pathBetween("dot-product", "multi-head-attention")).toEqual([
      "dot-product",
      "luong-attention",
      "attention-scores",
      "scaled-scores",
      "scaled-dot-product-attention",
      "multi-head-attention",
    ]);
    expect(pathBetween("multi-head-attention", "dot-product")).toBeNull();
  });
});
