/**
 * The sky's layout: a lineage disc.
 *
 * - The hub sits at the centre.
 * - Every other formula sits on the ring of its generation, where generation = the longest path
 *   from a root. A formula is therefore always on a larger ring than everything it derives from.
 * - Each family owns an angular sector, sized by its widest generation.
 * - Ring radii come from a capacity rule: every ring steps out by a base gap plus a little per
 *   formula it holds, and far enough that the most crowded family arc on it keeps a minimum
 *   spacing between stars. This keeps the disc legible at ~130 formulas, not only at 26.
 * - Inside those constraints, a seeded force relaxation spreads stars along their arcs (springs
 *   toward parents and children, repulsion between neighbours). Same input, same sky.
 * - A seeded vertical scatter gives the disc thickness, so camera drift has parallax.
 *
 * Pure data: no DOM, no three.js. Positions are layout, not math, so they live here and not in
 * src/core.
 */
import { mulberry32 } from "../core/random";
import { FAMILY_LABEL, type Edge, type Family, type Formula } from "./schema";

export interface LayoutOptions {
  seed: number;
  /** Minimum arc distance between neighbouring stars on a ring (world units). */
  spacing: number;
  /** Radius of the innermost ring before capacity is considered. */
  innerRadius: number;
  /** Base radial step between consecutive rings. */
  ringStep: number;
  /** Extra radial step per formula on a ring. */
  ringStepPerFormula: number;
  /** Angular gap between family sectors, radians. */
  sectorGap: number;
  /** Standard deviation of the vertical scatter at the rim. */
  thickness: number;
  /** Force-relaxation iterations. */
  iterations: number;
}

export const DEFAULT_LAYOUT: LayoutOptions = {
  seed: 20261107,
  spacing: 2.2,
  innerRadius: 3.4,
  ringStep: 2.6,
  ringStepPerFormula: 0.12,
  sectorGap: 0.09,
  thickness: 0.55,
  iterations: 240,
};

export interface StarLayout {
  id: string;
  family: Family;
  /** Longest path from a root. The hub is generation 0 at the centre. */
  generation: number;
  hub: boolean;
  angle: number;
  radius: number;
  x: number;
  y: number;
  z: number;
}

export interface RingLayout {
  generation: number;
  radius: number;
  count: number;
}

export interface SectorLayout {
  family: Family;
  start: number;
  end: number;
  /** Radius just outside this family's outermost star: where its name is lettered. */
  edgeRadius: number;
}

export interface SkyLayout {
  stars: StarLayout[];
  rings: RingLayout[];
  sectors: SectorLayout[];
  outerRadius: number;
}

/** Longest path from any root, for every formula. Throws on cycles (via the order). */
export function generations(formulas: Formula[], edges: Edge[]): Map<string, number> {
  const parents = new Map<string, string[]>(formulas.map((f) => [f.id, []]));
  const children = new Map<string, string[]>(formulas.map((f) => [f.id, []]));
  const indeg = new Map(formulas.map((f) => [f.id, 0]));
  for (const e of edges) {
    parents.get(e.to)!.push(e.from);
    children.get(e.from)!.push(e.to);
    indeg.set(e.to, indeg.get(e.to)! + 1);
  }
  const gen = new Map<string, number>();
  const ready = formulas.filter((f) => indeg.get(f.id) === 0).map((f) => f.id);
  for (const id of ready) gen.set(id, 0);
  let seen = 0;
  while (ready.length) {
    const id = ready.shift()!;
    seen++;
    for (const c of children.get(id)!) {
      gen.set(c, Math.max(gen.get(c) ?? 0, gen.get(id)! + 1));
      indeg.set(c, indeg.get(c)! - 1);
      if (indeg.get(c) === 0) ready.push(c);
    }
  }
  if (seen !== formulas.length) throw new Error("lineage graph has a cycle");
  return gen;
}

/**
 * Ring radii from ring occupancy. `counts[g]` is the number of formulas on ring g, and
 * `crowding[g]` is the largest (stars on that ring in one family) / (that family's usable arc).
 */
export function ringRadii(counts: number[], crowding: number[], o: LayoutOptions): number[] {
  const radii: number[] = [];
  for (let g = 0; g < counts.length; g++) {
    const prev = g === 0 ? o.innerRadius - o.ringStep : radii[g - 1];
    const stepped = prev + o.ringStep + o.ringStepPerFormula * counts[g];
    const capacity = crowding[g] * o.spacing;
    radii.push(Math.max(stepped, capacity));
  }
  return radii;
}

const wrapPi = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));

export function computeLayout(
  formulas: Formula[],
  edges: Edge[],
  options: Partial<LayoutOptions> = {},
): SkyLayout {
  const o = { ...DEFAULT_LAYOUT, ...options };
  const rng = mulberry32(o.seed);
  const gen = generations(formulas, edges);
  const hub = formulas.find((f) => f.hub);
  const others = formulas.filter((f) => f !== hub);

  const familyOrder = (Object.keys(FAMILY_LABEL) as Family[]).filter((fam) =>
    others.some((f) => f.family === fam),
  );
  const maxGen = Math.max(0, ...others.map((f) => gen.get(f.id)!));
  const G = maxGen + 1;

  // occupancy per (family, generation)
  const occ = new Map<Family, number[]>(familyOrder.map((fam) => [fam, Array(G).fill(0)]));
  for (const f of others) occ.get(f.family)![gen.get(f.id)!]++;

  // sectors sized by each family's widest generation
  const weights = familyOrder.map((fam) => Math.max(...occ.get(fam)!) + 0.6);
  const total = weights.reduce((a, b) => a + b, 0);
  const usable = 2 * Math.PI - o.sectorGap * familyOrder.length;
  const sectors: { family: Family; start: number; end: number }[] = [];
  let cursor = -Math.PI / 2 + o.sectorGap / 2;
  familyOrder.forEach((fam, i) => {
    const span = (usable * weights[i]) / total;
    sectors.push({ family: fam, start: cursor, end: cursor + span });
    cursor += span + o.sectorGap;
  });
  const sectorOf = new Map(sectors.map((s) => [s.family, s]));

  // ring radii from counts and crowding
  const counts = Array.from({ length: G }, (_, g) => familyOrder.reduce((a, fam) => a + occ.get(fam)![g], 0));
  const crowding = Array.from({ length: G }, (_, g) =>
    Math.max(
      0,
      ...sectors.map((s) => {
        const n = occ.get(s.family)![g];
        // n stars need n gaps of arc: n-1 between them plus half a gap at each sector edge
        return n / (s.end - s.start);
      }),
    ),
  );
  const radii = ringRadii(counts, crowding, o);

  // adjacency for the force relaxation
  const nbrs = new Map<string, string[]>(formulas.map((f) => [f.id, []]));
  for (const e of edges) {
    nbrs.get(e.from)!.push(e.to);
    nbrs.get(e.to)!.push(e.from);
  }
  const parentsOf = new Map<string, string[]>(formulas.map((f) => [f.id, []]));
  for (const e of edges) parentsOf.get(e.to)!.push(e.from);

  // initial angles: generation by generation, ordered by the mean angle of placed parents
  const angle = new Map<string, number>();
  const byId = new Map(formulas.map((f) => [f.id, f]));
  for (let g = 0; g < G; g++) {
    for (const s of sectors) {
      const ring = others.filter((f) => f.family === s.family && gen.get(f.id) === g);
      if (!ring.length) continue;
      const mid = (s.start + s.end) / 2;
      const key = (f: Formula) => {
        const ps = parentsOf.get(f.id)!.filter((p) => angle.has(p) && byId.get(p) !== hub);
        if (!ps.length) return mid;
        const sx = ps.reduce((a, p) => a + Math.cos(angle.get(p)!), 0);
        const sy = ps.reduce((a, p) => a + Math.sin(angle.get(p)!), 0);
        return mid + wrapPi(Math.atan2(sy, sx) - mid);
      };
      const ordered = ring
        .map((f, i) => ({ f, k: key(f), i }))
        .sort((a, b) => a.k - b.k || a.i - b.i)
        .map((x) => x.f);
      const span = s.end - s.start;
      ordered.forEach((f, i) => {
        const jitter = (rng() - 0.5) * 0.25 * (span / ordered.length);
        angle.set(f.id, s.start + ((i + 0.5) / ordered.length) * span + jitter);
      });
    }
  }

  // force relaxation along the arcs (angles only; radius is fixed by generation)
  const minArc = (r: number) => o.spacing / Math.max(r, 1e-6);
  for (let it = 0; it < o.iterations; it++) {
    const cool = 1 - it / o.iterations;
    const delta = new Map<string, number>();
    for (const f of others) {
      const g = gen.get(f.id)!;
      const r = radii[g];
      const a = angle.get(f.id)!;
      let d = 0;
      // springs: pull toward linked stars in the same sector
      for (const n of nbrs.get(f.id)!) {
        const nf = byId.get(n)!;
        if (nf === hub || nf.family !== f.family) continue;
        d += 0.04 * wrapPi(angle.get(n)! - a);
      }
      // repulsion: neighbours on the same ring and sector closer than the minimum arc
      for (const h of others) {
        if (h === f || h.family !== f.family || gen.get(h.id) !== g) continue;
        const diff = wrapPi(a - angle.get(h.id)!);
        const gap = minArc(r) - Math.abs(diff);
        if (gap > 0) d += 0.5 * gap * Math.sign(diff || (f.id < h.id ? -1 : 1));
      }
      delta.set(f.id, d * cool);
    }
    for (const f of others) {
      const s = sectorOf.get(f.family)!;
      const r = radii[gen.get(f.id)!];
      const margin = Math.min(minArc(r) / 2, (s.end - s.start) / 2);
      const a = angle.get(f.id)! + delta.get(f.id)!;
      angle.set(f.id, Math.min(s.end - margin, Math.max(s.start + margin, a)));
    }
  }

  // final spacing pass: the relaxation is soft, this is the guarantee. Sweep each arc in angle
  // order enforcing the minimum gap, then pull back from the sector end; capacity guarantees fit.
  for (let g = 0; g < G; g++) {
    const step = minArc(radii[g]);
    for (const s of sectors) {
      const arc = others
        .filter((f) => f.family === s.family && gen.get(f.id) === g)
        .sort((a, b) => angle.get(a.id)! - angle.get(b.id)! || (a.id < b.id ? -1 : 1));
      if (!arc.length) continue;
      const lo = s.start + Math.min(step / 2, (s.end - s.start) / 2);
      const hi = s.end - Math.min(step / 2, (s.end - s.start) / 2);
      let prev = -Infinity;
      for (const f of arc) {
        const a = Math.max(angle.get(f.id)!, lo, prev + step);
        angle.set(f.id, a);
        prev = a;
      }
      let next = Infinity;
      for (let i = arc.length - 1; i >= 0; i--) {
        const a = Math.min(angle.get(arc[i].id)!, hi, next - step);
        angle.set(arc[i].id, a);
        next = a;
      }
    }
  }

  // positions: radius from the ring (with a small radial wobble that never crosses rings),
  // seeded vertical scatter that thickens toward the rim
  const outer = radii[G - 1] ?? o.innerRadius;
  const wobble = 0.12 * o.ringStep;
  const gauss = () => {
    let u = 0;
    while (u === 0) u = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  };
  const stars: StarLayout[] = formulas.map((f) => {
    if (f === hub)
      return { id: f.id, family: f.family, generation: 0, hub: true, angle: 0, radius: 0, x: 0, y: 0, z: 0 };
    const g = gen.get(f.id)!;
    const a = angle.get(f.id)!;
    const r = radii[g] + (rng() - 0.5) * 2 * wobble;
    const y = gauss() * o.thickness * (0.35 + 0.65 * (r / outer));
    return {
      id: f.id,
      family: f.family,
      generation: g,
      hub: false,
      angle: a,
      radius: r,
      x: r * Math.cos(a),
      y,
      z: r * Math.sin(a),
    };
  });

  return {
    stars,
    rings: radii.map((radius, g) => ({ generation: g, radius, count: counts[g] })),
    sectors: sectors.map((s) => {
      const fam = occ.get(s.family)!;
      let last = 0;
      fam.forEach((n, g) => n && (last = g));
      return { ...s, edgeRadius: radii[last] + o.ringStep * 0.8 };
    }),
    outerRadius: outer,
  };
}
