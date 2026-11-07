/** Seeded PRNG (mulberry32): reproducible demos and recordings. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal sampler (Box–Muller) on top of a uniform PRNG. */
export function gaussian(rng: () => number): () => number {
  return () => {
    let u = 0;
    while (u === 0) u = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  };
}

export function randomMatrix(rows: number, cols: number, seed: number): number[][] {
  const g = gaussian(mulberry32(seed));
  return Array.from({ length: rows }, () => Array.from({ length: cols }, g));
}
