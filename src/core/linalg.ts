/** Dense linear algebra on plain arrays. Small, explicit, and checked against NumPy. */
export type Vec = number[];
export type Mat = number[][];

export function shape(m: Mat): [number, number] {
  const rows = m.length;
  const cols = rows ? m[0].length : 0;
  for (const r of m) if (r.length !== cols) throw new Error("ragged matrix");
  return [rows, cols];
}

export function dot(a: Vec, b: Vec): number {
  if (a.length !== b.length) throw new Error(`dot: length ${a.length} vs ${b.length}`);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a: Vec): number {
  return Math.sqrt(dot(a, a));
}

export function transpose(m: Mat): Mat {
  const [r, c] = shape(m);
  return Array.from({ length: c }, (_, j) => Array.from({ length: r }, (_, i) => m[i][j]));
}

export function matmul(a: Mat, b: Mat): Mat {
  const [ar, ac] = shape(a);
  const [br, bc] = shape(b);
  if (ac !== br) throw new Error(`matmul: ${ar}x${ac} @ ${br}x${bc}`);
  const out: Mat = Array.from({ length: ar }, () => new Array<number>(bc).fill(0));
  for (let i = 0; i < ar; i++)
    for (let k = 0; k < ac; k++) {
      const aik = a[i][k];
      for (let j = 0; j < bc; j++) out[i][j] += aik * b[k][j];
    }
  return out;
}

export function matvec(m: Mat, v: Vec): Vec {
  return m.map((row) => dot(row, v));
}

export function addVec(a: Vec, b: Vec): Vec {
  if (a.length !== b.length) throw new Error("addVec: length mismatch");
  return a.map((x, i) => x + b[i]);
}

export function scaleMat(m: Mat, s: number): Mat {
  return m.map((r) => r.map((x) => x * s));
}

export function addMat(a: Mat, b: Mat): Mat {
  return a.map((r, i) => r.map((x, j) => x + b[i][j]));
}

/** Columns [start, end) of a matrix. */
export function colSlice(m: Mat, start: number, end: number): Mat {
  return m.map((r) => r.slice(start, end));
}

export function hconcat(blocks: Mat[]): Mat {
  return blocks[0].map((_, i) => blocks.flatMap((b) => b[i]));
}
