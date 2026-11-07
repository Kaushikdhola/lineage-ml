import { EDGES, FORMULAS } from "./formulas";
import type { Edge, Formula } from "./schema";

export const byId = new Map<string, Formula>(FORMULAS.map((f) => [f.id, f]));

export function getFormula(id: string): Formula {
  const f = byId.get(id);
  if (!f) throw new Error(`unknown formula: ${id}`);
  return f;
}

export const parentsOf = (id: string): Edge[] => EDGES.filter((e) => e.to === id);
export const childrenOf = (id: string): Edge[] => EDGES.filter((e) => e.from === id);

/** All formulas reachable upstream of `id`, nearest first. */
export function ancestors(id: string): string[] {
  const seen = new Set<string>();
  const queue = [id];
  while (queue.length) {
    const cur = queue.shift()!;
    for (const e of parentsOf(cur))
      if (!seen.has(e.from)) {
        seen.add(e.from);
        queue.push(e.from);
      }
  }
  return [...seen];
}

/** Topological order (Kahn). Throws on cycles. */
export function topologicalOrder(): string[] {
  const indeg = new Map(FORMULAS.map((f) => [f.id, 0]));
  for (const e of EDGES) indeg.set(e.to, (indeg.get(e.to) ?? 0) + 1);
  const ready = FORMULAS.filter((f) => indeg.get(f.id) === 0).map((f) => f.id);
  const order: string[] = [];
  while (ready.length) {
    const id = ready.shift()!;
    order.push(id);
    for (const e of childrenOf(id)) {
      indeg.set(e.to, indeg.get(e.to)! - 1);
      if (indeg.get(e.to) === 0) ready.push(e.to);
    }
  }
  if (order.length !== FORMULAS.length) throw new Error("lineage graph has a cycle");
  return order;
}

/** One path from a root to `id`, preferring historical edges: the "story" of a formula. */
export function lineagePath(id: string): string[] {
  const path = [id];
  let cur = id;
  for (;;) {
    const ps = parentsOf(cur);
    if (!ps.length) return path.reverse();
    const pick = ps.find((e) => e.kind === "historical") ?? ps[0];
    cur = pick.from;
    path.push(cur);
  }
}

/** Token keys used in a TeX string, in order of appearance (duplicates kept). */
export function tokenKeys(tex: string): string[] {
  return [...tex.matchAll(/\\tok\{([^}]*)\}/g)].map((m) => m[1]);
}

export interface AtlasIssue {
  id: string;
  problem: string;
}

/** Structural validation. Rendering is checked separately (needs KaTeX). */
export function validateAtlas(coreExports: Record<string, unknown>): AtlasIssue[] {
  const issues: AtlasIssue[] = [];
  const ids = new Set<string>();
  for (const f of FORMULAS) {
    if (ids.has(f.id)) issues.push({ id: f.id, problem: "duplicate id" });
    ids.add(f.id);
    if (!/^[a-z0-9-]+$/.test(f.id)) issues.push({ id: f.id, problem: "id must be kebab-case" });
    for (const key of new Set(tokenKeys(f.tex))) {
      if (!/^[A-Za-z0-9]+$/.test(key)) issues.push({ id: f.id, problem: `bad token key "${key}"` });
      if (!(key in f.symbols)) issues.push({ id: f.id, problem: `token "${key}" has no entry in symbols` });
    }
    for (const key of Object.keys(f.symbols))
      if (!tokenKeys(f.tex).includes(key))
        issues.push({ id: f.id, problem: `symbol "${key}" is not used in tex` });
    if (f.compute !== null && typeof coreExports[f.compute] !== "function")
      issues.push({ id: f.id, problem: `compute "${f.compute}" is not a function exported by src/core` });
    if (f.compute === null && !f.computeNote)
      issues.push({ id: f.id, problem: "compute is null without a note" });
    for (const ab of f.ablations ?? []) {
      if (!tokenKeys(f.tex).includes(ab.token))
        issues.push({ id: f.id, problem: `ablation token "${ab.token}" is not in tex` });
      if (ab.choices.length !== 3 || new Set(ab.choices).size !== 3)
        issues.push({ id: f.id, problem: `ablation "${ab.token}" needs three distinct choices` });
      if (![0, 1, 2].includes(ab.answer))
        issues.push({ id: f.id, problem: `ablation "${ab.token}" answer must be 0, 1 or 2` });
    }
  }
  for (const e of EDGES) {
    if (!ids.has(e.from)) issues.push({ id: e.from, problem: `edge from unknown formula` });
    if (!ids.has(e.to)) issues.push({ id: e.to, problem: `edge to unknown formula` });
    if (e.from === e.to) issues.push({ id: e.from, problem: "self edge" });
  }
  const seen = new Set<string>();
  for (const e of EDGES) {
    const k = `${e.from}->${e.to}`;
    if (seen.has(k)) issues.push({ id: e.from, problem: `duplicate edge ${k}` });
    seen.add(k);
  }
  try {
    topologicalOrder();
  } catch (err) {
    issues.push({ id: "*", problem: String(err) });
  }
  return issues;
}

/** Shortest directed path from `from` to `to` (inclusive), preferring historical edges on ties. */
export function pathBetween(from: string, to: string): string[] | null {
  const prev = new Map<string, string>();
  const seen = new Set([from]);
  const queue = [from];
  while (queue.length) {
    const cur = queue.shift()!;
    if (cur === to) {
      const path = [to];
      while (path[0] !== from) path.unshift(prev.get(path[0])!);
      return path;
    }
    const out = childrenOf(cur).sort(
      (a, b) => Number(b.kind === "historical") - Number(a.kind === "historical"),
    );
    for (const e of out)
      if (!seen.has(e.to)) {
        seen.add(e.to);
        prev.set(e.to, cur);
        queue.push(e.to);
      }
  }
  return null;
}
