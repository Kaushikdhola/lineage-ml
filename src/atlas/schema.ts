/**
 * The atlas data model.
 *
 * TeX uses one custom macro, \tok{key}{content}, to tag a term. Tagged terms are what the
 * morph engine tracks between formulas and what the 3D scenes bind to. Every key used in a
 * formula must be explained in that formula's `symbols` (this is the notation table).
 */
export type Family =
  "foundations" | "optimization" | "architecture" | "attention" | "generative" | "agents" | "evals";

export interface Citation {
  authors: string;
  year: number;
  title: string;
  venue?: string;
  url?: string;
  /** Where in the source: "Eq. (1)", "Section 3.2.2". */
  locator?: string;
  /** "verified" only after someone checked the locator against the source itself. */
  status: "unverified" | "verified";
}

export interface Formula {
  id: string;
  name: string;
  family: Family;
  tex: string;
  /** One-sentence intuition, no jargon. */
  idea: string;
  /** token key → meaning */
  symbols: Record<string, string>;
  /** Name of the export in src/core that computes this formula (traceability), or null with a reason. */
  compute: string | null;
  computeNote?: string;
  citation?: Citation;
  /** Hub formulas appear across families ("one idea, many worlds"). */
  hub?: boolean;
}

export interface Edge {
  from: string;
  to: string;
  /** The named transformation, phrased as an instruction: "divide logits by a temperature τ". */
  op: string;
  /** historical: the child was actually developed from the parent. conceptual: a teaching link. */
  kind: "historical" | "conceptual";
}

export const FAMILY_LABEL: Record<Family, string> = {
  foundations: "Foundations",
  optimization: "Optimization",
  architecture: "Architecture",
  attention: "Attention",
  generative: "Generative",
  agents: "Agents",
  evals: "Evals",
};
