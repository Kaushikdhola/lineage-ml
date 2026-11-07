import type { MeanCI } from "../core/stats";

/**
 * A simulator runs an experiment twice, intact and with one term deleted, over N seeds, and reports
 * means with 95% confidence intervals. All numbers come from src/core; simulators only orchestrate.
 */

export interface Band {
  mean: number[];
  lo: number[];
  hi: number[];
}

export interface SimStat {
  label: string;
  /** What the number means, in one plain sentence. */
  hint: string;
  intact: MeanCI;
  ablated: MeanCI;
  digits: number;
}

export interface SimReport {
  id: string;
  seeds: number;
  chart: {
    title: string;
    xLabel: string;
    yLabel: string;
    xs: number[];
    intact: Band;
    ablated: Band;
  };
  stats: SimStat[];
  labels: { intact: string; ablated: string };
  /** The sentence the evidence supports. Tests assert it. */
  claim: string;
  /** How the comparison was made fair, and what is a toy. */
  method: string;
}

export interface RunOptions {
  seeds?: number;
  onProgress?: (done: number, total: number) => void;
  signal?: AbortSignal;
}

export interface Simulator {
  id: string;
  run(options?: RunOptions): Promise<SimReport>;
}

/** Yield to the browser between seeds so the page stays responsive while a simulator runs. */
export const yieldToHost = (): Promise<void> => new Promise((r) => setTimeout(r, 0));
