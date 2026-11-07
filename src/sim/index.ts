import { sqrtDkSimulator } from "./sqrtDk";
import type { Simulator } from "./types";

export type { Band, RunOptions, SimReport, SimStat, Simulator } from "./types";

/** Every simulator an ablation can name. */
export const SIMULATORS: Record<string, Simulator> = {
  [sqrtDkSimulator.id]: sqrtDkSimulator,
};
