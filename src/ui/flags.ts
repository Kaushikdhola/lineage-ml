/**
 * Feature flags, read once from the page URL at load.
 *
 *   ?ablation=1   ablation mode in the lab (drag a term out of a formula). Off by default while the
 *                 experience is unfinished: without it there is no hint, no marked term and no handler.
 */
export function readFlags(search: string): { ablation: boolean } {
  return { ablation: new URLSearchParams(search).get("ablation") === "1" };
}

export const FLAGS = readFlags(typeof location === "undefined" ? "" : location.search);

/** Query-string suffix that keeps enabled flags across in-app URL rewrites, e.g. "&ablation=1". */
export const flagQuery = (): string => (FLAGS.ablation ? "&ablation=1" : "");
