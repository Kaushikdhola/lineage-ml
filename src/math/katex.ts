import katex from "katex";

/**
 * Shared KaTeX configuration. \tok{key}{content} expands to \htmlData{tok=key}{content},
 * which renders a <span data-tok="key"> around the term. `trust` allows only that command.
 */
export const KATEX_OPTIONS: katex.KatexOptions = {
  displayMode: true,
  throwOnError: true,
  strict: "ignore",
  output: "html",
  macros: { "\\tok": "\\htmlData{tok=#1}{#2}" },
  trust: (ctx) => ctx.command === "\\htmlData",
};

export function renderTex(tex: string, el: HTMLElement): void {
  katex.render(tex, el, { ...KATEX_OPTIONS, macros: { ...KATEX_OPTIONS.macros } });
}

export function renderTexToString(tex: string): string {
  return katex.renderToString(tex, { ...KATEX_OPTIONS, macros: { ...KATEX_OPTIONS.macros } });
}
