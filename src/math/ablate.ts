/**
 * Deleting a term from TeX for ablation mode. Removes every \tok{key}{…} group, then tidies what
 * is left so the formula still renders: a fraction that lost its denominator becomes its numerator,
 * and an operator left hanging next to a bracket or the end of a group is dropped.
 */

/** Index just past the brace group that opens at `open` (tex[open] must be "{"). */
function groupEnd(tex: string, open: number): number {
  if (tex[open] !== "{") throw new Error(`expected "{" at ${open}`);
  let depth = 0;
  for (let i = open; i < tex.length; i++) {
    const c = tex[i];
    if (c === "\\") {
      i++; // skip escaped character, e.g. \{ or \}
      continue;
    }
    if (c === "{") depth++;
    else if (c === "}" && --depth === 0) return i + 1;
  }
  throw new Error("unbalanced braces");
}

function removeToken(tex: string, key: string): string {
  const marker = `\\tok{${key}}{`;
  let out = tex;
  for (let at = out.indexOf(marker); at !== -1; at = out.indexOf(marker)) {
    const end = groupEnd(out, at + marker.length - 1);
    out = out.slice(0, at) + out.slice(end);
  }
  return out;
}

/** \frac{A}{} → {A}; \frac{}{B} → \frac{1}{B}. Repeats until nothing changes. */
function tidyFractions(tex: string): string {
  const re = /\\(?:d|t)?frac\s*\{/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(tex))) {
    const numOpen = m.index + m[0].length - 1;
    const numEnd = groupEnd(tex, numOpen);
    let denOpen = numEnd;
    while (tex[denOpen] === " ") denOpen++;
    if (tex[denOpen] !== "{") continue;
    const denEnd = groupEnd(tex, denOpen);
    const num = tex.slice(numOpen + 1, numEnd - 1).trim();
    const den = tex.slice(denOpen + 1, denEnd - 1).trim();
    if (!den) {
      tex = `${tex.slice(0, m.index)}{${num}}${tex.slice(denEnd)}`;
      re.lastIndex = m.index;
    } else if (!num) {
      tex = `${tex.slice(0, numOpen)}{1}${tex.slice(numEnd)}`;
      re.lastIndex = m.index;
    }
  }
  return tex;
}

/** Drops binary operators stranded by a deletion: "(a + )" → "(a)", "( + b)" → "(b)". */
function tidyOperators(tex: string): string {
  return tex
    .replace(/\s*[+-]\s*(?=\\right[)\]|]|\}|$)/g, "")
    .replace(/(\\left[([|]|\{|=)\s*\+\s*/g, "$1 ")
    .replace(/\s{2,}/g, " ")
    .trim();
}

export function ablateTex(tex: string, key: string): string {
  if (!tex.includes(`\\tok{${key}}{`)) throw new Error(`ablateTex: no token "${key}"`);
  return tidyOperators(tidyFractions(removeToken(tex, key)));
}
