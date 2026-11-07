/**
 * Token identity across two formulas. A token is (key, occurrence): the second "q" in a formula is
 * "q#1". Matching by occurrence keeps morphs stable when a term appears more than once.
 */
export function tokenIds(keys: string[]): string[] {
  const seen = new Map<string, number>();
  return keys.map((k) => {
    const n = seen.get(k) ?? 0;
    seen.set(k, n + 1);
    return `${k}#${n}`;
  });
}

export interface TokenDiff {
  persist: string[]; // in both: glide from old position to new
  enter: string[]; // only in the new formula: the "added term", highlighted
  exit: string[]; // only in the old formula: fades out
}

export function diffTokens(oldKeys: string[], newKeys: string[]): TokenDiff {
  const a = new Set(tokenIds(oldKeys));
  const b = tokenIds(newKeys);
  const bSet = new Set(b);
  return {
    persist: b.filter((id) => a.has(id)),
    enter: b.filter((id) => !a.has(id)),
    exit: [...a].filter((id) => !bSet.has(id)),
  };
}
