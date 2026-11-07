/** Validates the atlas and prints a coverage report. Exit code 1 on any issue. */
import * as core from "../src/core";
import { EDGES, FORMULAS } from "../src/atlas/formulas";
import { tokenKeys, validateAtlas } from "../src/atlas/graph";
import { FAMILY_LABEL, type Family } from "../src/atlas/schema";
import { renderTexToString } from "../src/math/katex";

const issues = validateAtlas(core as unknown as Record<string, unknown>);
for (const f of FORMULAS) {
  try {
    const html = renderTexToString(f.tex);
    for (const key of new Set(tokenKeys(f.tex)))
      if (!html.includes(`data-tok="${key}"`))
        issues.push({ id: f.id, problem: `token ${key} missing from render` });
  } catch (err) {
    issues.push({ id: f.id, problem: `KaTeX: ${(err as Error).message}` });
  }
}

const fams = Object.keys(FAMILY_LABEL) as Family[];
console.log("\nLineage atlas");
for (const fam of fams) {
  const fs = FORMULAS.filter((f) => f.family === fam);
  const computed = fs.filter((f) => f.compute !== null).length;
  console.log(
    `  ${FAMILY_LABEL[fam].padEnd(12)} ${String(fs.length).padStart(3)} formulas  (${computed} computed)`,
  );
}
const cited = FORMULAS.filter((f) => f.citation);
const verified = cited.filter((f) => f.citation!.status === "verified").length;
console.log(
  `  ${"Total".padEnd(12)} ${String(FORMULAS.length).padStart(3)} formulas, ${EDGES.length} lineage edges`,
);
console.log(
  `  Citations: ${cited.length} (${verified} verified against the source, ${cited.length - verified} unverified)`,
);

if (issues.length) {
  console.error(`\n✗ ${issues.length} issue(s):`);
  for (const i of issues) console.error(`  ${i.id}: ${i.problem}`);
  process.exit(1);
}
console.log("\n✓ atlas is valid\n");
