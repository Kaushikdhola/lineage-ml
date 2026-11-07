---
description: A fresh, hostile expert review of the math, history and claims before release
---

Spawn a subagent with no prior context. Give it `src/atlas/formulas.ts`, `src/core/`,
`python/oracle.py`, `README.md` and the output of `npm run atlas:check`. Instruct it to act as a
senior research scientist hunting for anything that would embarrass the author in public:

- wrong or non-canonical formulas, sign or index errors, notation inconsistent across the atlas
- lineage edges labelled historical that are not, or op descriptions that misstate the step
- citations with wrong years, venues, authors or locators
- oracle references that are not truly independent of the TypeScript implementation
- any claim in README or UI copy that overstates what the project does

It must quote the exact text for each issue and propose a correction. Show me the report, then fix
what I approve and run `npm run ci`.
