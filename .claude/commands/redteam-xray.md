---
description: Hostile expert review of every claim in the Token X-ray story, README and package
---

Spawn a subagent with no prior context. Give it `index.html` (the story), `src/story/main.ts`,
`public/xray/data.json`, `xray/README.md`, `xray/src/token_xray/`, `xray/tests/`, the output of
`token-xray report`, and `src/core/aggregation.ts` with its oracle. Instruct it to act as a senior
research scientist who trains LLMs at a frontier lab and wants to find anything that would embarrass the
author on Reddit or X:

- any sentence the measurements do not support, or that presents an emulation as a library's behaviour
- the readout method: conditions under which w_i or the cosine would be wrong, and whether the tests cover them
- the GRPO / Dr. GRPO / DAPO coefficients and the descriptions of each paper (check against the papers)
- the use_cache finding: is it real, is it already known, is the Trainer claim correct
- the historical bug descriptions (dates, versions, which project, which mechanism)
- anything that overstates novelty or impact

It must quote the exact text for each issue and propose corrected wording. Save the report verbatim to
`docs/redteam/<date>.md` and commit it on its own. Then fix every issue, choosing between the reviewer's
proposals and alternatives by the Autonomy rules; for anything you decide not to fix, or defer, log why in
docs/DECISIONS.md (or docs/ROADMAP.md for deferred method work).

**Finish.** Follow the Autonomy rules in CLAUDE.md. Update the docs and README this command touched, run `npm run ci` and `node scripts/xray.mjs test`, commit (one commit per logical change, no attribution lines), and print a report of 5 lines or fewer.
