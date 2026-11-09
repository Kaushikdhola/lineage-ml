---
description: Add a batch of formulas to the atlas (from docs/ROADMAP.md) at full quality
argument-hint: <formula names, a family, or a chain from the roadmap>
---

Add these formulas to the atlas: $ARGUMENTS

Follow CLAUDE.md and CONTRIBUTING.md exactly. For each formula:

1. Find the canonical form in the original source (fetch the paper or book page). Record a precise
   locator. Keep `status: "unverified"`; `/verify-citations` does verification separately.
2. Write TeX with `\tok{}` tokens, reusing parent token keys for carried-over terms. Fill `symbols`.
3. Add edges with an instruction-style `op` and an honest `kind`.
4. Computed tier (marked `C` in the roadmap): implement in `src/core`, add an independent reference
   in `python/oracle.py`, add tests, then plant a bug and confirm a test fails. Catalogued tier
   (marked `·`): `compute: null` with a `computeNote`.
5. Update the status marks in `docs/ROADMAP.md`.

Work in batches of at most 8 formulas. After each batch run `npm run ci` and fix everything. Log anything
you were unsure about (notation choices, contested history, ambiguous sources) in docs/DECISIONS.md with
the choice you made, commit the batch, and continue with the next one. The final report includes the
`npm run atlas:check` summary.

**Finish.** Follow the Autonomy rules in CLAUDE.md. Update the docs and README this command touched, run `npm run ci` and `node scripts/xray.mjs test`, commit (one commit per logical change, no attribution lines), and print a report of 5 lines or fewer.
