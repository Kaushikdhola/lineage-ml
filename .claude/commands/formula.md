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

Work in batches of at most 8 formulas. After each batch run `npm run ci`, fix everything, and show me
the `npm run atlas:check` report plus a list of anything you were unsure about (notation choices,
contested history, ambiguous sources). Do not continue to the next batch until I say so.
