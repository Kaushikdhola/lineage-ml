# Contributing to Lineage

Lineage is only useful if it is right. The process below exists to keep it that way.

## Adding a formula

1. **Find the canonical form in the original source.** Record the authors, year, title, venue, URL and
   a precise locator (`Eq. (7)`, `Section 3.2.2`). Leave `status: "unverified"` until someone has
   checked the locator against the source itself.
2. **Write the TeX with tokens.** Wrap every meaningful term in `\tok{key}{...}`. Reuse the parent's
   keys for terms that carry over, so the morph shows real continuity (`q` stays the query from
   Luong attention through multi-head attention).
3. **Fill the notation.** Every token key needs a plain-language entry in `symbols`.
4. **Connect it.** Add at least one edge with an `op` phrased as an instruction ("replace the policy
   average with a max") and an honest `kind`: `historical` only if the child was actually developed
   from the parent.
5. **Compute it (computed tier).** Implement it in `src/core`, add an independent reference to
   `python/oracle.py` (use a library routine or a different method), and test it in
   `tests/core.test.ts`. Then plant a bug and confirm the test catches it.
6. **Run `npm run ci`.** It must pass.

## Notation conventions

- Vectors are bold lowercase (`\mathbf{q}`), matrices uppercase (`Q`), scalars italic.
- Queries, keys and values are `Q, K, V` throughout; their token keys are `q`, `k`, `v`.
- RL follows Sutton & Barto (2nd ed.): `S_t, A_t, R_{t+1}`, returns `G_t`, discount `\gamma`.
- Probabilities in evals are proportions in [0, 1], displayed as percentages in the UI.

## Code style

TypeScript in strict mode, ESLint and Prettier (`npm run lint`, `npm run format`). `src/core` stays
free of DOM and rendering code. UI and scenes never compute statistics or formulas themselves.
