# Contributing

This repository holds two things: **Token X-ray** (the `xray/` package and the story at `/`) and the
**Lineage** atlas (`/atlas/`). Both are only useful if they are right; the process below exists to keep
them that way. Run `bash scripts/setup.sh` once, and `npm run ci` and `node scripts/xray.mjs test` before
every pull request.

## Token X-ray

### Adding a check

1. **State the intended weight.** Write the closed form in `xray/src/token_xray/intended.py`: the weight
   every token should receive under the documented behaviour.
2. **Run the real code.** Add a runner that drives the library's own training loop, trainer or collator on
   the probe (`runners.py`, `trl_runners.py`, `ddp.py`). Change as few defaults as possible, and list every
   change in `xray/README.md` under "What each real run changes from the defaults".
3. **Test both directions.** The test must pass on correct code and fail on an injected fault. Add the
   fault to `scripts/mutate.py` (or, for a library bug, to `xray/tests/conftest.py`) and confirm it is
   caught. Tolerances follow the precision floor (1e-6 in float64); never loosen one to pass.
4. **Label it honestly.** A film is a **Real run** (the library's own code), an **Emulation** (the
   documented mechanism of a bug, reproduced in plain PyTorch, never described as the library's behaviour)
   or **Computed** (from `src/core`). Claims about upstream behaviour link the issue or source line.
5. **Export, don't type.** Add the scenario to `scenarios.py`, run `node scripts/xray.mjs export`, and
   draw it from `public/xray/data.json`. The story never contains a number typed by hand.

### Reporting a training-stack bug you found with it

Open an issue here with the scenario, the versions (`token-xray report` prints them) and the measured
versus intended weights. Before reporting upstream, check the library's open issues and contribution
guidelines.

## Lineage atlas

### Adding a formula

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

### Notation conventions

- Vectors are bold lowercase (`\mathbf{q}`), matrices uppercase (`Q`), scalars italic.
- Queries, keys and values are `Q, K, V` throughout; their token keys are `q`, `k`, `v`.
- RL follows Sutton & Barto (2nd ed.): `S_t, A_t, R_{t+1}`, returns `G_t`, discount `\gamma`.
- Probabilities in evals are proportions in [0, 1], displayed as percentages in the UI.

## Code style

TypeScript in strict mode, ESLint and Prettier (`npm run lint`, `npm run format`). `src/core` stays free
of DOM and rendering code; the story and the atlas UI never compute statistics or formulas themselves.
Python in `xray/` targets 3.10.2+ and keeps the probe, readout and closed forms free of library-specific
code, which lives in the runners.

## Design

Check changes to the story at 390 px wide and with reduced motion.

## Security

See [SECURITY.md](SECURITY.md).
