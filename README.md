# Lineage

> **New: [Token X-ray](story/index.html)** (`/story/`): radiographs of real training steps, showing the
> gradient weight every token actually received and where documents leak into each other, measured
> through unmodified training code. Python package in [`xray/`](xray/README.md).

**The family tree of AI math.** Watch the equations behind modern AI evolve into each other: from the
dot product to multi-head attention, from the Bellman equation to DPO, from benchmark accuracy to
pass^k.

Every formula in Lineage is **computed, not drawn**. Each one is backed by a real implementation that is
tested against independent NumPy / SciPy / statsmodels references (and PyTorch where installed).

## What makes it different

- **Formulas evolve in front of you.** Moving from a formula to its child morphs the equation term by
  term: shared terms glide into place, the new term arrives highlighted, and the step is named
  ("divide by √d_k to keep unit variance") and labelled _historical_ or _conceptual_.
- **The formula is the interface.** Hover a symbol and its meaning, its control and its object in the
  scene light up. Hover the object and the symbol lights up.
- **One idea, many worlds.** Softmax is the hub: the same function is attention weights, an agent's
  policy, DPO's preference model and Bradley–Terry leaderboard odds.
- **Honest by construction.** Notation is consistent across the atlas, every formula names the code that
  computes it, and every citation shows whether its locator has been checked against the source.

## Atlas today

| Family      | Formulas | Highlights                                                    |
| ----------- | -------: | ------------------------------------------------------------- |
| Foundations |        4 | dot product, cosine similarity, softmax, temperature          |
| Attention   |        7 | Bahdanau → Luong → scaled dot-product → causal → multi-head   |
| Agents      |        8 | return → Bellman → Q-learning, REINFORCE → PPO → RLHF → DPO   |
| Evals       |        7 | accuracy, Wilson, McNemar, pass@k, pass^k, Bradley–Terry, Elo |

Planned: ~130 formulas across seven families. See [docs/ROADMAP.md](docs/ROADMAP.md).

## Quick start

Requires Node 20+, Python 3.10+ and git.

```bash
bash scripts/setup.sh   # installs dependencies and runs the full verification suite
npm run dev             # open the printed URL
```

## Commands

| Command               | What it does                                                    |
| --------------------- | --------------------------------------------------------------- |
| `npm run dev`         | Local development server                                        |
| `npm run verify`      | Regenerate oracle references, validate the atlas, run all tests |
| `npm run atlas:check` | Atlas report: coverage, citations, structural validation        |
| `npm run ci`          | Lint, format check, type check, verify and build (what CI runs) |
| `npm run deploy`      | Build and deploy to Cloudflare Pages                            |

The story's share image, `public/og/story.png`, is rendered from the hero film with
`.venv/bin/python scripts/render_og.py http://localhost:5181` while `npm run dev` runs; re-render it when the hero
data changes.

## Architecture

```
src/
  core/      Numerical engine. Pure TypeScript, no DOM. The only place math happens.
  atlas/     Formula data (TeX, notation, citations), lineage edges, graph utilities, validation.
  math/      KaTeX rendering, token identity, the morph engine and the symbol bus.
  scene/     three.js scenes, driven by core and bound to the symbol bus.
  ui/        The application shell and live probes.
python/
  oracle.py  Independent reference implementations → tests/fixtures/core.json
tests/       Engine-vs-oracle tests, atlas validation, token diffing.
```

**Data flow:** `atlas` describes a formula → `math` renders and morphs it → `ui` probes and `scene`
visualize it → every number comes from `core` → `core` is held to `python/oracle.py`.

## Quality bar

- A computed formula must match the oracle. Tests are mutation-checked: planting a wrong scaling
  factor, an off-by-one mask, a reversed preference or a one-sided test makes them fail.
- Fixtures are generated, never hand-edited. Tolerances are never loosened to make a test pass.
- Every tagged term in a formula has a notation entry. Every edge names its transformation.
- A citation is marked verified only after its locator has been checked against the source.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
