# Two-day launch plan

Goal: launch **Token X-ray** as a story from the math to measured radiographs of real training stacks,
plus a pip package researchers can run on their own training step. Lineage (the formula atlas) is the
supporting act: the story links to it and reuses its verified core and morph engine.

Rules for every step: `npm run ci` and `node scripts/xray.mjs test` pass before every commit. No number appears
on the site that a test or the exporter did not produce. Emulations are always labelled.

## Day 1: make it real and complete

| Order | Run in Claude Code | Outcome                                                                                       |
| ----- | ------------------ | --------------------------------------------------------------------------------------------- |
| 1     | `/xray-trl`        | Real TRL runs on the films: SFT with packing / padding-free, DPO, GRPO with scripted rollouts |
| 2     | `/xray-ddp`        | Data-parallel relation (world size 1 vs 2 on gloo) as a real run                              |
| 3     | `/report-finding`  | Issue draft and minimal reproduction for the `use_cache` packing behaviour; you file it       |
| 4     | `/story-polish`    | Story wired into the site, Open Graph image, copy pass, mobile pass                           |

## Day 2: make it unbreakable, then launch

| Order | Run in Claude Code            | Outcome                                                                   |
| ----- | ----------------------------- | ------------------------------------------------------------------------- |
| 5     | `/qa` (scope: `/`, the story) | Screenshots at 1440, 1080×1350 and 390 wide; fixes                        |
| 6     | `/redteam-xray`               | A fresh hostile reviewer checks every claim on the page and in the README |
| 7     | `/launch`                     | Deploy, publish the package, post drafts for Reddit, X and LinkedIn       |

## Cut lines if time runs out

Keep, in this order: the story with real Trainer runs and the emulations (done), the leak chapter (done),
`/report-finding`, deploy. Drop first: DDP, then GRPO through TRL (the GRPO chapter already uses verified
closed forms and says so on the film).

## Before posting anywhere

- The `use_cache` behaviour is described as a known issue (huggingface/transformers#50419) and linked (it is).
- Every film edge says Real run, Emulation, or Computed.
- `npm run ci`, `node scripts/xray.mjs test` and `token-xray report` are green on the deployed commit.
