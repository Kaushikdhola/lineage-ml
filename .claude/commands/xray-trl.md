---
description: Add real TRL runs (SFT packing / padding-free, DPO, GRPO with scripted rollouts) to token-xray
---

Read CLAUDE.md, xray/README.md and docs/DESIGN.md. Work in `xray/` with `.venv` (install `trl` and CPU
torch if missing). Every new scenario must be a REAL run of the installed library, use the existing probe
and readout, and come with tests that pass on correct behaviour and fail on an injected fault.

1. **SFT, padding-free / packing.** Run `trl.SFTTrainer` on the probe with `padding_free=True` (and with
   packing if the installed version supports it on CPU). Weight mode: assert the token mean. Leak mode:
   build the leak matrix through the trainer (one run per document, labels only on that document via a
   custom dataset, as in `scenarios.custom_loop_leak`). Record what the installed TRL does; do not assume.
2. **DPO.** Pairs of tracer sequences (chosen / rejected). Derive the intended per-token coefficient from
   the DPO loss in closed form, add it to `intended.py`, verify it with autograd in a test, then compare
   with what `DPOTrainer` actually applies.
3. **GRPO.** Patch generation so `GRPOTrainer` gets scripted completions with unique tracer ids and a
   fixed reward function; set `num_iterations >= 2` so ratios move. Compare the measured per-token
   coefficients with `src/core/aggregation.ts` (GRPO, Dr. GRPO / DAPO via `loss_type` if available) for
   the installed version, and report exactly which loss type the default uses.
4. Add the scenarios to `scenarios.py`, regenerate `public/xray/data.json` (`node scripts/xray.mjs export`), and
   give the story a film for each that is worth showing (Plate numbers continue; edge says Real run).
5. Run `node scripts/xray.mjs test` and `npm run ci`. Then plant one fault per new scenario and show it is caught.

Report: a table of scenario → result → library version. Anything that looks like a library bug is a
candidate: check upstream for existing reports (as `/report-finding` does) before any public claim, call it
a candidate on the page until then, and never file or comment upstream (hard stop).

**Finish.** Follow the Autonomy rules in CLAUDE.md. Update the docs and README this command touched, run `npm run ci` and `node scripts/xray.mjs test`, commit (one commit per logical change, no attribution lines), and print a report of 5 lines or fewer.
