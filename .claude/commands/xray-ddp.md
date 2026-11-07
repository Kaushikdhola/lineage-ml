---
description: Add the data-parallel relation (world size 1 vs 2 on CPU gloo) as a real run
---

Read CLAUDE.md and xray/README.md.

Add relation R2 to token-xray: per-token weights must be identical for world size 1 and world size 2 with
uneven token counts per rank.

Important: activation gradients on each rank do NOT include the cross-rank average (that is applied to
parameter gradients by the all-reduce). So for this relation read the weights from the embedding
**parameter** gradient after the all-reduce, using unique tracer rows: weight of token i is
`<grad_row_i, r_i> / ||r_i||^2` with `r_i` the unit gradient of that row. Add this "row readout" to
`readout.py` with a test that it agrees with the activation readout on a single process.

1. Launch with `torch.multiprocessing.spawn` (gloo backend, CPU) wrapping the model in DDP; also run the
   transformers Trainer under `torchrun --nproc_per_node 2` if feasible on CPU.
2. Cases: equal tokens per rank, uneven tokens per rank, and a `DistributedSampler` that pads by
   duplicating samples (duplicates collide on tracer ids: report the collision as a finding, it is real).
3. Tests: passes for a correct global token mean across ranks; an injected per-rank mean (average of
   per-rank token means) must fail.
4. Export and add one film to the accumulation chapter if the result is worth showing.
   Run `node scripts/xray.mjs test` and `npm run ci`.
