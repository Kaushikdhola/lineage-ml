# token-xray

**See what your training stack actually does to every token.**

Most silent LLM training bugs, and many objective design choices, change one quantity: the
gradient weight each token receives. Others leak gradient from one sequence into another. Token X-ray
measures both, through your **unmodified** training code, and compares them with what the equation
says.

```python
from token_xray import xray

def my_step(model, sequences):
    # batch `sequences` exactly as your trainer does, then forward + backward
    ...

report = xray(my_step, lengths=[5, 9, 3, 12, 7, 4, 10, 6])
report.assert_token_mean()   # every target token received weight 1/N
print(report.table())
```

## How it works

A tiny LLaMA "probe" gives every token in the batch a unique id. In aggregation mode the attention
output is zeroed and frozen, so a position's loss depends only on its own embedding. The gradient the
stack produces at token *i* is then `w_i · r_i`, where `r_i` is the token's unit-weight gradient
(computed on an identical probe), and the readout recovers:

- `w_i = ⟨g_i, r_i⟩ / ‖r_i‖²` — the weight the stack applied (×N, so a global token mean is 1.0)
- `cos(g_i, r_i)` — below 1 when the stack changed the gradient in a way that is not a rescaling

In leak mode attention is live. One backward per sequence, with only that sequence labelled; any
gradient reaching another sequence is a leak. On a correct stack it is exactly zero.

## What is checked today

| Check | Runs | Result (transformers 5.22.2, torch 2.15) |
| --- | --- | --- |
| Gradient accumulation 1×8 vs 4×2 | real `Trainer` | matches the token mean (≤1e-7) |
| Sample order (shuffled) | real `Trainer` | invariant |
| Logged loss equals the objective | real `Trainer` | equal |
| Left vs right padding | real `Trainer` | **differs**: with labels = input ids, the pad before each left-padded sequence predicts its first token, adding a target (labelling convention, not a library bug) |
| Label smoothing | real `Trainer` | flagged: cosine < 1 (declared configuration) |
| Mean of micro-batch means | emulation | detected; matches closed form |
| Later micro-batches dropped | emulation | detected; matches closed form |
| pad = EOS label masking | emulation | detected: EOS targets get zero weight |
| Padded rows | real model | no leak |
| Packed, no boundaries | real model | leaks (later documents into earlier) |
| Packed, `position_ids` reset, `use_cache=False` | real model | no leak |
| Packed, `position_ids` reset, default `use_cache` | real model | **leaks**, no warning — candidate finding, see below |

**Emulations** reproduce the documented mechanism of a historical bug in plain PyTorch. They are always
labelled and never presented as the library itself.

### Candidate finding: packed sequences and `use_cache`

transformers detects packed sequences from `position_ids` only when no cache object exists. Most model
configs default to `use_cache=True`, which creates a cache even in training mode, so a hand-written loop
that feeds packed `position_ids` without `use_cache=False` attends across documents silently. The
`Trainer` is not affected: it sets `model.config.use_cache` from `TrainingArguments.use_cache`. This
needs confirmation from the maintainers before it is described as a bug.

## Precision

LLaMA's RMSNorm and the HF loss compute in float32 inside a float64 model, so measured weights carry
~1e-7 relative error. Tests use a tolerance of 1e-6. The `eager` attention path in float64 produces NaN
on fully padded rows; the probe uses `sdpa`, the default.

## Development

```bash
pip install -e ".[dev]"
pytest
token-xray report
token-xray export ../public/xray/data.json
```

Every check is mutation-tested: corrupting the readout, the recorder, the probe, a closed form or the
leak harness makes the suite fail.
