"""
The frozen scenario set used by the test suite and the public site.

Every scenario states what it runs, whether it is a REAL library run or an EMULATION of a documented
mechanism, and what the equation says should happen. Results are measured, never assumed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import intended
from . import runners as R
from .leak import _inputs, leak_matrix, max_leak
from .probe import build_probe, make_batch
from .readout import read, recording, unit_gradients

# a deterministic batch with varied lengths; 12 sequences in 4 micro-batches of 3
SITE_LENGTHS = [6, 14, 4, 22, 9, 5, 17, 8, 12, 3, 19, 7]
MICRO = 3
LEAK_LENGTHS = [6, 9, 5, 8, 7]


def _versions() -> dict:
    import transformers

    v = {"torch": torch.__version__, "transformers": transformers.__version__}
    try:
        import trl

        v["trl"] = trl.__version__
    except ImportError:
        pass
    return v


def aggregation_panels(lengths=SITE_LENGTHS, micro=MICRO) -> list[dict]:
    b = make_batch(lengths)
    unit = unit_gradients(build_probe(b.vocab_size), b)
    k = len(lengths) // micro

    def panel(pid, title, kind, res, want, note, source=None):
        x = read(res.grads, unit, b)
        got = x.normalized
        dev = max(abs(g - w) for gr, wr in zip(got, want) for g, w in zip(gr, wr))
        return {
            "id": pid,
            "title": title,
            "kind": kind,
            "note": note,
            "source": source,
            "weights": [[round(v, 6) for v in row] for row in got],
            "intended": [[round(v, 6) for v in row] for row in want],
            "maxDeviation": dev,
            "minCos": x.min_cos,
            "loggedLoss": res.logged_loss,
        }

    tm = intended.token_mean(lengths)
    return [
        panel("reference", "Reference (plain PyTorch, global token mean)", "reference", R.reference(b, micro), tm,
              "The definition: every target token gets weight 1/N."),
        panel("hf-ga", f"transformers Trainer, {k} accumulation steps", "real", R.hf_trainer(b, micro, k), tm,
              "Unmodified Trainer. Matches the definition."),
        panel("hf-left", f"transformers Trainer, left padding", "real", R.hf_trainer(b, micro, k, padding_side="left"), tm,
              "With labels = input_ids, the padding position before each left-padded sequence predicts its first "
              "token, adding one target per padded sequence and re-weighting every other token. A labelling "
              "convention, not a library bug."),
        panel("hf-smoothing", "transformers Trainer, label smoothing 0.1", "real", R.hf_trainer(b, len(lengths), 1, label_smoothing=0.1), tm,
              "Declared configuration. Cosine < 1: the gradient is no longer a rescaling of each token's own gradient."),
        panel("emu-mean-of-means", "Emulation: mean of micro-batch means", "emulation", R.emulate_mean_of_means(b, micro),
              tm, "Mechanism of the gradient-accumulation loss bug fixed in transformers in October 2024.",
              "https://huggingface.co/blog/gradient_accumulation"),
        panel("emu-dropped", "Emulation: later micro-batches dropped", "emulation", R.emulate_dropped_microbatches(b, micro),
              tm, "Mechanism reported for an optimizer-offload path (DeepSpeed ZeRO-1/2 CPU offload, PR #6550).",
              "https://arxiv.org/abs/2604.23747"),
        panel("emu-pad-eos", "Emulation: pad token = EOS", "emulation", R.emulate_pad_equals_eos(b), tm,
              "Masking every label equal to the pad id also removes the real end-of-sequence targets."),
    ]


def leak_panels(lengths=LEAK_LENGTHS) -> list[dict]:
    b = make_batch(lengths)
    out = []
    for lay, title, note in [
        ("padded", "One sequence per row", "Reference. Zero leak by construction."),
        ("packed_naive", "Packed, no boundaries", "Concatenated documents with no position reset and no mask attend to each other."),
    ]:
        M = leak_matrix(b, lay)
        out.append({"id": lay, "title": title, "kind": "real", "note": note, "matrix": M.round(6).tolist(), "maxLeak": max_leak(M)})
    for uc, title, note in [
        (False, "Packed, position_ids reset, use_cache=False",
         "transformers detects packed sequences from position_ids. Zero leak. The Trainer sets use_cache from TrainingArguments, so this is what it runs."),
        (None, "Packed, position_ids reset, model default use_cache",
         "In a hand-written loop with the model's default use_cache=True, a cache object is created and packed-sequence "
         "detection is skipped: documents attend to each other with no warning. Candidate finding, pending maintainer review."),
    ]:
        M = custom_loop_leak(b, uc)
        out.append({"id": f"packed_pos_cache_{uc}", "title": title, "kind": "real", "note": note, "matrix": M.round(6).tolist(), "maxLeak": max_leak(M)})
    return out


def custom_loop_leak(b, use_cache) -> np.ndarray:
    K = len(b.sequences)
    M = np.zeros((K, K))
    for i in range(K):
        m = build_probe(b.vocab_size, "leak")
        m.train()
        inputs, lab = _inputs(b, "packed_position_ids", i)
        kw = {} if use_cache is None else {"use_cache": use_cache}
        with recording(m) as rec:
            logits = m(**inputs, **kw).logits
            F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum").backward()
        n = np.zeros(K)
        for t, g in rec.grads.items():
            n[b.seq_of_token[t]] += float(g.norm()) ** 2
        n = np.sqrt(n)
        M[i] = n / n[i]
    return M


def export() -> dict:
    return {
        "schema": 1,
        "versions": _versions(),
        "aggregation": {"lengths": SITE_LENGTHS, "micro": MICRO, "panels": aggregation_panels()},
        "leak": {"lengths": LEAK_LENGTHS, "panels": leak_panels()},
    }
