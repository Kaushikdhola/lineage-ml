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
from .readout import read, read_pairs, recording, unit_gradients, unit_pair_gradients

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
         "detection is skipped: documents attend to each other with no warning. Known issue, reported in transformers#50419."),
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


# ── TRL: real runs of the unmodified trainers ──────────────────────────────

DPO_SHAPE = {"prompt_len": 3, "chosen": [4, 6, 3], "rejected": [5, 3, 7], "beta": 0.1}
GRPO_SHAPE = {"prompt_len": 3, "lengths": [3, 7, 4, 10], "rewards": [1.0, 0.2, 0.6, 0.0], "lr": 0.05,
              "max_completion_length": 16}
GRPO_RUNS = [  # (id, loss_type passed, scale_rewards passed, title)
    ("default", None, None, "GRPOTrainer, default loss"),
    ("grpo", "grpo", None, "GRPOTrainer, loss_type='grpo'"),
    ("dr_grpo", "dr_grpo", "none", "GRPOTrainer, loss_type='dr_grpo', scale_rewards='none'"),
]


def _rel_dev(got, want) -> float:
    scale = max(abs(w) for row in want for w in row) or 1.0
    return max(abs(g - w) for gr, wr in zip(got, want) for g, w in zip(gr, wr)) / scale


def sft_panels(lengths=SITE_LENGTHS, micro=MICRO, leak_lengths=LEAK_LENGTHS) -> dict:
    from .trl_runners import sft_leak_matrix, sft_trainer

    b = make_batch(lengths)
    unit = unit_gradients(build_probe(b.vocab_size), b)
    k = len(lengths) // micro
    tm = intended.token_mean(lengths)
    weights = []
    for pid, title, kw in [
        ("sft-padding-free", f"TRL SFTTrainer, padding_free=True, {k} accumulation steps", dict(padding_free=True)),
        ("sft-packing", f"TRL SFTTrainer, packing=True (bfd), {k} accumulation steps", dict(packing=True, max_length=32)),
    ]:
        res = sft_trainer(b, micro, k, **kw)
        x = read(res.grads, unit, b)
        got = x.normalized
        weights.append({
            "id": pid, "title": title, "kind": "real", "source": None,
            "note": "Documents flattened into one row per micro-batch, position_ids reset, first token of each document "
                    "masked as a target. Matches the global token mean.",
            "weights": [[round(v, 6) for v in row] for row in got],
            "intended": [[round(v, 6) for v in row] for row in tm],
            "maxDeviation": max(abs(g - w) for gr, wr in zip(got, tm) for g, w in zip(gr, wr)),
            "minCos": x.min_cos, "loggedLoss": res.logged_loss,
            "microBatches": len(res.info["micro_batches"]), "rows": res.info["rows"],
        })
    lb = make_batch(leak_lengths)
    leaks = []
    for pid, title, kw in [
        ("sft-leak-padding-free", "TRL SFTTrainer, padding_free=True", dict(packing=False)),
        ("sft-leak-packing", "TRL SFTTrainer, packing=True (bfd)", dict(packing=True, max_length=64)),
    ]:
        M = sft_leak_matrix(lb, **kw)
        leaks.append({"id": pid, "title": title, "kind": "real", "matrix": M.round(6).tolist(), "maxLeak": max_leak(M),
                      "note": "All documents in one flattened row, labels on one document per run. SFTTrainer passes "
                              "use_cache=False, so transformers detects the packed sequences from position_ids."})
    return {"lengths": lengths, "micro": micro, "weights": weights, "leakLengths": leak_lengths, "leaks": leaks}


def dpo_panel(shape=DPO_SHAPE) -> dict:
    from .trl_runners import dpo_batch, dpo_trainer

    pb = dpo_batch(shape["prompt_len"], shape["chosen"], shape["rejected"])
    res = dpo_trainer(pb, beta=shape["beta"])
    rows = [pb.pairs(k, pb.prompts[k // 2]) for k in range(len(pb.completions))]
    allp = [p for r in rows for p in r]
    pr = read_pairs(res["grads"], unit_pair_gradients(build_probe(pb.vocab_size), allp), allp)
    got = [[pr.weights[p] for p in r] for r in rows]
    ch, rj = intended.dpo_sigmoid([len(c) for c in pb.completions[0::2]], [len(c) for c in pb.completions[1::2]],
                                  res["deltas"], shape["beta"])
    want = [x for pair in zip(ch, rj) for x in pair]
    return {
        "id": "dpo", "title": "TRL DPOTrainer, sigmoid loss", "kind": "real", **shape, "lossType": res["loss_type"],
        "rowLabels": [f"Pair {k // 2 + 1}, {'chosen' if k % 2 == 0 else 'rejected'}, {len(c)} tokens"
                      for k, c in enumerate(pb.completions)],
        "deltas": res["deltas"], "weights": got, "intended": want,
        "maxRelDeviation": _rel_dev(got, want), "residual": pr.residual,
        "note": "Every token of a completion gets the same weight, beta * sigmoid(-beta * margin) / pairs, pushed up on "
                "the chosen answer and down on the rejected one. Sequence log-probs are sums, so a longer answer gets "
                "more total push.",
    }


def grpo_panels(shape=GRPO_SHAPE) -> list[dict]:
    from .trl_runners import grpo_batch, grpo_trainer, probe_at, ratios_between

    pb = grpo_batch(shape["prompt_len"], shape["lengths"])
    rows = [pb.pairs(k, pb.prompts[0]) for k in range(len(pb.completions))]
    allp = [p for r in rows for p in r]
    scored = [len(c) for c in pb.completions]  # including EOS
    out = []
    for pid, lt, sr, title in GRPO_RUNS:
        res = grpo_trainer(pb, shape["rewards"], loss_type=lt, scale_rewards=sr, learning_rate=shape["lr"],
                           max_completion_length=shape["max_completion_length"])
        cfg = res["config"]
        steps = []
        for s, st in enumerate(res["steps"]):
            pr = read_pairs(st["grads"], unit_pair_gradients(probe_at(st["state"], pb.vocab_size), allp), allp)
            got = [[pr.weights[p] for p in r] for r in rows]
            ratios = ratios_between(pb, st["state"], res["steps"][0]["state"])
            adv, want = intended.group_policy_gradient(
                scored, shape["rewards"], cfg["loss_type"], ratios=ratios, epsilon_low=cfg["epsilon"],
                epsilon_high=cfg["epsilon_high"], max_completion_length=cfg["max_completion_length"],
                scale_rewards=cfg["scale_rewards"] != "none")
            steps.append({"step": s + 1, "weights": got, "intended": want, "ratios": ratios, "advantages": adv,
                          "clipped": sum(1 for r in want for w in r if w == 0.0),
                          "maxRelDeviation": _rel_dev(got, want), "residual": pr.residual})
        out.append({"id": f"grpo-{pid}", "title": title, "kind": "real", "config": cfg, "lengths": scored,
                    "rewards": shape["rewards"], "learningRate": shape["lr"], "steps": steps})
    return out


def trl_section() -> dict:
    return {"sft": sft_panels(), "dpo": dpo_panel(), "grpo": grpo_panels()}


def export() -> dict:
    return {
        "schema": 1,
        "versions": _versions(),
        "aggregation": {"lengths": SITE_LENGTHS, "micro": MICRO, "panels": aggregation_panels()},
        "leak": {"lengths": LEAK_LENGTHS, "panels": leak_panels()},
        "trl": trl_section(),
    }
