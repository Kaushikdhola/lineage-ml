"""
Leak mode: does one sequence's loss send gradient into another sequence?

With attention live, run K backward passes. In pass i, only sequence i keeps its labels. The
gradient that reaches the positions of any other sequence j is a leak. On a correct stack the leak
matrix is exactly zero off the diagonal.

Layouts:
  * padded:            one sequence per row, right padding, attention mask. Reference: no leaks.
  * packed_naive:      sequences concatenated into one row with no boundary information.
  * packed_position_ids: concatenated, position_ids restart at 0 for each sequence (the
                       padding-free / flattening convention), no attention mask.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .probe import TracerBatch, build_probe
from .readout import recording
from .runners import collate

LAYOUTS = ("padded", "packed_naive", "packed_position_ids")


def _inputs(batch: TracerBatch, layout: str, keep: int):
    seqs = batch.sequences
    if layout == "padded":
        b = collate([{"input_ids": s} for s in seqs])
        lab = b["labels"].clone()
        for i in range(len(seqs)):
            if i != keep:
                lab[i] = -100
        return {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}, lab
    ids = torch.tensor([[t for s in seqs for t in s]])
    lab = torch.full_like(ids, -100)
    start = sum(len(s) for s in seqs[:keep])
    L = len(seqs[keep])
    lab[0, start : start + L] = ids[0, start : start + L]
    # the first token of a sequence must never be a target of the previous sequence's last token
    lab[0, start] = -100
    inputs = {"input_ids": ids}
    if layout == "packed_position_ids":
        inputs["position_ids"] = torch.tensor([[p for s in seqs for p in range(len(s))]])
    return inputs, lab


def leak_matrix(batch: TracerBatch, layout: str, attn_implementation: str = "sdpa", seed: int = 0) -> np.ndarray:
    """M[i, j] = ||grad reaching sequence j from sequence i's loss|| / ||grad reaching sequence i||."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    K = len(batch.sequences)
    M = np.zeros((K, K))
    for i in range(K):
        model = build_probe(batch.vocab_size, "leak", seed=seed, attn_implementation=attn_implementation)
        inputs, lab = _inputs(batch, layout, i)
        with recording(model) as rec:
            logits = model(**inputs).logits
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum"
            )
            loss.backward()
        norms = np.zeros(K)
        for tok, g in rec.grads.items():
            norms[batch.seq_of_token[tok]] += float(g.norm()) ** 2
        norms = np.sqrt(norms)
        M[i] = norms / norms[i] if norms[i] > 0 else norms
    return M


def max_leak(M: np.ndarray) -> float:
    off = M.copy()
    np.fill_diagonal(off, 0.0)
    return float(off.max())
