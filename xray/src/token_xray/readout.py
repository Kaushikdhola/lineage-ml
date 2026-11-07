"""
Readout: recover the weight the stack applied to each token.

Gradients are captured at the output of the embedding layer, per position, keyed by the tracer
token id at that position. Because tracer ids are unique, this works whatever the stack does with
order, padding side, micro-batching or packing.

    w_i  = <g_i, r_i> / ||r_i||^2        (the weight)
    cos_i = <g_i, r_i> / (||g_i|| ||r_i||) (1.0 means a pure rescaling)

r_i is the token's unit-weight gradient: the gradient of its own cross-entropy alone, computed on
an identical probe. cos_i < 1 means the stack changed the gradient in a way that is not a
rescaling (label smoothing, extra loss terms, leaks).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .probe import PAD_ID, TracerBatch


class GradRecorder:
    """Records d(loss)/d(embedding output) for every tracer token seen in backward passes."""

    def __init__(self, model):
        self.model = model
        self.grads: dict[int, torch.Tensor] = {}
        self._handle = None

    def _hook(self, module, inputs, output):
        ids = inputs[0].detach().clone()
        if not output.requires_grad:
            return

        def save(grad):
            g = grad.detach()
            flat_ids = ids.reshape(-1).tolist()
            flat_g = g.reshape(-1, g.shape[-1])
            for k, tok in enumerate(flat_ids):
                if tok == PAD_ID:
                    continue
                prev = self.grads.get(tok)
                self.grads[tok] = flat_g[k].clone() if prev is None else prev + flat_g[k]

        output.register_hook(save)

    def __enter__(self):
        self._handle = self.model.get_input_embeddings().register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()


@contextmanager
def recording(model):
    rec = GradRecorder(model)
    with rec:
        yield rec


def unit_gradients(model, batch: TracerBatch) -> dict[int, torch.Tensor]:
    """r_i for every target token: each sequence alone, summed per-token cross-entropy."""
    out: dict[int, torch.Tensor] = {}
    for seq in batch.sequences:
        ids = torch.tensor([seq])
        with recording(model) as rec:
            logits = model(input_ids=ids).logits
            loss = F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum")
            loss.backward()
        for t in seq[:-1]:
            out[t] = rec.grads[t]
        model.zero_grad(set_to_none=True)
    return out


@dataclass
class XRay:
    """Per-token weights, in sequence order. weights[i][t] is w for token t of sequence i (targets only)."""

    weights: list[list[float]]
    cosines: list[list[float]]
    n_targets: int

    @property
    def normalized(self) -> list[list[float]]:
        """w * N: equals 1.0 everywhere under a correct global token mean."""
        return [[w * self.n_targets for w in row] for row in self.weights]

    def as_array(self, normalized: bool = True) -> np.ndarray:
        rows = self.normalized if normalized else self.weights
        L = max(len(r) for r in rows)
        a = np.full((len(rows), L), np.nan)
        for i, r in enumerate(rows):
            a[i, : len(r)] = r
        return a

    @property
    def min_cos(self) -> float:
        return min(c for row in self.cosines for c in row)


def read(grads: dict[int, torch.Tensor], unit: dict[int, torch.Tensor], batch: TracerBatch) -> XRay:
    W, C = [], []
    for seq in batch.sequences:
        wr, cr = [], []
        for t in seq[:-1]:
            r = unit[t]
            g = grads.get(t, torch.zeros_like(r))
            rr = float(r @ r)
            gr = float(g @ r)
            wr.append(gr / rr)
            gn = float(g.norm())
            cr.append(1.0 if gn == 0.0 else gr / (gn * rr**0.5))
        W.append(wr)
        C.append(cr)
    return XRay(weights=W, cosines=C, n_targets=batch.n_targets)
