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
    """Records d(loss)/d(embedding output) for every tracer token seen in backward passes.

    Positions holding the pad id (0) are not tracers, but they are not ignored either: their gradient
    goes into a padding sink. A correct stack sends exactly zero gradient there; anything else means a
    loss reached padding (or a scored position after EOS), which the per-token weights alone can miss
    when the normaliser does not change."""

    def __init__(self, model):
        self.model = model
        self.grads: dict[int, torch.Tensor] = {}
        self.pad_sq = 0.0  # Σ ||gradient||² over every pad-id position
        self._handle = None

    @property
    def pad_sink(self) -> float:
        """Norm of all the gradient that reached pad-id positions."""
        return self.pad_sq**0.5

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
                    self.pad_sq += float(flat_g[k].norm()) ** 2
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


# ── pair readout: several targets predicted from one input token ───────────
#
# Preference and RL trainers repeat the prompt in every row, so the prompt's last token predicts a
# different first completion token in each row: one input id, several targets. In aggregation mode a
# position's output depends only on its own embedding, so the unit gradient of "input a predicts b"
# is the same wherever it occurs, and the gradient recorded at a is a linear combination of the unit
# gradients of its targets. Solving that small least-squares problem recovers every weight when the
# unit gradients are well conditioned. The final RMSNorm removes one direction, so with hidden size h
# at most h - 1 targets per input are identifiable; the condition number is reported, and the readout
# refuses above MAX_COND rather than return weights it cannot pin down.

MAX_COND = 1e3

Pair = tuple[int, int]  # (input token id, target token id)


def unit_pair_gradients(model, pairs: list[Pair]) -> dict[Pair, torch.Tensor]:
    """d CE(target | input) / d(embedding output of input), each pair scored alone at position 0."""
    pairs = list(dict.fromkeys(pairs))
    ids = torch.tensor([[a] for a, _ in pairs])
    emb = model.get_input_embeddings()(ids).detach().requires_grad_()
    logits = model(inputs_embeds=emb).logits[:, 0]
    F.cross_entropy(logits, torch.tensor([b for _, b in pairs]), reduction="sum").backward()
    model.zero_grad(set_to_none=True)
    return {p: emb.grad[k, 0].detach().clone() for k, p in enumerate(pairs)}


@dataclass
class PairRead:
    weights: dict[Pair, float]
    residual: float  # max over inputs of ||g − Σ w u|| / ||g||: 0 when the gradient is fully explained
    cond: float  # largest condition number of any input's unit-gradient matrix (1 for a single target)


def read_pairs(grads: dict[int, torch.Tensor], unit: dict[Pair, torch.Tensor], pairs: list[Pair]) -> PairRead:
    """CE weight of every (input, target) pair from the gradients recorded at each input token."""
    by_input: dict[int, list[int]] = {}
    for a, b in dict.fromkeys(pairs):
        by_input.setdefault(a, []).append(b)
    weights: dict[Pair, float] = {}
    residual = 0.0
    worst = 1.0
    for a, targets in by_input.items():
        U = torch.stack([unit[(a, b)] for b in targets], dim=1)  # (hidden, k)
        if U.shape[1] > U.shape[0] - 1:
            raise ValueError(f"input {a} has {U.shape[1]} targets; a hidden size of {U.shape[0]} identifies at most "
                             f"{U.shape[0] - 1} (the final RMSNorm removes one direction)")
        c = float(torch.linalg.cond(U)) if U.shape[1] > 1 else 1.0
        if c > MAX_COND:
            raise ValueError(f"input {a}: condition number {c:.3g} > {MAX_COND:g}, weights not identifiable")
        worst = max(worst, c)
        g = grads.get(a, torch.zeros(U.shape[0], dtype=U.dtype))
        w = torch.linalg.lstsq(U, g.unsqueeze(1)).solution[:, 0]
        gn = float(g.norm())
        if gn > 0:
            residual = max(residual, float((g - U @ w).norm()) / gn)
        for b, wb in zip(targets, w.tolist()):
            weights[(a, b)] = wb
    return PairRead(weights=weights, residual=residual, cond=worst)


# ── row readout: the embedding PARAMETER gradient ──────────────────────────
#
# Under data parallelism each rank's activation gradients are local: DDP averages PARAMETER
# gradients in its all-reduce, after backward. To see the cross-rank weighting, read the input
# embedding's parameter gradient instead. Tracer ids are unique, so row t of that gradient is the sum
# of the activation gradients at every position holding token t, after any all-reduce. On one process
# it equals the activation readout exactly; when a token occurs more than once (a duplicated sample),
# the row holds the sum of its occurrences, which is exactly what reaches the parameters.


def embedding_row_grads(model, token_ids) -> dict[int, torch.Tensor]:
    """Rows of the input-embedding parameter gradient for the given tracer ids."""
    g = model.get_input_embeddings().weight.grad
    if g is None:
        raise RuntimeError("the embedding has no gradient: run backward first")
    return {int(t): g[int(t)].detach().clone() for t in token_ids if int(t) != PAD_ID}
