"""
Public API: X-ray your own training step.

    from token_xray import xray

    def my_step(model, sequences):
        # build batches exactly as your trainer does, run forward + backward (no optimizer step needed)
        ...

    report = xray(my_step, lengths=[5, 9, 3, 12, 7, 4, 10, 6])
    report.assert_token_mean()          # every target token got weight 1/N
    print(report.table())

The probe model is a tiny LLaMA whose token ids are unique tracers. `sequences` is a list of lists of
token ids; treat them like tokenized documents. Labels follow the usual convention (labels = input ids,
shifted inside the loss).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import intended
from .probe import build_probe, make_batch
from .readout import XRay, read, recording, unit_gradients

Step = Callable[[object, list[list[int]]], None]


@dataclass
class Report:
    xray: XRay
    lengths: list[int]

    def deviation_from(self, want: list[list[float]]) -> float:
        got = self.xray.normalized
        return max(abs(g - w) for gr, wr in zip(got, want) for g, w in zip(gr, wr))

    def assert_token_mean(self, tol: float = 1e-6) -> None:
        dev = self.deviation_from(intended.token_mean(self.lengths))
        if dev > tol:
            raise AssertionError(
                f"token weights deviate from a global token mean by up to {dev:.3g} (tol {tol:g}).\n" + self.table()
            )

    def assert_pure_rescaling(self, tol: float = 1e-9) -> None:
        if self.xray.min_cos < 1 - tol:
            raise AssertionError(f"gradient is not a pure rescaling of each token's own gradient (min cos {self.xray.min_cos:.9f})")

    def table(self) -> str:
        lines = ["seq  len   mean(w*N)   min(w*N)   max(w*N)"]
        for i, row in enumerate(self.xray.normalized):
            a = np.array(row)
            lines.append(f"{i:3d} {self.lengths[i]:4d}   {a.mean():9.4f}  {a.min():9.4f}  {a.max():9.4f}")
        return "\n".join(lines)


def xray(step: Step, lengths: list[int], seed: int = 0) -> Report:
    batch = make_batch(lengths)
    unit = unit_gradients(build_probe(batch.vocab_size, seed=seed), batch)
    model = build_probe(batch.vocab_size, seed=seed)
    with recording(model) as rec:
        step(model, [list(s) for s in batch.sequences])
    return Report(xray=read(rec.grads, unit, batch), lengths=lengths)
