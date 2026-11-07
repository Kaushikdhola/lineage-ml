"""
Oracle for src/core/aggregation.ts.

Independent method: build each loss as an explicit function of per-token log-probabilities and
differentiate it NUMERICALLY (central differences). The TypeScript side uses closed forms, so the
two can only agree if the closed forms are right.

    python python/oracle_aggregation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "aggregation.json"


def num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e.flat[i] = h
        g.flat[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def flat_index(lengths):
    idx, k = [], 0
    for L in lengths:
        idx.append(list(range(k, k + L)))
        k += L
    return idx, k


def lm_weights(lengths, loss_kind, micro=None):
    """Weights x N from d(loss)/d(per-token CE) for language-model aggregation schemes."""
    n = [L - 1 for L in lengths]
    idx, T = flat_index(n)
    N = sum(n)

    def loss(ce):
        if loss_kind == "token_mean":
            return ce.sum() / N
        if loss_kind == "sequence_mean":
            return np.mean([ce[i].mean() for i in idx])
        groups = [list(range(j, min(j + micro, len(n)))) for j in range(0, len(n), micro)]
        if loss_kind == "mean_of_means":
            return np.mean([np.concatenate([ce[idx[i]] for i in g]).mean() for g in groups])
        if loss_kind == "first_only":
            return np.concatenate([ce[idx[i]] for i in groups[0]]).sum() / N
        raise ValueError(loss_kind)

    g = num_grad(loss, np.random.default_rng(0).uniform(0.5, 3.0, T)) * N
    return [g[i].tolist() for i in idx]


def group_weights(lengths, rewards, objective, eps=1e-4):
    r = np.asarray(rewards, float)
    G = len(lengths)
    adv = r - r.mean() if objective == "dr-grpo" else (r - r.mean()) / (r.std(ddof=1) + eps)
    idx, T = flat_index(lengths)
    Lmax = max(lengths)

    def loss(logp):
        ratio = np.exp(logp - logp_old)  # = 1 at the evaluation point
        if objective == "grpo":
            return -np.mean([(adv[i] * ratio[idx[i]]).mean() for i in range(G)])
        if objective == "dapo":
            return -sum((adv[i] * ratio[idx[i]]).sum() for i in range(G)) / T
        return -sum((adv[i] * ratio[idx[i]]).sum() for i in range(G)) / (G * Lmax)

    logp_old = np.random.default_rng(1).uniform(-3, -0.1, T)
    g = num_grad(loss, logp_old.copy())
    return {"advantages": adv.tolist(), "weights": [g[i].tolist() for i in idx]}


if __name__ == "__main__":
    lengths = [6, 14, 4, 22, 9, 5, 17, 8]
    group = {"lengths": [12, 40, 7, 25, 60, 18], "rewards": [1, 0, 1, 1, 0, 0]}
    data = {
        "lm": {
            "lengths": lengths,
            "micro": 3,
            "token_mean": lm_weights(lengths, "token_mean"),
            "sequence_mean": lm_weights(lengths, "sequence_mean"),
            "mean_of_means": lm_weights(lengths, "mean_of_means", 3),
            "first_only": lm_weights(lengths, "first_only", 3),
        },
        "group": {**group, **{o: group_weights(group["lengths"], group["rewards"], o) for o in ["grpo", "dr-grpo", "dapo"]}},
    }
    OUT.write_text(json.dumps(data))
    print(f"oracle_aggregation: wrote {OUT.relative_to(ROOT)}")
