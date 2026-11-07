"""
Intended weights: what the equation says each token should receive (closed form).

All functions return per-sequence lists of weights for target tokens (length L_i - 1 each),
already multiplied by N (total target tokens), so a correct global token mean is 1.0 everywhere.
"""
from __future__ import annotations


def _targets(lengths: list[int]) -> list[int]:
    return [L - 1 for L in lengths]


def token_mean(lengths: list[int]) -> list[list[float]]:
    """Global mean over all target tokens in the effective batch: w = 1/N."""
    return [[1.0] * n for n in _targets(lengths)]


def sequence_mean(lengths: list[int]) -> list[list[float]]:
    """Mean over tokens within each sequence, then over sequences: w = 1/(B * n_i)."""
    n = _targets(lengths)
    N, B = sum(n), len(n)
    return [[N / (B * ni)] * ni for ni in n]


def mean_of_microbatch_means(lengths: list[int], micro: int) -> list[list[float]]:
    """Token mean inside each micro-batch, then averaged over micro-batches (sequential split)."""
    n = _targets(lengths)
    N = sum(n)
    groups = [n[i : i + micro] for i in range(0, len(n), micro)]
    k = len(groups)
    out = []
    for g in groups:
        Ng = sum(g)
        out += [[N / (k * Ng)] * ni for ni in g]
    return out


def first_microbatch_only(lengths: list[int], micro: int) -> list[list[float]]:
    """Only the first micro-batch survives; its tokens keep the global normalizer."""
    n = _targets(lengths)
    return [[1.0 if i < micro else 0.0] * ni for i, ni in enumerate(n)]
