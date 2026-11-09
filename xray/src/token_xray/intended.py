"""
Intended weights: what the equation says each token should receive (closed form).

All functions return per-sequence lists of weights for target tokens (length L_i - 1 each),
already multiplied by N (total target tokens), so a correct global token mean is 1.0 everywhere.
"""
from __future__ import annotations

import math


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


# ── preference and RL objectives ───────────────────────────────────────────
#
# These return the raw weight on each completion token's cross-entropy, w = −∂L/∂log π(token)
# (positive = the token is reinforced), not multiplied by N. Rows are completions, in batch order.


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def dpo_sigmoid(
    chosen_lens: list[int], rejected_lens: list[int], deltas: list[float], beta: float
) -> tuple[list[list[float]], list[list[float]]]:
    """DPO (Rafailov et al., 2023), sigmoid loss averaged over P pairs:

        L = −1/P Σ_i log σ(β Δ_i),   Δ_i = (log π_c − log π_ref,c) − (log π_r − log π_ref,r)

    with sequence log-probs summed over completion tokens, so every token of a completion gets the
    same weight: +β σ(−β Δ_i) / P on the chosen tokens, −β σ(−β Δ_i) / P on the rejected ones.
    """
    P = len(deltas)
    if not (len(chosen_lens) == len(rejected_lens) == P):
        raise ValueError("one chosen length, one rejected length and one margin per pair")
    k = [beta * _sigmoid(-beta * d) / P for d in deltas]
    return [[k[i]] * n for i, n in enumerate(chosen_lens)], [[-k[i]] * n for i, n in enumerate(rejected_lens)]


def group_advantages(rewards: list[float], scale: bool = True, eps: float = 1e-4) -> list[float]:
    """GRPO advantages for one group: (r − mean) / (std + eps), std with ddof = 1 (torch default)."""
    G = len(rewards)
    m = sum(rewards) / G
    c = [r - m for r in rewards]
    if not scale:
        return c
    sd = math.sqrt(sum(x * x for x in c) / (G - 1))
    return [x / (sd + eps) for x in c]


def group_policy_gradient(
    lengths: list[int],
    rewards: list[float],
    loss_type: str,
    ratios: list[list[float]] | None = None,
    epsilon_low: float = 0.2,
    epsilon_high: float | None = None,
    max_completion_length: int | None = None,
    scale_rewards: bool = True,
) -> tuple[list[float], list[list[float]]]:
    """One group of G completions, one optimizer step, KL off (beta = 0), clipped surrogate

        ℓ_t = −min(ρ_t A_i, clip(ρ_t, 1 − ε_low, 1 + ε_high) A_i)

    reduced as the installed TRL does for each loss_type (single process, no accumulation):
        grpo    : mean over completions of the per-completion token mean   (norm G · |o_i|)
        bnpo    : token mean over the batch                                (norm Σ_j |o_j|)
        dapo    : token mean over the generation batch                     (norm Σ_j |o_j|)
        dr_grpo : sum over tokens / (G · max_completion_length)            (norm G · L_max)

    Weight on token t of completion i: A_i ρ_t / norm, or 0 where the clipped branch is selected
    (A > 0 and ρ > 1 + ε_high, or A < 0 and ρ < 1 − ε_low). Returns (advantages, weights).
    """
    G = len(lengths)
    eh = epsilon_low if epsilon_high is None else epsilon_high
    A = group_advantages(rewards, scale=scale_rewards)
    total = sum(lengths)
    rho = ratios if ratios is not None else [[1.0] * L for L in lengths]
    out = []
    for i, L in enumerate(lengths):
        if loss_type == "grpo":
            norm = G * L
        elif loss_type in ("bnpo", "dapo"):
            norm = total
        elif loss_type == "dr_grpo":
            if max_completion_length is None:
                raise ValueError("dr_grpo normalises by max_completion_length")
            norm = G * max_completion_length
        else:
            raise ValueError(f"unsupported loss_type {loss_type!r}")
        row = []
        for r in rho[i]:
            clipped = (A[i] > 0 and r > 1 + eh) or (A[i] < 0 and r < 1 - epsilon_low)
            row.append(0.0 if clipped else A[i] * r / norm)
        out.append(row)
    return A, out
