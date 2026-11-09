"""
Real TRL runs on the tracer probe. Each test checks a measured claim against its closed form; the
closed forms are themselves checked against autograd on an independently written loss.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from token_xray import intended
from token_xray.leak import max_leak
from token_xray.probe import build_probe, make_batch
from token_xray.readout import read, read_pairs, recording, unit_gradients, unit_pair_gradients
from token_xray.trl_runners import (
    dpo_batch,
    dpo_trainer,
    grpo_batch,
    grpo_trainer,
    probe_at,
    ratios_between,
    sft_leak_matrix,
    sft_trainer,
)

trl = pytest.importorskip("trl")

LENGTHS = [5, 9, 3, 12, 7, 4, 10, 6]
TOL = 1e-6  # instrument precision floor (see test_xray.py)
REL = 1e-5  # relative tolerance for the raw (un-normalised) preference and RL weights


# ── the instrument: several targets behind one input token ─────────────────


def test_pair_unit_gradient_equals_in_sequence_unit_gradient():
    """Aggregation mode is position-independent: a pair scored alone equals the same pair inside a sequence."""
    b = make_batch([6])
    m = build_probe(b.vocab_size)
    seq = b.sequences[0]
    inside = unit_gradients(m, b)
    alone = unit_pair_gradients(m, list(zip(seq[:-1], seq[1:])))
    for a, t in zip(seq[:-1], seq[1:]):
        assert torch.allclose(alone[(a, t)], inside[a], atol=1e-12)


def test_pair_readout_recovers_planted_weights_on_a_shared_token():
    """One input token predicting three different targets in three rows, with planted weights."""
    m = build_probe(12)
    rows = [[1, 2, 3], [1, 4, 5, 6], [1, 7]]  # token 1 is shared, like a prompt's last token
    planted = [[0.7, -1.3], [2.1, 0.4, -0.9], [-0.25]]
    with recording(m) as rec:
        for r, w in zip(rows, planted):
            ids = torch.tensor([r])
            ce = F.cross_entropy(m(input_ids=ids).logits[0, :-1], ids[0, 1:], reduction="none")
            (ce * torch.tensor(w, dtype=ce.dtype)).sum().backward()
    pairs = [p for r in rows for p in zip(r[:-1], r[1:])]
    pr = read_pairs(rec.grads, unit_pair_gradients(build_probe(12), pairs), pairs)
    got = [[pr.weights[p] for p in zip(r[:-1], r[1:])] for r in rows]
    # same precision floor as the single-target readout (float32 RMSNorm inside the float64 probe)
    assert max(abs(g - w) for gr, wr in zip(got, planted) for g, w in zip(gr, wr)) < TOL
    assert pr.residual < TOL


# ── closed forms against autograd ──────────────────────────────────────────


def test_dpo_closed_form_matches_autograd():
    g = torch.Generator().manual_seed(3)
    beta, lens_c, lens_r = 0.3, [4, 2, 5], [3, 6, 1]
    lc = [torch.randn(n, generator=g, dtype=torch.float64).requires_grad_() for n in lens_c]
    lr = [torch.randn(n, generator=g, dtype=torch.float64).requires_grad_() for n in lens_r]
    rc, rr = torch.randn(3, generator=g, dtype=torch.float64), torch.randn(3, generator=g, dtype=torch.float64)
    delta = torch.stack([(c.sum() - rc[i]) - (r.sum() - rr[i]) for i, (c, r) in enumerate(zip(lc, lr))])
    (-F.logsigmoid(beta * delta)).mean().backward()
    ch, rj = intended.dpo_sigmoid(lens_c, lens_r, delta.tolist(), beta)
    for t, w in zip(lc + lr, ch + rj):  # weight on CE = −∂L/∂log π
        assert torch.allclose(-t.grad, torch.tensor(w, dtype=torch.float64), atol=1e-14)


@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dapo", "dr_grpo"])
@pytest.mark.parametrize("scale", [True, False])
def test_group_policy_gradient_matches_autograd(loss_type, scale):
    """Clipped surrogate with ratios on both sides of the clip range, reduced as TRL reduces each loss_type."""
    g = torch.Generator().manual_seed(7)
    lengths, rewards, eps, Lmax = [3, 6, 2, 5], [1.0, 0.0, 0.4, 0.9], 0.2, 8
    logp = [torch.randn(L, generator=g, dtype=torch.float64).requires_grad_() for L in lengths]
    old = [lp.detach() + 0.3 * torch.randn(L, generator=g, dtype=torch.float64) for lp, L in zip(logp, lengths)]
    r = torch.tensor(rewards, dtype=torch.float64)
    A = r - r.mean()
    if scale:
        A = A / (r.std() + 1e-4)
    per = []
    for i in range(len(lengths)):
        rho = torch.exp(logp[i] - old[i])
        per.append(-torch.min(rho * A[i], torch.clamp(rho, 1 - eps, 1 + eps) * A[i]))
    G, total = len(lengths), sum(lengths)
    if loss_type == "grpo":
        loss = torch.stack([p.mean() for p in per]).mean()
    elif loss_type in ("bnpo", "dapo"):
        loss = torch.cat(per).sum() / total
    else:
        loss = torch.cat(per).sum() / (G * Lmax)
    loss.backward()
    ratios = [torch.exp(lp - o).tolist() for lp, o in zip(logp, old)]
    adv, want = intended.group_policy_gradient(lengths, rewards, loss_type, ratios=ratios, epsilon_low=eps,
                                               max_completion_length=Lmax, scale_rewards=scale)
    assert np.allclose(adv, A.tolist(), atol=1e-12)
    assert any(w == 0.0 for row in want for w in row)  # the clipped branch is exercised
    for lp, w in zip(logp, want):
        assert torch.allclose(-lp.grad, torch.tensor(w, dtype=torch.float64), atol=1e-14)


# ── SFT ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def batch():
    return make_batch(LENGTHS)


@pytest.fixture(scope="module")
def unit(batch):
    return unit_gradients(build_probe(batch.vocab_size), batch)


def _dev(res, unit, batch):
    got = read(res.grads, unit, batch).normalized
    return max(abs(g - w) for gr, wr in zip(got, intended.token_mean(LENGTHS)) for g, w in zip(gr, wr))


@pytest.mark.parametrize("kw", [dict(padding_free=True), dict(packing=True, max_length=24)], ids=["padding-free", "packing"])
def test_sft_matches_token_mean_across_accumulation(batch, unit, kw):
    res = sft_trainer(batch, 2, 4, **kw)
    assert res.info["padding_free"] is True  # bfd packing turns padding-free on
    assert all(mb["shape"][0] == 1 and "position_ids" in mb["keys"] for mb in res.info["micro_batches"])
    assert _dev(res, unit, batch) < TOL


# bf16 keeps 8 significant bits: unit roundoff u = 2^-8 ≈ 3.9e-3.
BF16_U = 2.0**-8


def bf16_within_rounding(got) -> bool:
    """Rounding, not re-weighting. Per token: |w·N − 1| ≤ 2u (one bf16 rounding forward, one backward).
    Per sequence: |mean(w·N) − 1| ≤ u. The bound does not shrink with the number of targets: the measured
    errors within a sequence are correlated (a 6-target mean moved 2.09e-3, above the u/√6 = 1.6e-3 that
    independent errors allow), so it uses the worst case, a mean no larger than one token's error. A 1%
    re-weighting moves a sequence's mean by 1e-2, 2.6× this bound."""
    per_token = max(abs(w - 1.0) for row in got for w in row)
    per_seq = max(abs(sum(row) / len(row) - 1.0) for row in got)
    return per_token <= 2 * BF16_U and per_seq <= BF16_U


def test_sft_bf16_default_rounds_but_does_not_reweight(batch, unit):
    """TRL defaults to bf16 mixed precision. The weights are visibly rounded but stay within the bf16
    bounds above, and the same check rejects a 1% re-weighting of the same measurement."""
    got = read(sft_trainer(batch, 2, 4, padding_free=True, bf16=True).grads, unit, batch).normalized
    assert max(abs(w - 1.0) for row in got for w in row) > 1e-5  # rounding is visible
    assert bf16_within_rounding(got)
    reweighted = [[w * 1.01 for w in row] if i == 0 else row for i, row in enumerate(got)]
    assert not bf16_within_rounding(reweighted)


@pytest.fixture(scope="module")
def leak_batch():
    return make_batch([5, 7, 4, 6])


@pytest.mark.parametrize("kw", [dict(packing=False), dict(packing=True, max_length=64)], ids=["padding-free", "packing"])
def test_sft_packed_documents_do_not_leak(leak_batch, kw):
    assert max_leak(sft_leak_matrix(leak_batch, **kw)) == 0.0


def test_sft_leak_control_detects_a_collator_that_does_not_reset_positions(leak_batch, monkeypatch):
    """Negative control: with position_ids that run on across documents, the same harness must leak."""
    import trl.trainer.sft_trainer as sft

    orig = sft.DataCollatorForLanguageModeling.torch_call

    def no_reset(self, examples):
        out = orig(self, examples)
        n = out["position_ids"].shape[1]
        out["position_ids"] = torch.arange(n).unsqueeze(0).expand_as(out["position_ids"]).clone()
        return out

    monkeypatch.setattr(sft.DataCollatorForLanguageModeling, "torch_call", no_reset)
    assert max_leak(sft_leak_matrix(leak_batch)) > 1e-3


# ── padding sink and identifiability ───────────────────────────────────────


def test_padding_sink_catches_a_loss_on_positions_after_eos():
    """Planted: a Dr. GRPO-style loss (constant normaliser) that also scores the EOS padding (EOS -> EOS).
    The real tokens' weights are unchanged, so only the padding sink can reveal it."""
    from token_xray.trl_runners import EOS_ID

    def run(score_padding: bool):
        m = build_probe(12)
        rows = [[1, 2, 3, EOS_ID, EOS_ID], [4, 5, EOS_ID, EOS_ID, EOS_ID]]  # completions padded with EOS
        real = [3, 2]  # scored positions per row (up to and including the prediction of the first EOS)
        with recording(m) as rec:
            ids = torch.tensor(rows)
            logp = torch.log_softmax(m(input_ids=ids).logits[:, :-1], -1).gather(2, ids[:, 1:, None])[..., 0]
            mask = torch.zeros_like(logp)
            for i, n in enumerate(real):
                mask[i, : (logp.shape[1] if score_padding else n)] = 1
            (-(logp * mask).sum() / (2 * 8)).backward()  # constant normaliser, as dr_grpo
        return rec

    clean, faulty = run(False), run(True)
    for t in clean.grads:  # identical weights on every real token
        assert torch.allclose(clean.grads[t], faulty.grads[t], atol=1e-15)
    assert clean.pad_sink == 0.0
    assert faulty.pad_sink > 1e-3


def test_pair_readout_refuses_unidentifiable_systems():
    """With hidden size h the final RMSNorm leaves h - 1 directions: h targets per input must be refused."""
    m = build_probe(40, hidden_size=16)
    pairs = [(1, b) for b in range(2, 18)]  # 16 targets for one input
    with pytest.raises(ValueError, match="identifies at most 15"):
        read_pairs({1: torch.zeros(16, dtype=torch.float64)}, unit_pair_gradients(m, pairs), pairs)
    ok = pairs[:4]
    pr = read_pairs({1: torch.zeros(16, dtype=torch.float64)}, unit_pair_gradients(m, ok), ok)
    assert 1.0 <= pr.cond < 10


# ── DPO ─────────────────────────────────────────────────────────────────────


def test_dpo_trainer_applies_the_sigmoid_closed_form():
    pb = dpo_batch(3, [4, 6, 3], [5, 3, 7])
    res = dpo_trainer(pb, beta=0.1)
    assert res["loss_type"] == ["sigmoid"]
    rows = [pb.pairs(k, pb.prompts[k // 2]) for k in range(len(pb.completions))]
    allp = [p for r in rows for p in r]
    pr = read_pairs(res["grads"], unit_pair_gradients(pb.probe(), allp), allp)
    got = [[pr.weights[p] for p in r] for r in rows]
    ch, rj = intended.dpo_sigmoid([len(c) for c in pb.completions[0::2]], [len(c) for c in pb.completions[1::2]],
                                  res["deltas"], 0.1)
    want = [x for pair in zip(ch, rj) for x in pair]
    scale = max(abs(w) for r in want for w in r)
    assert max(abs(g - w) for gr, wr in zip(got, want) for g, w in zip(gr, wr)) < REL * scale
    assert pr.residual < 1e-5 and pr.cond < 10
    assert all(w > 0 for w in sum(want[0::2], [])) and all(w < 0 for w in sum(want[1::2], []))
    assert res["pad_sink"] == 0.0  # padding and the EOS's own position received nothing


# ── GRPO ────────────────────────────────────────────────────────────────────

GRPO_LENGTHS, GRPO_REWARDS = [3, 7, 4, 10], [1.0, 0.2, 0.6, 0.0]


def _grpo_check(loss_type=None, scale_rewards=None):
    pb = grpo_batch(3, GRPO_LENGTHS)
    res = grpo_trainer(pb, GRPO_REWARDS, loss_type=loss_type, scale_rewards=scale_rewards, learning_rate=0.05)
    cfg = res["config"]
    rows = [pb.pairs(k, pb.prompts[0]) for k in range(len(pb.completions))]
    allp = [p for r in rows for p in r]
    scored = [len(c) for c in pb.completions]
    out = []
    for st in res["steps"]:
        pr = read_pairs(st["grads"], unit_pair_gradients(probe_at(st["state"], pb), allp), allp)
        got = [[pr.weights[p] for p in r] for r in rows]
        ratios = ratios_between(pb, st["state"], res["steps"][0]["state"])
        _, want = intended.group_policy_gradient(scored, GRPO_REWARDS, cfg["loss_type"], ratios=ratios,
                                                 epsilon_low=cfg["epsilon"], epsilon_high=cfg["epsilon_high"],
                                                 max_completion_length=cfg["max_completion_length"],
                                                 scale_rewards=cfg["scale_rewards"] != "none")
        scale = max(abs(w) for r in want for w in r)
        dev = max(abs(g - w) for gr, wr in zip(got, want) for g, w in zip(gr, wr))
        assert st["pad_sink"] == 0.0 and pr.cond < 10
        out.append((dev / scale, pr.residual, sum(1 for r in want for w in r if w == 0.0), ratios))
    return cfg, out


def test_grpo_default_loss_type_and_both_iterations_match():
    cfg, steps = _grpo_check()
    assert cfg["loss_type"] == "dapo"  # the installed default, recorded on the film
    assert len(steps) == 2
    (d1, r1, c1, rho1), (d2, r2, c2, rho2) = steps
    assert d1 < REL and r1 < 1e-5 and c1 == 0 and all(abs(x - 1) < 1e-12 for r in rho1 for x in r)
    assert d2 < REL and r2 < 1e-5 and c2 > 0  # ratios moved and the clip zeroed some tokens, as predicted


@pytest.mark.parametrize("loss_type,scale", [("grpo", None), ("dr_grpo", "none")])
def test_grpo_loss_types_match(loss_type, scale):
    cfg, steps = _grpo_check(loss_type, scale)
    assert cfg["loss_type"] == loss_type
    assert all(d < REL and r < 1e-5 for d, r, _, _ in steps)
