"""
Relation R2: a token's weight must not depend on how the batch is split across data-parallel ranks.
Read from the all-reduced embedding PARAMETER gradient (activation gradients are local to a rank).
"""
import pytest
import torch
import torch.nn.functional as F

from token_xray import intended
from token_xray import runners as R
from token_xray.ddp import ddp_step, hf_trainer_ddp
from token_xray.probe import build_probe, make_batch
from token_xray.readout import embedding_row_grads, read, recording, unit_gradients

LENGTHS = [6, 14, 4, 22, 9, 5, 17, 8, 12, 3, 19, 7]
TOL = 1e-6  # instrument precision floor (see test_xray.py)
EQUAL = [[1, 3, 6, 7], [0, 2, 4, 5, 8, 9, 10, 11]]  # 57 target tokens on each rank
SHORT_LONG = sorted(range(len(LENGTHS)), key=lambda i: LENGTHS[i])
UNEVEN = [SHORT_LONG[:6], SHORT_LONG[6:]]  # 27 vs 87 target tokens


def dev(a, b):
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def rank_of(assignment, n):
    out = [None] * n
    for r, seqs in enumerate(assignment):
        for i in seqs:
            out[i] = r
    return out


@pytest.fixture(scope="module")
def batch():
    return make_batch(LENGTHS)


@pytest.fixture(scope="module")
def unit(batch):
    return unit_gradients(build_probe(batch.vocab_size), batch)


@pytest.fixture(scope="module")
def runs():
    return {
        "world1": ddp_step(LENGTHS, 1),
        "equal": ddp_step(LENGTHS, 2, split=EQUAL),
        "uneven": ddp_step(LENGTHS, 2, split=UNEVEN),
    }


# ── the instrument ──────────────────────────────────────────────────────────


def test_row_readout_agrees_with_activation_readout_on_one_process(batch, unit):
    """Unique tracer ids: each embedding row's gradient is exactly that token's activation gradient."""
    model = build_probe(batch.vocab_size)
    b = R.collate([{"input_ids": s} for s in batch.sequences])
    with recording(model) as rec:
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), b["labels"][:, 1:].reshape(-1),
                        ignore_index=-100, reduction="sum").div(batch.n_targets).backward()
    rows = embedding_row_grads(model, range(1, batch.vocab_size))
    for t, g in rec.grads.items():
        assert torch.allclose(rows[t], g, atol=1e-14)
    assert dev(read(rows, unit, batch).normalized, read(rec.grads, unit, batch).normalized) < 1e-12


# ── relation R2 ─────────────────────────────────────────────────────────────


def test_splits_record_what_they_claim(runs):
    assert runs["equal"].local_targets == [57, 57]
    assert runs["uneven"].local_targets == [27, 87]


@pytest.mark.parametrize("split", ["equal", "uneven"])
def test_global_token_mean_is_identical_for_world_size_1_and_2(runs, batch, unit, split):
    w1 = read(runs["world1"].rows["global_token_mean"], unit, batch).normalized
    w2 = read(runs[split].rows["global_token_mean"], unit, batch).normalized
    assert dev(w1, intended.token_mean(LENGTHS)) < TOL
    assert dev(w2, w1) < TOL


def test_per_rank_mean_breaks_the_relation_when_ranks_are_uneven(runs, batch, unit):
    """The injected fault: average of per-rank token means. Caught, and matches its closed form."""
    w = read(runs["uneven"].rows["per_rank_mean"], unit, batch).normalized
    w1 = read(runs["world1"].rows["global_token_mean"], unit, batch).normalized
    assert dev(w, w1) > 0.5
    predicted = intended.per_rank_mean(LENGTHS, rank_of(runs["uneven"].assignment, len(LENGTHS)), 2)
    assert dev(w, predicted) < TOL


def test_per_rank_mean_is_invisible_when_ranks_hold_equal_tokens(runs, batch, unit):
    """Honest limit: with equal token counts per rank the two reductions coincide exactly."""
    w = read(runs["equal"].rows["per_rank_mean"], unit, batch).normalized
    assert dev(w, intended.token_mean(LENGTHS)) < TOL


def test_distributed_sampler_padding_duplicates_a_sample():
    """DistributedSampler (drop_last=False) pads 11 samples to 12 by repeating sample 0 on the other
    rank. The repeat carries the same tracer ids, so its embedding rows hold the sum of both copies:
    sample 0 is weighted twice, and every other token loses a share of the normaliser."""
    L = LENGTHS[:11]
    b = make_batch(L)
    unit = unit_gradients(build_probe(b.vocab_size), b)
    r = ddp_step(L, 2, sampler=True, reductions=("global_token_mean",))
    copies = [sum(s.count(i) for s in r.assignment) for i in range(len(L))]
    assert copies == [2] + [1] * 10
    w = read(r.rows["global_token_mean"], unit, b).normalized
    assert dev(w, intended.token_mean(L)) > 0.5  # the step is not a token mean over the dataset
    assert dev(w, intended.with_multiplicity(L, copies)) < TOL  # it is one over what the ranks saw


# ── the real transformers Trainer under torchrun ───────────────────────────


@pytest.fixture(scope="module")
def trainer_runs():
    return {avg: hf_trainer_ddp(LENGTHS, per_device_batch=6, average_tokens_across_devices=avg) for avg in (True, False)}


def test_trainer_ddp_averages_tokens_across_ranks_by_default(trainer_runs, batch, unit):
    run = trainer_runs[True]
    assert run["args"] == {"average_tokens_across_devices": True, "world_size": 2}
    assert sorted(i for s in run["seen"] for i in s) == list(range(len(LENGTHS)))  # every sample once
    assert dev(read(run["rows"], unit, batch).normalized, intended.token_mean(LENGTHS)) < TOL


def test_trainer_ddp_without_token_averaging_is_a_per_rank_mean(trainer_runs, batch, unit):
    run = trainer_runs[False]
    assert run["args"]["average_tokens_across_devices"] is False
    w = read(run["rows"], unit, batch).normalized
    predicted = intended.per_rank_mean(LENGTHS, rank_of(run["seen"], len(LENGTHS)), 2)
    assert dev(w, predicted) < TOL
    assert dev(w, intended.token_mean(LENGTHS)) > 1e-3  # the Trainer's own split is uneven
