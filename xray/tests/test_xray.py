"""
Every claim the project makes is a test here. Each check must pass on correct code AND fail on the
failure it claims to detect, otherwise it is not evidence.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from token_xray import intended, xray
from token_xray import runners as R
from token_xray.leak import leak_matrix, max_leak
from token_xray.probe import build_probe, make_batch
from token_xray.readout import read, unit_gradients
from token_xray.runners import collate
from token_xray.scenarios import custom_loop_leak

LENGTHS = [5, 9, 3, 12, 7, 4, 10, 6]
# Precision floor of the instrument: LLaMA RMSNorm (and HF's loss) compute in float32 even in a
# float64 model, so recovered weights carry ~1e-7 relative error. Null runs confirm it.
TOL = 1e-6


@pytest.fixture(scope="module")
def batch():
    return make_batch(LENGTHS)


@pytest.fixture(scope="module")
def unit(batch):
    return unit_gradients(build_probe(batch.vocab_size), batch)


def dev(res, unit, batch, want):
    got = read(res.grads, unit, batch).normalized
    return max(abs(g - w) for gr, wr in zip(got, want) for g, w in zip(gr, wr))


# ── the instrument ──────────────────────────────────────────────────────────


def _cross_document_gradient(mode: str) -> float:
    """Two documents in ONE row, no mask, no position reset: the loss of document B (second), and the
    gradient that reaches document A's positions. Only isolation inside the probe can make it zero."""
    a, bdoc = [1, 2, 3, 4, 5], [6, 7, 8, 9]
    m = build_probe(10, mode)
    ids = torch.tensor([a + bdoc])
    emb = m.get_input_embeddings()(ids).detach().requires_grad_()
    logits = m(inputs_embeds=emb).logits[0]
    start = len(a)
    F.cross_entropy(logits[start:-1], ids[0, start + 1 :], reduction="sum").backward()
    return float(emb.grad[0, :start].abs().max())


def test_probe_isolates_positions():
    """Aggregation mode: B's loss sends exactly zero gradient into A, although A is in B's attention window."""
    assert _cross_document_gradient("aggregation") == 0.0


def test_probe_isolation_control_leaks_with_attention_live():
    """Negative control: the same row with attention live must leak, or the check above proves nothing."""
    assert _cross_document_gradient("leak") > 1e-3


def test_reference_is_exact(batch, unit):
    assert dev(R.reference(batch, 2), unit, batch, intended.token_mean(LENGTHS)) < TOL


def test_readout_recovers_known_weights(batch, unit):
    """Plant arbitrary per-token weights in a hand-written loss and read them back."""
    rng = np.random.default_rng(1)
    planted = [list(rng.uniform(0.1, 3.0, L - 1)) for L in LENGTHS]
    m = build_probe(batch.vocab_size)
    from token_xray.readout import recording

    with recording(m) as rec:
        for s, w in zip(batch.sequences, planted):
            ids = torch.tensor([s])
            ce = F.cross_entropy(m(input_ids=ids).logits[0, :-1], ids[0, 1:], reduction="none")
            (ce * torch.tensor(w, dtype=ce.dtype)).sum().div(batch.n_targets).backward()
    got = read(rec.grads, unit, batch).normalized
    assert max(abs(g - p) for gr, pr in zip(got, planted) for g, p in zip(gr, pr)) < TOL


# ── real library runs ───────────────────────────────────────────────────────


def test_hf_trainer_gradient_accumulation_matches_token_mean(batch, unit):
    tm = intended.token_mean(LENGTHS)
    assert dev(R.hf_trainer(batch, 8, 1), unit, batch, tm) < TOL
    res = R.hf_trainer(batch, 2, 4)
    assert dev(res, unit, batch, tm) < TOL
    assert res.info["pad_sink"] == 0.0  # no loss reached a padding position


def test_hf_trainer_order_invariant(batch, unit):
    tm = intended.token_mean(LENGTHS)
    for seed in (1, 2):
        assert dev(R.hf_trainer(batch, 2, 4, shuffle=True, seed=seed), unit, batch, tm) < TOL


def test_hf_trainer_logged_loss_equals_objective(batch):
    ref = R.reference(batch, 8).logged_loss
    assert abs(R.hf_trainer(batch, 2, 4).logged_loss - ref) < 1e-4 * ref


def test_left_padding_adds_one_target_per_padded_sequence(batch, unit):
    """labels = input_ids + left padding: the pad before each shorter sequence predicts its first token."""
    res = R.hf_trainer(batch, 8, 1, padding_side="left")
    extra = sum(1 for L in LENGTHS if L < max(LENGTHS))
    want = [[batch.n_targets / (batch.n_targets + extra)] * (L - 1) for L in LENGTHS]
    assert dev(res, unit, batch, want) < TOL


def test_label_smoothing_is_flagged_by_cosine(batch, unit):
    x = read(R.hf_trainer(batch, 8, 1, label_smoothing=0.1).grads, unit, batch)
    assert x.min_cos < 1 - 1e-7
    assert read(R.hf_trainer(batch, 8, 1).grads, unit, batch).min_cos > 1 - 1e-9


# ── emulated historical mechanisms: detected, and exactly as predicted ─────


@pytest.mark.parametrize(
    "run,predict",
    [
        (lambda b: R.emulate_mean_of_means(b, 2), lambda L: intended.mean_of_microbatch_means(L, 2)),
        (lambda b: R.emulate_dropped_microbatches(b, 2), lambda L: intended.first_microbatch_only(L, 2)),
    ],
    ids=["mean-of-means", "dropped-microbatches"],
)
def test_emulated_bug_is_detected_and_predicted(batch, unit, run, predict):
    res = run(batch)
    assert dev(res, unit, batch, intended.token_mean(LENGTHS)) > 0.05  # detected
    assert dev(res, unit, batch, predict(LENGTHS)) < TOL  # and matches the closed form


def test_library_collator_with_pad_equal_to_eos_removes_eos_targets(batch):
    """Real run: transformers' DataCollatorForLanguageModeling with pad token = EOS token."""
    eb = R.with_eos(batch)
    eunit = unit_gradients(build_probe(eb.vocab_size), eb)
    res = R.hf_trainer_pad_equals_eos(batch)
    got = read(res.grads, eunit, eb).normalized
    lengths = [L + 1 for L in LENGTHS]
    assert all(abs(row[-1]) < 1e-12 for row in got)  # every sequence's EOS target gets zero weight
    pred = intended.labels_equal_to_pad_masked(lengths)
    assert max(abs(g - w) for gr, wr in zip(got, pred) for g, w in zip(gr, wr)) < TOL
    assert res.info["pad_sink"] == 0.0


# ── public API ──────────────────────────────────────────────────────────────


def _custom_loop(buggy: bool, micro: int = 2):
    def step(model, sequences):
        n_total = sum(len(s) - 1 for s in sequences)
        k = len(sequences) // micro
        for i in range(0, len(sequences), micro):
            b = collate([{"input_ids": s} for s in sequences[i : i + micro]])
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"])
            if buggy:
                (out.loss / k).backward()
            else:
                n_mb = int((b["labels"][:, 1:] != -100).sum())
                (out.loss * n_mb / n_total).backward()

    return step


def test_api_passes_correct_custom_loop():
    xray(_custom_loop(buggy=False), LENGTHS).assert_token_mean()


def test_api_catches_buggy_custom_loop():
    with pytest.raises(AssertionError):
        xray(_custom_loop(buggy=True), LENGTHS).assert_token_mean()


# ── leaks ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def leak_batch():
    return make_batch([5, 7, 4, 6])


def test_padded_rows_never_leak(leak_batch):
    assert max_leak(leak_matrix(leak_batch, "padded")) == 0.0


def test_naive_packing_leaks_only_forward_in_time(leak_batch):
    M = leak_matrix(leak_batch, "packed_naive")
    assert max_leak(M) > 1e-3
    assert np.allclose(np.triu(M, 1), 0.0)  # causal: later documents leak into earlier ones, never the reverse


def test_packed_position_ids_without_cache_do_not_leak(leak_batch):
    assert max_leak(custom_loop_leak(leak_batch, use_cache=False)) == 0.0


def test_packed_position_ids_with_default_cache_leak(leak_batch):
    """Documents the known issue (transformers#50419) for the installed transformers version."""
    assert max_leak(custom_loop_leak(leak_batch, use_cache=None)) > 1e-3
