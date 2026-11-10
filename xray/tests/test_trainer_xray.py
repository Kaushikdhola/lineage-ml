"""
xray_trainer: one line that replays a configured Trainer on the probe. Each test passes on a correct setup
and fails on the misconfiguration it names (scripts/mutate.py breaks the replay itself to prove it).
"""
import tempfile

import pytest
import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from token_xray import build_probe, xray_trainer
from token_xray.trainer_xray import tracer_start
from token_xray.trl_runners import tracer_tokenizer

TOL = 1e-6  # instrument precision floor (see test_xray.py)
VOCAB = 400


def _trainer(cls=Trainer, collator=None, **kw):
    tok = tracer_tokenizer(VOCAB)
    args = dict(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=3, gradient_accumulation_steps=4,
                report_to=[], use_cpu=True)
    args.update(kw)
    # the user's model does not matter: the replay swaps in the probe
    return cls(model=build_probe(VOCAB), args=TrainingArguments(**args), processing_class=tok,
               data_collator=collator or DataCollatorForLanguageModeling(tok, mlm=False))


def test_correct_trainer_with_accumulation_passes():
    r = xray_trainer(_trainer())
    r.assert_token_mean(TOL)
    assert len(r.lengths) == 12  # 3 per device × 4 accumulation steps: one optimizer step
    assert r.sink == 0.0


class IgnoresTokenCount(Trainer):
    """The pre-October-2024 shape of the bug: each micro-batch's mean loss, without the step's token count."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch=None)


def test_custom_compute_loss_is_replayed_and_caught():
    with pytest.raises(AssertionError, match="global token mean"):
        xray_trainer(_trainer(IgnoresTokenCount)).assert_token_mean(TOL)


def test_training_arguments_are_replayed():
    """Label smoothing from the user's arguments must reach the replay (it re-weights every token)."""
    with pytest.raises(AssertionError, match="global token mean"):
        xray_trainer(_trainer(label_smoothing_factor=0.1)).assert_token_mean(TOL)


def _labels_padding(features):
    """A broken collator: pads with id 0 and leaves the padding in the labels."""
    L = max(len(f["input_ids"]) for f in features)
    ids = torch.tensor([list(f["input_ids"]) + [0] * (L - len(f["input_ids"])) for f in features])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def test_loss_on_padding_is_reported():
    r = xray_trainer(_trainer(collator=_labels_padding))
    assert r.sink > 1e-3
    with pytest.raises(AssertionError, match="padding"):
        r.assert_token_mean(TOL)


class _Specials:
    all_special_ids = [0, 5]
    pad_token_id = 0
    eos_token_id = 7
    bos_token_id = None


def test_tracer_ids_never_collide_with_special_tokens():
    assert tracer_start(_Specials()) == 8
    assert tracer_start(None) == 1


def test_sft_trainer_padding_free_is_replayed():
    trl = pytest.importorskip("trl")
    import datasets

    tok = tracer_tokenizer(VOCAB)
    cfg = trl.SFTConfig(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=3, gradient_accumulation_steps=2,
                        report_to=[], use_cpu=True, padding_free=True, max_length=None, bf16=True)
    t = trl.SFTTrainer(model=build_probe(VOCAB), args=cfg, processing_class=tok,
                       train_dataset=datasets.Dataset.from_list([{"input_ids": [1, 2, 3]}]))
    r = xray_trainer(t)
    r.assert_token_mean(TOL)
    assert "bf16" in r.changes  # the replay says what it had to change
