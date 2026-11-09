"""
Runners execute ONE training step on a tracer batch and return the recorded per-token gradients.

Real runners drive unmodified libraries (the HF Trainer). Emulations reproduce the documented
mechanism of a historical bug in plain PyTorch so it can be demonstrated on any machine; they are
always labelled as emulations, never presented as the library itself.
"""
from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

from .probe import PAD_ID, TracerBatch, build_probe
from .readout import recording

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class StepResult:
    grads: dict[int, torch.Tensor]
    logged_loss: float | None = None
    info: dict = field(default_factory=dict)


def collate(features, padding_side: str = "right", pad_label_ids: tuple[int, ...] = ()):
    """Pads tracer sequences; labels = input_ids with padding (and any pad_label_ids) set to -100."""
    L = max(len(f["input_ids"]) for f in features)
    ids, att, lab = [], [], []
    for f in features:
        x = list(f["input_ids"])
        pad = [PAD_ID] * (L - len(x))
        a = [1] * len(x)
        y = [(-100 if t in pad_label_ids else t) for t in x]
        if padding_side == "right":
            ids.append(x + pad), att.append(a + [0] * len(pad)), lab.append(y + [-100] * len(pad))
        else:
            ids.append(pad + x), att.append([0] * len(pad) + a), lab.append([-100] * len(pad) + y)
    return {
        "input_ids": torch.tensor(ids),
        "attention_mask": torch.tensor(att),
        "labels": torch.tensor(lab),
    }


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, batch: TracerBatch):
        self.items = [{"input_ids": s} for s in batch.sequences]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def hf_trainer(
    batch: TracerBatch,
    per_device_batch: int,
    grad_accum: int,
    padding_side: str = "right",
    seed: int = 0,
    shuffle: bool = False,
    label_smoothing: float = 0.0,
    model_seed: int = 0,
) -> StepResult:
    """One optimizer step of the unmodified transformers Trainer on the probe."""
    from transformers import Trainer, TrainingArguments

    if per_device_batch * grad_accum != len(batch.sequences):
        raise ValueError("per_device_batch * grad_accum must equal the number of sequences")
    model = build_probe(batch.vocab_size, "aggregation", seed=model_seed)
    ds = _ListDataset(batch)
    with tempfile.TemporaryDirectory() as out:
        args = TrainingArguments(
            output_dir=out,
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=grad_accum,
            max_steps=1,
            learning_rate=1e-3,
            max_grad_norm=0.0,  # declared: no clipping, so absolute weights are meaningful
            label_smoothing_factor=label_smoothing,
            report_to=[],
            use_cpu=True,
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=1,
            seed=seed,
            data_seed=seed,
            remove_unused_columns=False,
            dataloader_drop_last=False,
            disable_tqdm=True,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds,
            data_collator=lambda f: collate(f, padding_side),
        )
        if not shuffle:
            trainer._get_train_sampler = lambda *a, **k: torch.utils.data.SequentialSampler(ds)
        with recording(model) as rec:
            trainer.train()
        logs = [h for h in trainer.state.log_history if "loss" in h]
        return StepResult(
            grads=rec.grads,
            logged_loss=float(logs[0]["loss"]) if logs else None,
            info={"library": "transformers.Trainer", "pad_sink": rec.pad_sink},
        )


def reference(batch: TracerBatch, micro: int, padding_side: str = "right", model_seed: int = 0) -> StepResult:
    """Plain PyTorch, correct global token mean across micro-batches (the definition)."""
    model = build_probe(batch.vocab_size, "aggregation", seed=model_seed)
    N = batch.n_targets
    total = 0.0
    with recording(model) as rec:
        for i in range(0, len(batch.sequences), micro):
            b = collate([{"input_ids": s} for s in batch.sequences[i : i + micro]], padding_side)
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                b["labels"][:, 1:].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ) / N
            loss.backward()
            total += float(loss.detach())
    return StepResult(grads=rec.grads, logged_loss=total, info={"library": "pytorch reference"})


def _hf_loss(model, b):
    """The model's own loss (mean over the micro-batch's target tokens)."""
    return model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"]).loss


def emulate_mean_of_means(batch: TracerBatch, micro: int, model_seed: int = 0) -> StepResult:
    """EMULATION of the pre-October-2024 gradient-accumulation behaviour in transformers:
    each micro-batch's mean loss divided by the number of accumulation steps."""
    model = build_probe(batch.vocab_size, "aggregation", seed=model_seed)
    k = (len(batch.sequences) + micro - 1) // micro
    with recording(model) as rec:
        for i in range(0, len(batch.sequences), micro):
            b = collate([{"input_ids": s} for s in batch.sequences[i : i + micro]])
            (_hf_loss(model, b) / k).backward()
    return StepResult(grads=rec.grads, info={"emulates": "transformers GA loss averaging (fixed Oct 2024)"})


def emulate_dropped_microbatches(batch: TracerBatch, micro: int, model_seed: int = 0) -> StepResult:
    """EMULATION of an optimizer-offload path that keeps only the first micro-batch's gradient
    (the mechanism reported for DeepSpeed ZeRO-1/2 CPU offload, introduced in PR #6550)."""
    model = build_probe(batch.vocab_size, "aggregation", seed=model_seed)
    N = batch.n_targets
    with recording(model) as rec:
        b = collate([{"input_ids": s} for s in batch.sequences[:micro]])
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), b["labels"][:, 1:].reshape(-1), ignore_index=-100, reduction="sum"
        ) / N
        loss.backward()
    return StepResult(grads=rec.grads, info={"emulates": "gradient of micro-batches 2..k dropped"})


def with_eos(batch: TracerBatch) -> TracerBatch:
    """The same tracer sequences, each ending with an EOS whose id is the pad id (0)."""
    return TracerBatch(sequences=[s + [PAD_ID] for s in batch.sequences], vocab_size=batch.vocab_size,
                       seq_of_token=dict(batch.seq_of_token))


def hf_trainer_pad_equals_eos(batch: TracerBatch, model_seed: int = 0) -> StepResult:
    """One step of the unmodified Trainer with the library's DataCollatorForLanguageModeling (mlm=False)
    and a tokenizer whose pad token is its EOS token. Every sequence ends with that EOS."""
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    from .trl_runners import tracer_tokenizer

    eos = with_eos(batch)
    model = build_probe(batch.vocab_size, "aggregation", seed=model_seed)
    tok = tracer_tokenizer(batch.vocab_size)
    assert tok.pad_token_id == tok.eos_token_id == PAD_ID
    ds = _ListDataset(eos)
    with tempfile.TemporaryDirectory() as out:
        args = TrainingArguments(output_dir=out, per_device_train_batch_size=len(eos.sequences), max_steps=1,
                                 learning_rate=1e-3, max_grad_norm=0.0, report_to=[], use_cpu=True, save_strategy="no",
                                 logging_strategy="no", seed=0, remove_unused_columns=False, disable_tqdm=True)
        trainer = Trainer(model=model, args=args, train_dataset=ds,
                          data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
        trainer._get_train_sampler = lambda *a, **k: torch.utils.data.SequentialSampler(ds)
        with recording(model) as rec:
            trainer.train()
    return StepResult(grads=rec.grads, info={"library": "transformers DataCollatorForLanguageModeling",
                                             "pad_sink": rec.pad_sink})


Runner = Callable[[TracerBatch], StepResult]
