"""
X-ray a configured Hugging Face Trainer in one line, before you spend GPU hours on it.

    from token_xray import xray_trainer

    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    xray_trainer(trainer).assert_token_mean()   # about a second on CPU
    trainer.train()

What is replayed: one optimizer step of YOUR trainer class (including an overridden compute_loss or a
compute_loss_func), YOUR TrainingArguments (batch size, gradient accumulation, label smoothing, sampler,
average_tokens_across_devices) and YOUR data collator. What is swapped: the model becomes the tracer
probe and the data become tracer sequences, so every token's gradient weight can be read back exactly.
Settings that cannot run on the probe are changed and listed in `report.changes` (precision, compilation,
devices, checkpointing, logging). It runs in one process: a data-parallel launch is not replayed.
"""
from __future__ import annotations

import dataclasses
import itertools
import tempfile
from dataclasses import dataclass, field

import torch

from .api import Report
from .probe import build_probe, make_batch
from .readout import read, recording, unit_gradients

# varied lengths, so a normaliser that depends on micro-batch or sequence size shows up
DEFAULT_LENGTHS = [6, 14, 4, 22, 9, 5, 17, 8, 12, 3, 19, 7]

# settings replaced for the replay: what they are set to, and why
OVERRIDES = {
    "max_steps": (1, "one optimizer step"),
    "num_train_epochs": (1, "one optimizer step"),
    "use_cpu": (True, "the probe runs on CPU in float64"),
    "bf16": (False, "the readout needs full precision"),
    "fp16": (False, "the readout needs full precision"),
    "torch_compile": (False, "not needed for a 2-layer probe"),
    "deepspeed": (None, "single process"),
    "fsdp": ("", "single process"),
    "report_to": ("none", "no experiment tracking"),
    "save_strategy": ("no", "nothing is saved"),
    "eval_strategy": ("no", "no evaluation"),
    "logging_strategy": ("no", "no logs"),
    "load_best_model_at_end": (False, "no evaluation"),
    "push_to_hub": (False, "nothing is uploaded"),
    "dataloader_num_workers": (0, "in-process data loading"),
    "dataloader_persistent_workers": (False, "in-process data loading"),
    "remove_unused_columns": (False, "the tracer dataset has only input_ids"),
    "disable_tqdm": (True, "quiet"),
    "resume_from_checkpoint": (None, "fresh run"),
}


# changed settings worth telling the user about (the rest are mechanics of a one-step replay)
MENTION = ("bf16", "fp16", "torch_compile", "deepspeed", "fsdp")


@dataclass
class TrainerReport(Report):
    changes: dict = field(default_factory=dict)  # setting -> (your value, replay value, reason)
    sink: float = 0.0  # gradient norm that reached non-tracer positions (padding, special tokens)
    notes: list[str] = field(default_factory=list)

    def assert_token_mean(self, tol: float = 1e-6) -> None:
        if self.sink > tol:
            raise AssertionError(
                f"gradient reached padding or special-token positions (norm {self.sink:.3g}): a loss is scored "
                "where the labels should be masked.\n" + self.table()
            )
        super().assert_token_mean(tol)

    def summary(self) -> str:
        lines = [self.table()]
        if self.changes:
            lines.append("changed for the replay: " + ", ".join(f"{k}={v[1]!r}" for k, v in self.changes.items()))
        lines += self.notes
        return "\n".join(lines)


def _special_ids(processing_class) -> list[int]:
    ids = set()
    for name in ("all_special_ids",):
        ids.update(i for i in (getattr(processing_class, name, None) or []) if isinstance(i, int))
    for name in ("pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"):
        i = getattr(processing_class, name, None)
        if isinstance(i, int):
            ids.add(i)
    return sorted(ids)


def tracer_start(processing_class) -> int:
    """First tracer id: above every special id of the tokenizer the collator pads with."""
    return max([0, *_special_ids(processing_class)]) + 1


def _replay_args(args, out_dir: str):
    names = {f.name for f in dataclasses.fields(args) if f.init}
    want = {k: v for k, (v, _) in OVERRIDES.items() if k in names}
    want["output_dir"] = out_dir
    # report only the settings the replay changes that a user would want to know about
    changes = {k: (getattr(args, k), want[k], OVERRIDES[k][1]) for k in MENTION if k in want and getattr(args, k)}
    return dataclasses.replace(args, **want), changes


def _dataset(batch):
    rows = [{"input_ids": list(s)} for s in batch.sequences]
    try:
        import datasets

        return datasets.Dataset.from_list(rows)
    except ImportError:

        class _Rows(torch.utils.data.Dataset):
            def __len__(self):
                return len(rows)

            def __getitem__(self, i):
                return rows[i]

        return _Rows()


def xray_trainer(trainer, lengths: list[int] | None = None, seed: int = 0) -> TrainerReport:
    """Replay one optimizer step of a configured Trainer on the tracer probe and read every token's weight."""
    args = trainer.args
    n = args.per_device_train_batch_size * args.gradient_accumulation_steps
    lengths = list(lengths) if lengths else list(itertools.islice(itertools.cycle(DEFAULT_LENGTHS), n))
    if len(lengths) != n:
        raise ValueError(f"one optimizer step takes {n} sequences "
                         f"(per_device_train_batch_size × gradient_accumulation_steps); got {len(lengths)} lengths")
    processing_class = getattr(trainer, "processing_class", None)
    batch = make_batch(lengths, first_id=tracer_start(processing_class))
    vocab = max([batch.vocab_size, *[i + 1 for i in _special_ids(processing_class)]])
    unit = unit_gradients(build_probe(vocab, seed=seed), batch)
    probe = build_probe(vocab, seed=seed)

    notes = []
    if args.world_size > 1 or getattr(args, "average_tokens_across_devices", False) is False:
        notes.append("Replayed in one process: how a data-parallel launch normalises across ranks "
                     "(average_tokens_across_devices) is not covered by this check.")

    cls = type(trainer)  # your class: an overridden compute_loss is part of what is checked
    with tempfile.TemporaryDirectory() as out:
        replay_args, changes = _replay_args(args, out)
        kwargs = dict(model=probe, args=replay_args, train_dataset=_dataset(batch), data_collator=trainer.data_collator)
        if processing_class is not None:
            kwargs["processing_class"] = processing_class
        if getattr(trainer, "compute_loss_func", None) is not None:
            kwargs["compute_loss_func"] = trainer.compute_loss_func
        try:
            try:
                replay = cls(**kwargs)
            except ValueError as e:
                # some trainers (TRL's SFTTrainer when padding-free) refuse an explicit collator, even their
                # own: let the class build its default collator again from the same arguments
                if "collator" not in str(e):
                    raise
                kwargs.pop("data_collator")
                replay = cls(**kwargs)
                notes.append(f"{cls.__name__} rebuilt its own default data collator from your arguments.")
        except TypeError as e:
            raise TypeError(
                f"could not rebuild {type(trainer).__name__} with a probe model ({e}). If its constructor needs "
                "extra arguments, use token_xray.xray(step, lengths) with your own step function instead."
            ) from e
        from transformers.trainer_callback import PrinterCallback, ProgressCallback

        for cb in (PrinterCallback, ProgressCallback):  # the replay is silent
            replay.remove_callback(cb)
        with recording(probe) as rec:
            replay.train()

    tracers = set(batch.seq_of_token)
    sink_sq = rec.pad_sq + sum(float(g.norm()) ** 2 for t, g in rec.grads.items() if t not in tracers)
    return TrainerReport(xray=read(rec.grads, unit, batch), lengths=lengths, changes=changes,
                         sink=sink_sq**0.5, notes=notes)
