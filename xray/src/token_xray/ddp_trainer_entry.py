"""
torchrun entry point: one optimizer step of the unmodified transformers Trainer on the probe, on CPU
with the gloo backend. Launched by `ddp.hf_trainer_ddp`; not meant to be run by hand.

Writes, into the spec's output directory:
  rank0.pt      all-reduced input-embedding gradient rows, captured before the optimizer step
  seen<r>.json  the sequence indices rank r received, in order
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from .probe import build_probe, make_batch
from .readout import embedding_row_grads
from .runners import _ListDataset, collate


def main(path: str) -> None:
    spec = json.load(open(path))
    rank = int(os.environ.get("RANK", "0"))
    batch = make_batch(spec["lengths"])
    model = build_probe(batch.vocab_size)
    ds = _ListDataset(batch)
    seen: list[int] = []

    class Capture(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kw):
            # DDP has all-reduced (averaged) the parameter gradients during the last backward
            if rank == 0:
                rows = embedding_row_grads(model, range(1, batch.vocab_size))
                torch.save({"rows": rows, "args": {"average_tokens_across_devices": args.average_tokens_across_devices,
                                                   "world_size": args.world_size}},
                           os.path.join(spec["out"], "rank0.pt"))

    with tempfile.TemporaryDirectory() as out:
        args = TrainingArguments(
            output_dir=out,
            per_device_train_batch_size=spec["per_device_batch"],
            gradient_accumulation_steps=1,
            max_steps=1,
            learning_rate=1e-3,
            max_grad_norm=0.0,
            report_to=[],
            use_cpu=True,
            ddp_backend="gloo",
            save_strategy="no",
            logging_strategy="no",
            seed=0,
            data_seed=0,
            remove_unused_columns=False,
            dataloader_drop_last=False,
            disable_tqdm=True,
            average_tokens_across_devices=spec["average_tokens_across_devices"],
        )
        trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=lambda f: collate(f),
                          callbacks=[Capture()])
        trainer._get_train_sampler = lambda *a, **k: torch.utils.data.SequentialSampler(ds)
        compute = trainer.compute_loss

        def spy(m, inputs, *a, **k):
            for row in inputs["input_ids"].tolist():
                first = next(t for t in row if t != 0)
                seen.append(batch.seq_of_token[first])
            return compute(m, inputs, *a, **k)

        trainer.compute_loss = spy
        trainer.train()
    with open(os.path.join(spec["out"], f"seen{rank}.json"), "w") as f:
        json.dump(seen, f)


if __name__ == "__main__":
    main(sys.argv[1])
