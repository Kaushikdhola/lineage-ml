"""
Does the gradient-accumulation weighting bug change what a model learns? A small, honest experiment.

A tiny LLaMA learns two synthetic "languages" at once: SHORT sequences (4-8 symbols) and LONG ones
(48-64 symbols), each with its own stochastic rule, marked by a source tag. Every optimizer step holds
four micro-batches grouped by length, as `group_by_length` produces: two short, two long. Under a
token mean the short data is ~8% of the step's tokens; when each micro-batch's mean loss counts
equally (transformers' Trainer before the October 2024 fix) it gets 50%.

Arms (same initial weights, same data, same order, same optimizer):
  fixed     transformers 5.x Trainer, 4 micro-batches x 8 sequences          (real run)
  buggy     transformers 4.45.2 Trainer, 4 micro-batches x 8 sequences       (real run, .venv-hf445)
  control   transformers 4.45.2 Trainer, 1 batch x 32 sequences, no accumulation (real run)
If `control` matches `fixed`, the difference between `fixed` and `buggy` comes from accumulation, not
from other changes between the two releases.

    node scripts/py.mjs xray/experiments/accumulation_effect.py          # all arms, 3 seeds, ~5 min on CPU
The single-arm entry point (`--arm`) is what the driver runs inside each environment.

This is a toy: absolute sizes depend on the model, data and step budget. The direction is the point.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results" / "accumulation_effect.json"

PAD, TAG_SHORT, TAG_LONG, FIRST = 0, 1, 2, 3
SYMBOLS = 64
VOCAB = FIRST + SYMBOLS
STEPS = 300
MICRO = 8  # sequences per micro-batch
K = 4  # random increments per step of each rule: the best achievable loss per symbol is ln 4


def _rule(source: str, x: int, k: int) -> int:
    return (x + 1 + k) % SYMBOLS if source == "short" else (3 * x + k) % SYMBOLS


def _sequence(rng, source: str) -> list[int]:
    n = rng.integers(4, 9) if source == "short" else rng.integers(48, 65)
    xs = [int(rng.integers(SYMBOLS))]
    for _ in range(n - 1):
        xs.append(_rule(source, xs[-1], int(rng.integers(K))))
    return [TAG_SHORT if source == "short" else TAG_LONG] + [FIRST + x for x in xs]


def make_data(seed: int) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed)
    train = []
    for _ in range(STEPS):  # one optimizer step = [short, short, long, long] micro-batches, in order
        for source in ("short", "short", "long", "long"):
            train += [_sequence(rng, source) for _ in range(MICRO)]
    held = {s: [_sequence(rng, s) for _ in range(300)] for s in ("short", "long")}
    return {"train": train, "held": held}


def floor_loss(seqs: list[list[int]]) -> float:
    """Best achievable mean loss per target: the tag predicts a uniform first symbol, each rule step has K choices."""
    n = sum(len(s) - 1 for s in seqs)
    return sum(math.log(SYMBOLS) + (len(s) - 2) * math.log(K) for s in seqs) / n


# ── one arm, run inside whichever environment holds the transformers version under test ──


def run_arm(arm: str, data_path: str, init_path: str, out_path: str, seed: int) -> None:
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainerCallback, TrainingArguments

    data = json.load(open(data_path))
    model = LlamaForCausalLM(_config(LlamaConfig))
    model.load_state_dict(load_file(init_path), strict=False)

    def collate(rows):
        L = max(len(r["input_ids"]) for r in rows)
        ids = torch.tensor([r["input_ids"] + [PAD] * (L - len(r["input_ids"])) for r in rows])
        att = torch.tensor([[1] * len(r["input_ids"]) + [0] * (L - len(r["input_ids"])) for r in rows])
        lab = ids.masked_fill(att == 0, -100)
        return {"input_ids": ids, "attention_mask": att, "labels": lab}

    held = {s: collate([{"input_ids": x} for x in seqs]) for s, seqs in data["held"].items()}

    def evaluate():
        model.eval()
        out = {}
        with torch.no_grad():
            for s, b in held.items():
                logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits[:, :-1]
                out[s] = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b["labels"][:, 1:].reshape(-1),
                                               ignore_index=-100))
        model.train()
        return out

    curve = [{"step": 0, **evaluate()}]

    class Eval(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if state.global_step % 25 == 0:
                curve.append({"step": state.global_step, **evaluate()})

    class Ordered(Trainer):  # keep the data in its prepared order: [short, short, long, long] per step
        def _get_train_sampler(self, *a, **k):
            return torch.utils.data.SequentialSampler(self.train_dataset)

    per_device, accum = (MICRO, 4) if arm in ("fixed", "buggy") else (4 * MICRO, 1)
    rows = [{"input_ids": x} for x in data["train"]]

    class Rows(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            return rows[i]

    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(output_dir=tmp, per_device_train_batch_size=per_device, gradient_accumulation_steps=accum,
                                 max_steps=STEPS, learning_rate=3e-3, lr_scheduler_type="constant", warmup_steps=0,
                                 weight_decay=0.0, max_grad_norm=0.0, optim="adamw_torch", report_to="none",
                                 save_strategy="no", logging_strategy="no", use_cpu=True, seed=seed, data_seed=seed,
                                 remove_unused_columns=False, disable_tqdm=True, dataloader_drop_last=False)
        trainer = Ordered(model=model, args=args, train_dataset=Rows(), data_collator=collate, callbacks=[Eval()])
        trainer.train()
    import transformers

    json.dump({"arm": arm, "seed": seed, "transformers": transformers.__version__, "torch": torch.__version__,
               "per_device": per_device, "accumulation": accum, "curve": curve}, open(out_path, "w"))


def _config(LlamaConfig):
    return LlamaConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
                       pad_token_id=PAD, bos_token_id=None, eos_token_id=None, tie_word_embeddings=False,
                       attention_dropout=0.0)


# ── driver ──


def main(seeds=(0, 1, 2)) -> None:
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    legacy = os.environ.get("XRAY_HF445_PYTHON", str(ROOT / ".venv-hf445/bin/python"))
    pythons = {"fixed": sys.executable, "buggy": legacy, "control": legacy}
    results = []
    with tempfile.TemporaryDirectory() as d:
        for seed in seeds:
            data_path, init_path = f"{d}/data{seed}.json", f"{d}/init{seed}.safetensors"
            data = make_data(seed)
            json.dump(data, open(data_path, "w"))
            torch.manual_seed(seed)
            save_file({k: v.contiguous() for k, v in LlamaForCausalLM(_config(LlamaConfig)).state_dict().items()}, init_path)
            for arm, py in pythons.items():
                out = f"{d}/{arm}{seed}.json"
                cmd = [py, __file__, "--arm", arm, "--data", data_path, "--init", init_path, "--out", out, "--seed", str(seed)]
                p = subprocess.run(cmd, capture_output=True, text=True)
                if p.returncode:
                    raise RuntimeError(f"{arm} seed {seed} failed:\n{p.stderr[-3000:]}")
                r = json.load(open(out))
                r["floor"] = {s: floor_loss(data["held"][s]) for s in ("short", "long")}
                results.append(r)
                last = r["curve"][-1]
                print(f"seed {seed} {arm:8s} transformers {r['transformers']:7s} short {last['short']:.4f} long {last['long']:.4f}",
                      flush=True)
    tokens = _token_shares(make_data(seeds[0]))
    OUT.parent.mkdir(exist_ok=True)
    json.dump({"steps": STEPS, "micro": MICRO, "token_share": tokens, "runs": results}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


def _token_shares(data) -> dict:
    """Share of the step's target tokens that are SHORT data: as intended (token mean) and as weighted by the bug."""
    step = data["train"][: 4 * MICRO]
    t = [len(s) - 1 for s in step]
    short, total = sum(t[: 2 * MICRO]), sum(t)
    return {"short_intended": short / total, "short_buggy": 0.5}


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--arm")
    a.add_argument("--data")
    a.add_argument("--init")
    a.add_argument("--out")
    a.add_argument("--seed", type=int, default=0)
    x = a.parse_args()
    if x.arm:
        run_arm(x.arm, x.data, x.init, x.out, x.seed)
    else:
        main()
