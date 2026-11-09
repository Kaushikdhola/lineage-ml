"""
Real TRL runs on the tracer probe.

Each runner drives an unmodified TRL trainer for real optimizer steps and records the gradient every
tracer token received. The only things set by hand are declared configuration (batch sizes, learning
rate, loss type) and, for GRPO, the completions: `model.generate` is replaced by a script that returns
fixed tracer completions, so the trainer's own code still slices, masks, scores and optimizes them.

Precision: TRL's configs default to bf16 mixed precision (also on CPU). The runners pass bf16=False so
the measured weights are exact; with the default, every SFT weight carries a few 1e-3 of bf16 rounding
on this probe (the export records the measured value).
"""
from __future__ import annotations

import copy
import logging
import tempfile
from dataclasses import dataclass, field

import torch

from .probe import PAD_ID, TracerBatch, build_probe
from .readout import GradRecorder, recording
from .runners import StepResult

logging.getLogger("trl").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

EOS_ID = PAD_ID  # the tracer tokenizer uses one special token for pad and eos


def tracer_tokenizer(vocab_size: int):
    """A word-level tokenizer over the tracer ids: "t7" <-> 7, "<pad>" <-> 0 (also eos)."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": PAD_ID, **{f"t{i}": i for i in range(1, vocab_size)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="<pad>"))
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<pad>")


def _common(out: str, **kw) -> dict:
    base = dict(
        output_dir=out,
        max_steps=1,
        learning_rate=1e-3,
        max_grad_norm=0.0,  # declared: no clipping
        bf16=False,  # declared: TRL defaults to bf16 mixed precision; exact readout needs full precision
        report_to=[],
        use_cpu=True,
        save_strategy="no",
        logging_steps=1,
        seed=0,
        data_seed=0,
        disable_tqdm=True,
        dataloader_drop_last=False,
    )
    base.update(kw)
    return base


def _sequential(trainer):
    trainer._get_train_sampler = lambda *a, **k: torch.utils.data.SequentialSampler(trainer.train_dataset)


# ── SFT ─────────────────────────────────────────────────────────────────────


def sft_trainer(
    batch: TracerBatch,
    per_device_batch: int,
    grad_accum: int,
    padding_free: bool = True,
    packing: bool = False,
    max_length: int | None = None,
    labels_only: int | None = None,
    mode: str = "aggregation",
    bf16: bool = False,
    model_seed: int = 0,
) -> StepResult:
    """One optimizer step of trl.SFTTrainer on the probe.

    labels_only=i keeps the labels of sequence i only (a "labels" column, which SFT uses as given);
    with mode="leak" that is one row of the leak matrix through the trainer.
    """
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model = build_probe(batch.vocab_size, mode, seed=model_seed)
    rows = []
    for i, s in enumerate(batch.sequences):
        row = {"input_ids": list(s)}
        if labels_only is not None:
            row["labels"] = list(s) if i == labels_only else [-100] * len(s)
        rows.append(row)
    seen: list[dict] = []
    with tempfile.TemporaryDirectory() as out:
        kw = dict(padding_free=padding_free, packing=packing, max_length=max_length, bf16=bf16)
        if packing:
            kw["packing_strategy"] = "bfd"
        args = SFTConfig(**_common(out, per_device_train_batch_size=per_device_batch,
                                   gradient_accumulation_steps=grad_accum, **kw))
        trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows),
                             processing_class=tracer_tokenizer(batch.vocab_size))
        _sequential(trainer)
        compute = trainer.compute_loss

        def spy(m, inputs, *a, **k):
            seen.append({"keys": sorted(inputs), "shape": tuple(inputs["input_ids"].shape)})
            return compute(m, inputs, *a, **k)

        trainer.compute_loss = spy
        with recording(model) as rec:
            trainer.train()
        logs = [h for h in trainer.state.log_history if "loss" in h]
        return StepResult(
            grads=rec.grads,
            logged_loss=float(logs[0]["loss"]) if logs else None,
            info={"library": "trl.SFTTrainer", "micro_batches": seen, "rows": len(trainer.train_dataset),
                  "padding_free": trainer.padding_free, "bf16": bf16, "pad_sink": rec.pad_sink},
        )


def sft_leak_matrix(batch: TracerBatch, packing: bool = False, max_length: int | None = None):
    """Leak matrix through SFTTrainer: one real run per document, labels on that document only."""
    import numpy as np

    K = len(batch.sequences)
    M = np.zeros((K, K))
    for i in range(K):
        res = sft_trainer(batch, per_device_batch=K, grad_accum=1, padding_free=True, packing=packing,
                          max_length=max_length, labels_only=i, mode="leak")
        n = np.zeros(K)
        for t, g in res.grads.items():
            n[batch.seq_of_token[t]] += float(g.norm()) ** 2
        n = np.sqrt(n)
        M[i] = n / n[i] if n[i] > 0 else n
    return M


# ── preference and RL batches with tracer ids ──────────────────────────────


@dataclass
class PromptBatch:
    """A prompt and completions made of unique tracer ids (the prompt is shared, completions are not)."""

    prompts: list[list[int]]
    completions: list[list[int]]  # in batch order
    vocab_size: int
    owner: dict[int, int] = field(default_factory=dict)  # completion token id -> completion index
    eos_ids: list[int] = field(default_factory=list)  # per-completion EOS tracers (GRPO)
    hidden: int = 16  # probe width: at least 2G, so every shared prompt token's targets stay identifiable

    def probe(self, seed: int = 0):
        return build_probe(self.vocab_size, seed=seed, hidden_size=self.hidden)

    def pairs(self, k: int, prompt: list[int]) -> list[tuple[int, int]]:
        """(input, target) for every scored token of completion k: the prompt's last token predicts the
        first completion token, then each completion token predicts the next."""
        seq = [prompt[-1]] + self.completions[k]
        return list(zip(seq[:-1], seq[1:]))


def _ids(start: int, n: int) -> list[int]:
    return list(range(start, start + n))


# ── DPO ─────────────────────────────────────────────────────────────────────


def dpo_batch(prompt_len: int, chosen_lens: list[int], rejected_lens: list[int]) -> PromptBatch:
    """P pairs, completions in order chosen_0, rejected_0, chosen_1, ... Each completion ends with the EOS
    that DPOTrainer appends to non-conversational text (EOS is scored)."""
    nxt, prompts, comps, owner = 1, [], [], {}
    for c, r in zip(chosen_lens, rejected_lens):
        p = _ids(nxt, prompt_len)
        nxt += prompt_len
        prompts.append(p)
        for L in (c, r):
            ids = _ids(nxt, L)
            nxt += L
            for t in ids:
                owner[t] = len(comps)
            comps.append(ids + [EOS_ID])
    return PromptBatch(prompts=prompts, completions=comps, vocab_size=nxt, owner=owner)


def _text(ids: list[int]) -> str:
    return " ".join(f"t{t}" for t in ids if t != EOS_ID)


def sequence_logps(model, prompt: list[int], completion: list[int]) -> torch.Tensor:
    """Per-token log π of a completion after its prompt (plain forward, the definition)."""
    ids = torch.tensor([prompt + completion])
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, len(prompt) - 1 : -1]
    return torch.log_softmax(logits, -1).gather(1, torch.tensor(completion).unsqueeze(1))[:, 0]


def dpo_trainer(pb: PromptBatch, beta: float = 0.1, model_seed: int = 0, ref_seed: int = 1) -> dict:
    """One optimizer step of trl.DPOTrainer (sigmoid loss) on the probe, policy != reference."""
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    P = len(pb.prompts)
    model = pb.probe(model_seed)
    ref = pb.probe(ref_seed)
    # margins Δ_i from the definition, before the step
    deltas = []
    for i in range(P):
        p, c, r = pb.prompts[i], pb.completions[2 * i], pb.completions[2 * i + 1]
        lc, lr = sequence_logps(model, p, c).sum(), sequence_logps(model, p, r).sum()
        rc, rr = sequence_logps(ref, p, c).sum(), sequence_logps(ref, p, r).sum()
        deltas.append(float((lc - rc) - (lr - rr)))
    # this TRL version tokenizes text itself (no pre-tokenized path), and appends EOS to each completion
    rows = [{"prompt": _text(pb.prompts[i]), "chosen": " " + _text(pb.completions[2 * i]),
             "rejected": " " + _text(pb.completions[2 * i + 1])} for i in range(P)]
    with tempfile.TemporaryDirectory() as out:
        args = DPOConfig(**_common(out, per_device_train_batch_size=P, gradient_accumulation_steps=1, beta=beta,
                                   loss_type=["sigmoid"]))
        trainer = DPOTrainer(model=model, ref_model=ref, args=args, train_dataset=Dataset.from_list(rows),
                             processing_class=tracer_tokenizer(pb.vocab_size))
        _sequential(trainer)
        with recording(model) as rec:
            trainer.train()
    return {"grads": rec.grads, "deltas": deltas, "beta": beta, "loss_type": list(args.loss_type),
            "pad_sink": rec.pad_sink}


# ── GRPO ────────────────────────────────────────────────────────────────────


def grpo_batch(prompt_len: int, lengths: list[int]) -> PromptBatch:
    """One prompt, G completions. Each completion ends with its OWN end-of-sequence tracer (scored, like a
    real EOS), declared to the model as an EOS id, so no scored position ever holds the pad id."""
    prompt = _ids(1, prompt_len)
    nxt, comps, owner, eos_ids = prompt_len + 1, [], {}, []
    for L in lengths:
        ids = _ids(nxt, L + 1)  # L tokens, then this completion's EOS
        nxt += L + 1
        for t in ids:
            owner[t] = len(comps)
        comps.append(ids)
        eos_ids.append(ids[-1])
    return PromptBatch(prompts=[prompt], completions=comps, vocab_size=nxt, owner=owner, eos_ids=eos_ids,
                       hidden=max(16, 2 * len(lengths)))


def _scripted_generate(model, completions: list[list[int]]):
    """Replace model.generate: return the prompt followed by the scripted completions (EOS-padded)."""
    width = max(len(c) for c in completions)

    def generate(input_ids=None, attention_mask=None, **kwargs):
        if input_ids.shape[0] != len(completions):
            raise RuntimeError(f"expected a generation batch of {len(completions)}, got {input_ids.shape[0]}")
        tail = torch.tensor([c + [EOS_ID] * (width - len(c)) for c in completions], device=input_ids.device)
        return torch.cat([input_ids, tail], dim=1)

    model.generate = generate


def grpo_trainer(
    pb: PromptBatch,
    rewards: list[float],
    loss_type: str | None = None,
    num_iterations: int = 2,
    learning_rate: float = 0.05,
    max_completion_length: int = 16,
    scale_rewards: str | None = None,
    model_seed: int = 0,
) -> dict:
    """num_iterations optimizer steps of trl.GRPOTrainer on one generation batch of scripted completions.

    loss_type=None keeps the installed default. Returns, per optimizer step, the recorded gradients and
    a snapshot of the parameters the step used (for the unit gradients and the importance ratios).
    """
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    G = len(pb.completions)
    model = pb.probe(model_seed)
    model.generation_config.eos_token_id = list(pb.eos_ids)  # TRL ends a completion at any declared EOS id
    _scripted_generate(model, pb.completions)
    reward_of = {c[0]: r for c, r in zip(pb.completions, rewards)}

    def reward(completion_ids, **kwargs):
        return [reward_of[c[0]] for c in completion_ids]

    tok = tracer_tokenizer(pb.vocab_size)
    prompt_text = _text(pb.prompts[0])
    steps: list[dict] = []
    with tempfile.TemporaryDirectory() as out:
        kw = dict(per_device_train_batch_size=G, gradient_accumulation_steps=1, num_generations=G,
                  num_iterations=num_iterations, max_steps=num_iterations, learning_rate=learning_rate,
                  max_completion_length=max_completion_length, beta=0.0)
        if loss_type is not None:
            kw["loss_type"] = loss_type
        if scale_rewards is not None:
            kw["scale_rewards"] = scale_rewards
        args = GRPOConfig(**_common(out, **kw))
        trainer = GRPOTrainer(model=model, reward_funcs=reward, args=args,
                              train_dataset=Dataset.from_list([{"prompt": prompt_text}]), processing_class=tok)
        step = trainer.training_step

        def recorded_step(m, inputs, *a, **k):
            snap = copy.deepcopy(model.state_dict())
            rec = GradRecorder(model)
            with rec:
                out_ = step(m, inputs, *a, **k)
            steps.append({"grads": rec.grads, "state": snap, "pad_sink": rec.pad_sink})
            return out_

        trainer.training_step = recorded_step
        trainer.train()
        used = {"loss_type": args.loss_type, "epsilon": args.epsilon, "epsilon_high": args.epsilon_high,
                "scale_rewards": args.scale_rewards, "beta": args.beta, "num_iterations": args.num_iterations,
                "max_completion_length": args.max_completion_length}
    return {"steps": steps, "config": used, "vocab_size": pb.vocab_size}


def probe_at(state: dict, pb: PromptBatch, seed: int = 0):
    m = pb.probe(seed)
    m.load_state_dict(state)
    return m


def ratios_between(pb: PromptBatch, new_state: dict, old_state: dict) -> list[list[float]]:
    """ρ_t = π_new(token) / π_old(token) for every scored completion token (definition, plain forward)."""
    new, old = probe_at(new_state, pb), probe_at(old_state, pb)
    p = pb.prompts[0]
    return [torch.exp(sequence_logps(new, p, c) - sequence_logps(old, p, c)).tolist() for c in pb.completions]
