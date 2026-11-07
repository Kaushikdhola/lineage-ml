"""
The tracer probe.

A tiny HF-compatible LLaMA in which every token in the batch has a unique id, so any gradient
the training stack produces can be attributed to exactly one token.

Two modes:
  * aggregation: attention output projections are zeroed and frozen, so no information flows
    between positions. Each position's loss depends only on its own embedding, and the gradient
    at position i is w_i * r_i, where r_i is the token's unit-weight gradient and w_i is the
    weight the stack applied. Reading w_i for every token is the X-ray.
  * leak: attention is live. Used to check that one sequence's loss sends exactly zero gradient
    into another sequence (packing, masks, padding).

Only the input embedding is trainable and it is untied from the LM head, so the readout is
unaffected by weight tying.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import LlamaConfig, LlamaForCausalLM

PAD_ID = 0  # reserved; tracer ids start at 1


@dataclass
class TracerBatch:
    """Sequences of unique tracer ids. sequences[i] is a list of token ids; ids never repeat."""

    sequences: list[list[int]]
    vocab_size: int
    seq_of_token: dict[int, int] = field(default_factory=dict)

    @property
    def lengths(self) -> list[int]:
        return [len(s) for s in self.sequences]

    @property
    def n_targets(self) -> int:
        """Tokens that carry a next-token target (all but the last of each sequence)."""
        return sum(max(0, len(s) - 1) for s in self.sequences)


def make_batch(lengths: list[int]) -> TracerBatch:
    if any(L < 2 for L in lengths):
        raise ValueError("every sequence needs at least 2 tokens (one target)")
    seqs, nxt, owner = [], 1, {}
    for i, L in enumerate(lengths):
        ids = list(range(nxt, nxt + L))
        for t in ids:
            owner[t] = i
        seqs.append(ids)
        nxt += L
    return TracerBatch(sequences=seqs, vocab_size=nxt, seq_of_token=owner)


def build_probe(
    vocab_size: int,
    mode: str = "aggregation",
    seed: int = 0,
    hidden_size: int = 16,
    layers: int = 2,
    heads: int = 2,
    dtype: torch.dtype = torch.float64,
    attn_implementation: str = "sdpa",
) -> LlamaForCausalLM:
    if mode not in ("aggregation", "leak"):
        raise ValueError("mode must be 'aggregation' or 'leak'")
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        pad_token_id=PAD_ID,
        bos_token_id=None,
        eos_token_id=None,
        max_position_embeddings=4096,
    )
    cfg._attn_implementation = attn_implementation
    model = LlamaForCausalLM(cfg).to(dtype)
    if mode == "aggregation":
        for layer in model.model.layers:
            torch.nn.init.zeros_(layer.self_attn.o_proj.weight)
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("model.embed_tokens")
    model.eval()  # dropout off; Trainer will call train(), dropout is 0 anyway
    assert cfg.attention_dropout == 0.0
    return model
