"""
Minimal reproduction: packed sequences attend across documents when use_cache is left at the
model default, even though position_ids mark the document boundaries.

Only torch and transformers are needed. Run:  python packed_position_ids_use_cache.py

Expected: the logits of a document are the same whether it is run alone or packed with others
(position_ids restart at 0 for each document, no attention_mask, the padding-free convention).
Observed (transformers 5.22.2, torch 2.15): identical with use_cache=False, different with the
model default (config.use_cache=True), because a DynamicCache is created and packed-sequence
detection in masking_utils only runs when past_key_values is None. No warning is emitted.
The Trainer is not affected: it sets model.config.use_cache from TrainingArguments.use_cache.
"""
import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM

torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=4)
model = LlamaForCausalLM(cfg).train()  # training mode, as in a custom training loop

docs = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12], [13, 14, 15, 16]]
input_ids = torch.tensor([[t for d in docs for t in d]])
position_ids = torch.tensor([[p for d in docs for p in range(len(d))]])
start = len(docs[0]) + len(docs[1])

alone = model(input_ids=torch.tensor([docs[2]]), use_cache=False).logits[0]
for kwargs in ({}, {"use_cache": False}):
    packed = model(input_ids=input_ids, position_ids=position_ids, **kwargs).logits[0, start:]
    label = "default use_cache" if not kwargs else "use_cache=False  "
    print(f"{label}: max |packed - alone| for document 3 = {(packed - alone).abs().max().item():.3e}")

print(f"transformers {transformers.__version__}, torch {torch.__version__}, config.use_cache={cfg.use_cache}")
