"""
Library faults for mutation testing (scripts/mutate.py). Inactive unless XRAY_FAULT names one: each
patches the INSTALLED transformers or TRL in this test process to behave like a realistic library bug,
so the suite can show it would catch that bug. Never set XRAY_FAULT for a normal run.
"""
import os

FAULT = os.environ.get("XRAY_FAULT", "")


def _ga_reverted():
    # the October 2024 fix undone: no token count across accumulation steps, so each micro-batch's
    # mean loss is divided by the number of steps
    from transformers import Trainer

    Trainer._get_num_items_in_batch = lambda self, batch_samples, device: None


def _smoothing_off():
    from transformers.trainer_pt_utils import LabelSmoother

    orig = LabelSmoother.__call__

    def call(self, *a, **k):
        self.epsilon = 0.0
        return orig(self, *a, **k)

    LabelSmoother.__call__ = call


def _sft_boundary_targets():
    # the padding-free collator forgets to mask each document's first token as a target
    import trl.trainer.sft_trainer as sft

    orig = sft.DataCollatorForLanguageModeling.torch_call

    def call(self, examples):
        out = orig(self, examples)
        if "position_ids" in out:
            starts = out["position_ids"] == 0
            starts[:, 0] = False
            out["labels"][starts] = out["input_ids"][starts]
        return out

    sft.DataCollatorForLanguageModeling.torch_call = call


def _dpo_normalizer():
    # the loss averaged over all 2P rows instead of the P pairs
    from trl import DPOTrainer

    orig = DPOTrainer._compute_loss

    def compute(self, *a, **k):
        r = orig(self, *a, **k)
        return (r[0] * 0.5, *r[1:]) if isinstance(r, tuple) else r * 0.5

    DPOTrainer._compute_loss = compute


def _grpo_clip_removed():
    from trl import GRPOTrainer

    orig = GRPOTrainer.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        self.epsilon_low = self.epsilon_high = 1e9

    GRPOTrainer.__init__ = init


def _cache_never_created():
    # as if transformers#50419 were fixed: no cache object in training, so packed detection always runs
    import transformers.models.llama.modeling_llama as llama

    llama.DynamicCache = lambda *a, **k: None


def _collator_keeps_pad_labels():
    # a collator that does not mask labels equal to the pad id
    from transformers import DataCollatorForLanguageModeling

    orig = DataCollatorForLanguageModeling.torch_call

    def call(self, examples):
        out = orig(self, examples)
        out["labels"] = out["input_ids"].clone()
        if "attention_mask" in out:
            out["labels"][out["attention_mask"] == 0] = -100
        return out

    DataCollatorForLanguageModeling.torch_call = call


FAULTS = {
    "ga-reverted": _ga_reverted,
    "label-smoothing-off": _smoothing_off,
    "sft-boundary-targets": _sft_boundary_targets,
    "dpo-normalizer": _dpo_normalizer,
    "grpo-clip-removed": _grpo_clip_removed,
    "cache-never-created": _cache_never_created,
    "collator-keeps-pad-labels": _collator_keeps_pad_labels,
}

if FAULT:
    if FAULT not in FAULTS:
        raise RuntimeError(f"unknown XRAY_FAULT {FAULT!r}; known: {sorted(FAULTS)}")
    FAULTS[FAULT]()
