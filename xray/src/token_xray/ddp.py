"""
Data-parallel runs on CPU (gloo): does the weight a token receives depend on how the batch was split
across ranks?

Activation gradients are local to a rank; DDP averages PARAMETER gradients in the all-reduce after
backward. So these runs read the input-embedding parameter gradient (`readout.embedding_row_grads`)
after the all-reduce, on rank 0, where every rank's contribution has arrived.

Two launchers:
  * `ddp_step`: torch.multiprocessing.spawn + DistributedDataParallel around the probe, with a
    hand-written loss in one of two reductions (global token mean, per-rank mean).
  * `hf_trainer_ddp`: the unmodified transformers Trainer under torchrun (2 CPU processes, gloo).

Every run records which sequences each rank actually received, so the intended weights are computed
from the observed split rather than an assumed one.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from .probe import TracerBatch, build_probe, make_batch
from .readout import embedding_row_grads
from .runners import collate

REDUCTIONS = ("global_token_mean", "per_rank_mean")


@dataclass
class DDPResult:
    rows: dict[str, dict[int, torch.Tensor]]  # reduction -> token id -> all-reduced embedding gradient row
    assignment: list[list[int]]  # sequence indices seen by each rank, in order (duplicates included)
    local_targets: list[int]  # target tokens per rank
    world: int


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _shard(n_seqs: int, world: int, split, sampler: bool) -> list[list[int]]:
    if sampler:
        from torch.utils.data import DistributedSampler

        return [list(DistributedSampler(range(n_seqs), num_replicas=world, rank=r, shuffle=False, drop_last=False))
                for r in range(world)]
    return [list(s) for s in split]


def _worker(rank: int, world: int, port: int, lengths: list[int], split, sampler: bool, reductions, out: str):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        batch = make_batch(lengths)
        mine = _shard(len(lengths), world, split, sampler)[rank]
        b = collate([{"input_ids": batch.sequences[i]} for i in mine])
        tgt = b["labels"][:, 1:]
        n_local = torch.tensor(float((tgt != -100).sum()), dtype=torch.float64)
        n_all = [torch.zeros_like(n_local) for _ in range(world)]
        dist.all_gather(n_all, n_local)
        rows = {}
        for red in reductions:
            model = build_probe(batch.vocab_size)
            ddp = torch.nn.parallel.DistributedDataParallel(model)
            logits = ddp(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                 ignore_index=-100, reduction="sum")
            if red == "global_token_mean":
                # every rank divides by the GLOBAL token count; ×world undoes DDP's average
                loss = ce / sum(n_all) * world
            elif red == "per_rank_mean":
                loss = ce / n_local  # each rank's own token mean; DDP then averages the ranks
            else:
                raise ValueError(red)
            loss.backward()  # DDP all-reduces (averages) the embedding gradient here
            rows[red] = embedding_row_grads(model, range(1, batch.vocab_size))
        if rank == 0:
            torch.save({"rows": rows, "assignment": _shard(len(lengths), world, split, sampler),
                        "local_targets": [int(x) for x in n_all]}, out)
    finally:
        dist.destroy_process_group()


def ddp_step(
    lengths: list[int],
    world: int,
    split: list[list[int]] | None = None,
    sampler: bool = False,
    reductions: tuple[str, ...] = REDUCTIONS,
) -> DDPResult:
    """One backward pass under DDP with `world` CPU processes.

    split[r] lists the sequence indices given to rank r. sampler=True instead lets
    torch.utils.data.DistributedSampler (shuffle off, drop_last off) decide, including its padding.
    """
    if split is None and not sampler:
        split = [list(range(len(lengths)))] if world == 1 else None
        if split is None:
            raise ValueError("give a split or set sampler=True")
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "rank0.pt")
        mp.spawn(_worker, args=(world, _free_port(), lengths, split, sampler, tuple(reductions), out),
                 nprocs=world, join=True)
        r = torch.load(out)
    return DDPResult(rows=r["rows"], assignment=r["assignment"], local_targets=r["local_targets"], world=world)


# ── the transformers Trainer under torchrun ────────────────────────────────


def hf_trainer_ddp(lengths: list[int], per_device_batch: int, average_tokens_across_devices: bool = True,
                   world: int = 2, timeout: float = 300.0) -> dict:
    """One optimizer step of the unmodified Trainer under `torchrun --nproc_per_node world` (CPU, gloo).
    Returns the all-reduced embedding rows (rank 0) and the sequences each rank actually received."""
    with tempfile.TemporaryDirectory() as d:
        spec = {"lengths": lengths, "per_device_batch": per_device_batch,
                "average_tokens_across_devices": average_tokens_across_devices, "out": d}
        path = os.path.join(d, "spec.json")
        with open(path, "w") as f:
            json.dump(spec, f)
        env = {**os.environ, "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
        # Static rendezvous on loopback. `--standalone` resolves the machine's host name, which on some
        # networks is a reverse-DNS name that never answers, and the store then waits 300 s per attempt.
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", f"--nproc_per_node={world}",
               "--master_addr=127.0.0.1", f"--master_port={_free_port()}",
               "-m", "token_xray.ddp_trainer_entry", path]
        try:
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, start_new_session=True)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"torchrun did not finish within {timeout} s; stderr tail:\n{(e.stderr or '')[-2000:]}")
        if p.returncode != 0:
            raise RuntimeError(f"torchrun failed ({p.returncode}):\n{p.stderr[-3000:]}")
        r = torch.load(os.path.join(d, "rank0.pt"))
        seen = [json.load(open(os.path.join(d, f"seen{k}.json"))) for k in range(world)]
    return {"rows": r["rows"], "seen": seen, "world": world, "args": r["args"]}
