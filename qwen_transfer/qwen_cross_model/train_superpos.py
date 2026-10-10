#!/usr/bin/env python3
"""Train the labeled source SuperPos prompt; backbone weights remain frozen."""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from .config import (BASIS_SIZE, DATASET, DEFAULT_SEED, MAX_LENGTH, PROMPT_LENGTH,
                     SOURCE_MODEL, SUPERPOS_TEMPERATURE, TRAIN_BATCH_SIZE,
                     TRAIN_EPOCHS, TRAIN_LR, TRAIN_WARMUP_RATIO,
                     TRAIN_WEIGHT_DECAY)
from .data import build_loader, load_classification_data
from .superpos import SuperPosModel, save_source_prompt


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.inference_mode()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        labels = batch["label"].to(logits.device)
        correct += int((logits.argmax(-1) == labels).sum())
        total += labels.numel()
    model.train()
    return correct / max(total, 1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train source Qwen SuperPos prompt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default=SOURCE_MODEL)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--prompt-length", type=int, default=PROMPT_LENGTH)
    p.add_argument("--basis-size", type=int, default=BASIS_SIZE)
    p.add_argument("--temperature", type=float, default=SUPERPOS_TEMPERATURE)
    p.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    p.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    p.add_argument("--learning-rate", type=float, default=TRAIN_LR)
    p.add_argument("--weight-decay", type=float, default=TRAIN_WEIGHT_DECAY)
    p.add_argument("--warmup-ratio", type=float, default=TRAIN_WARMUP_RATIO)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    args = p.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.prompt_length < 1 or args.basis_size < 1:
        p.error("epochs, batch-size, prompt-length, and basis-size must be positive")
    if args.learning_rate <= 0 or args.temperature <= 0 or args.weight_decay < 0:
        p.error("learning-rate and temperature must be positive; weight-decay must be non-negative")
    if not 0 <= args.warmup_ratio < 1:
        p.error("--warmup-ratio must be in [0, 1)")
    return args


def run(args):
    distributed = int(__import__("os").environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    rank = dist.get_rank() if distributed else 0
    set_seed(args.seed + rank)
    out = Path(args.output_dir).expanduser().resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    stage_outputs = [out / "source_prompt.pt", out / "training_history.json"]
    existing = [path for path in stage_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = SuperPosModel(args.model, args.prompt_length, args.basis_size,
                          args.temperature, args.seed, dtype=dtype).to(device)
    train_set, valid_set = load_classification_data(args.dataset, model.tokenizer, args.max_length)
    sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed) if distributed else None
    train_loader = build_loader(train_set, args.batch_size, shuffle=sampler is None,
                                seed=args.seed, sampler=sampler)
    valid_loader = build_loader(valid_set, args.batch_size)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank]) if distributed else model
    optimizer = torch.optim.AdamW([model.prompt_logits], lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    from transformers import get_linear_schedule_with_warmup

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    best = -math.inf
    history = []
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        wrapped.train()
        loss_sum = count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = wrapped(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            labels = batch["label"].to(logits.device)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Source SuperPos loss contains NaN or Inf")
            loss.backward()
            if model.prompt_logits.grad is None or not torch.isfinite(model.prompt_logits.grad).all():
                raise FloatingPointError("Source prompt gradient is missing or non-finite")
            optimizer.step()
            if not torch.isfinite(model.prompt_logits).all():
                raise FloatingPointError("Source prompt logits became NaN or Inf")
            scheduler.step()
            loss_sum += float(loss.detach()) * labels.numel()
            count += labels.numel()
        if distributed:
            dist.barrier()
        totals = torch.tensor([loss_sum, count], dtype=torch.float64, device=device)
        if distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if rank == 0:
            score = accuracy(model, valid_loader, device)
            row = {"epoch": epoch + 1,
                   "train_loss": float(totals[0] / totals[1].clamp_min(1)),
                   "validation_accuracy": score,
                   "learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(row)
            print(json.dumps(row))
            if score > best:
                best = score
                save_source_prompt(model, out / "source_prompt.pt", best, args.dataset, vars(args))
        if distributed:
            # Keep every rank at the same epoch boundary while rank 0 validates.
            dist.barrier()
    if rank == 0:
        (out / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if distributed:
        dist.destroy_process_group()
    return out / "source_prompt.pt"


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
