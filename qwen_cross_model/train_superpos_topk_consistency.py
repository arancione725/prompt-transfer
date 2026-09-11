#!/usr/bin/env python3
"""Train a source-model SuperPos prompt with random Top-K consistency.

The target model is never loaded.  Each batch performs two source-model
forward passes using the same trainable SuperPos logits:

    loss = full_ce_weight * CE(full_prompt, label)
         + topk_ce_weight * CE(random_topk_prompt, label)
         + js_weight * JS(full_logits, random_topk_logits)

The saved checkpoint is compatible with eval_direct.py and
eval_soft_bridge.py.
"""

import argparse
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import (
    DEFAULT_SEED,
    MODEL_1_5B,
    NUM_LABELS,
    OUTPUT_DIR,
    PROMPT_BATCH_SIZE,
    PROMPT_EPOCHS,
    PROMPT_LEN,
    PROMPT_LR,
    PROMPT_WARMUP_RATIO,
    PROMPT_WEIGHT_DECAY,
    SUPERPOS_M,
    SUPERPOS_TEMPERATURE,
)
from .utils import (
    SuperPosPromptWrapper,
    build_dataloader,
    evaluate_model,
    format_time,
    load_dataset_by_name,
    set_seed,
)


DEFAULT_TOPK_VALUES = [3, 5, 8, 10]


def _is_ddp():
    return dist.is_available() and dist.is_initialized()


def _barrier():
    if _is_ddp():
        dist.barrier()


class TopKConsistencySuperPosWrapper(SuperPosPromptWrapper):
    """SuperPos wrapper that can return full and truncated-prompt logits."""

    def _build_prompt_pair(self, topk):
        if not 1 <= topk <= self.m:
            raise ValueError(f"topk must be in [1, {self.m}], got {topk}")

        basis_embeddings = self.backbone.model.embed_tokens.weight[
            self.sampled_ids
        ].float()
        probabilities = F.softmax(
            self.prompt_weights / self.temperature,
            dim=-1,
        )
        full_prompt = probabilities @ basis_embeddings

        topk_values, topk_indices = probabilities.topk(topk, dim=-1)
        topk_probabilities = topk_values / topk_values.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)
        selected_embeddings = basis_embeddings[topk_indices]
        topk_prompt = (
            topk_probabilities.unsqueeze(-1) * selected_embeddings
        ).sum(dim=1)
        return full_prompt, topk_prompt

    def forward(
        self,
        input_ids,
        attention_mask,
        prompt_embeds=None,
        topk=None,
    ):
        if topk is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_embeds=prompt_embeds,
            )
        if prompt_embeds is not None:
            raise ValueError("prompt_embeds and topk cannot be supplied together")

        full_prompt, topk_prompt = self._build_prompt_pair(topk)
        batch_size = input_ids.size(0)
        full_prompt = full_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        topk_prompt = topk_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        full_logits = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_embeds=full_prompt,
        )
        topk_logits = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_embeds=topk_prompt,
        )
        return full_logits, topk_logits


def js_divergence(first_logits, second_logits, temperature=2.0):
    """Symmetric Jensen-Shannon divergence between two class distributions."""
    if temperature <= 0:
        raise ValueError(f"JS temperature must be positive, got {temperature}")

    first_log_prob = F.log_softmax(first_logits / temperature, dim=-1)
    second_log_prob = F.log_softmax(second_logits / temperature, dim=-1)
    first_prob = first_log_prob.exp()
    second_prob = second_log_prob.exp()
    mixture = 0.5 * (first_prob + second_prob)
    mixture_log_prob = mixture.clamp_min(1e-12).log()

    first_to_mixture = (
        first_prob * (first_log_prob - mixture_log_prob)
    ).sum(dim=-1).mean()
    second_to_mixture = (
        second_prob * (second_log_prob - mixture_log_prob)
    ).sum(dim=-1).mean()
    return 0.5 * (first_to_mixture + second_to_mixture) * temperature**2


def choose_topk(topk_values, rng, device, is_main):
    """Choose one K on rank 0 and broadcast it to every DDP rank."""
    if _is_ddp():
        chosen = rng.choice(topk_values) if is_main else 0
        chosen_tensor = torch.tensor(chosen, dtype=torch.long, device=device)
        dist.broadcast(chosen_tensor, src=0)
        return int(chosen_tensor.item())
    return rng.choice(topk_values)


def reduce_epoch_totals(totals, device):
    """Sum metric numerators and sample count across all ranks."""
    values = torch.tensor(totals, dtype=torch.float64, device=device)
    if _is_ddp():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values.cpu().tolist()


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    topk_values,
    topk_rng,
    full_ce_weight,
    topk_ce_weight,
    js_weight,
    js_temperature,
    is_main,
    epoch,
    num_epochs,
):
    model.train()
    total_loss = 0.0
    total_full_ce = 0.0
    total_topk_ce = 0.0
    total_js = 0.0
    total_samples = 0

    progress = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{num_epochs}",
        leave=False,
        disable=not is_main,
    )
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        selected_topk = choose_topk(topk_values, topk_rng, device, is_main)

        optimizer.zero_grad(set_to_none=True)
        full_logits, topk_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            topk=selected_topk,
        )
        full_ce = F.cross_entropy(full_logits, labels)
        topk_ce = F.cross_entropy(topk_logits, labels)
        consistency = js_divergence(
            full_logits,
            topk_logits,
            temperature=js_temperature,
        )
        loss = (
            full_ce_weight * full_ce
            + topk_ce_weight * topk_ce
            + js_weight * consistency
        )
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_full_ce += full_ce.item() * batch_size
        total_topk_ce += topk_ce.item() * batch_size
        total_js += consistency.item() * batch_size
        total_samples += batch_size

        if is_main:
            progress.set_postfix(
                {
                    "K": selected_topk,
                    "loss": f"{loss.item():.4f}",
                    "full": f"{full_ce.item():.4f}",
                    "topk": f"{topk_ce.item():.4f}",
                    "js": f"{consistency.item():.4f}",
                }
            )

    totals = reduce_epoch_totals(
        [
            total_loss,
            total_full_ce,
            total_topk_ce,
            total_js,
            total_samples,
        ],
        device,
    )
    denominator = max(totals[4], 1.0)
    return {
        "loss": totals[0] / denominator,
        "full_ce": totals[1] / denominator,
        "topk_ce": totals[2] / denominator,
        "js": totals[3] / denominator,
    }


def save_checkpoint(model, path, args, best_accuracy):
    """Save an eval-compatible checkpoint plus objective metadata."""
    torch.save(
        {
            "type": "superpos_lmhead",
            "format_version": 2,
            "weight_mode": "softmax",
            "temperature": float(model.temperature),
            "seed": model.seed,
            "source_model": model.model_name,
            "sampled_ids": model.sampled_ids.detach().cpu(),
            "prompt_weights": model.prompt_weights.detach().cpu(),
            "neg_ids": list(model.neg_ids),
            "pos_ids": list(model.pos_ids),
            "prompt_len": model.prompt_len,
            "m": model.m,
            "hidden_size": model.hidden_size,
            "training_objective": "ce_random_topk_ce_js",
            "topk_values": list(args.topk_values),
            "full_ce_weight": float(args.full_ce_weight),
            "topk_ce_weight": float(args.topk_ce_weight),
            "js_weight": float(args.js_weight),
            "js_temperature": float(args.js_temperature),
            "best_source_eval_accuracy": float(best_accuracy),
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SuperPos with CE + random Top-K CE + JS consistency"
    )
    parser.add_argument(
        "--src",
        "--src-model",
        dest="src_model",
        default=MODEL_1_5B,
    )
    parser.add_argument("--dataset", default="sst2")
    parser.add_argument("--prompt-len", type=int, default=PROMPT_LEN)
    parser.add_argument("--superpos-m", type=int, default=SUPERPOS_M)
    parser.add_argument("--prompt-lr", type=float, default=PROMPT_LR)
    parser.add_argument("--prompt-epochs", type=int, default=PROMPT_EPOCHS)
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=PROMPT_BATCH_SIZE,
        help="Per-GPU batch size",
    )
    parser.add_argument("--warmup-ratio", type=float, default=PROMPT_WARMUP_RATIO)
    parser.add_argument("--weight-decay", type=float, default=PROMPT_WEIGHT_DECAY)
    parser.add_argument("--temperature", type=float, default=SUPERPOS_TEMPERATURE)
    parser.add_argument(
        "--topk-values",
        type=int,
        nargs="+",
        default=DEFAULT_TOPK_VALUES,
        help="Candidate K values sampled once per batch",
    )
    parser.add_argument("--full-ce-weight", type=float, default=0.5)
    parser.add_argument("--topk-ce-weight", type=float, default=1.0)
    parser.add_argument("--js-weight", type=float, default=0.1)
    parser.add_argument("--js-temperature", type=float, default=2.0)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--save-name", default="superpos_topk_consistency.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def validate_args(args):
    if args.prompt_epochs < 1:
        raise ValueError("--prompt-epochs must be at least 1")
    if args.prompt_batch_size < 1:
        raise ValueError("--prompt-batch-size must be at least 1")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.full_ce_weight < 0 or args.topk_ce_weight < 0 or args.js_weight < 0:
        raise ValueError("Loss weights must be non-negative")
    if args.js_temperature <= 0:
        raise ValueError("--js-temperature must be positive")
    if not args.topk_values:
        raise ValueError("--topk-values cannot be empty")
    if any(k < 1 or k > args.superpos_m for k in args.topk_values):
        raise ValueError(
            f"Every K must be in [1, {args.superpos_m}], got {args.topk_values}"
        )
    args.topk_values = list(dict.fromkeys(args.topk_values))


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        is_main = local_rank == 0
    else:
        device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        is_main = True
    world_size = dist.get_world_size() if _is_ddp() else 1

    if is_main:
        print("\n" + "=" * 70)
        print("SuperPos training: CE + random Top-K CE + JS consistency")
        print(f"Source model:       {args.src_model}")
        print(f"Dataset:            {args.dataset}")
        print(f"GPUs:               {world_size}")
        print(f"Per-GPU batch:      {args.prompt_batch_size}")
        print(f"Epochs / LR:        {args.prompt_epochs} / {args.prompt_lr}")
        print(f"Random K values:    {args.topk_values}")
        print(f"Full CE weight:     {args.full_ce_weight}")
        print(f"Top-K CE weight:    {args.topk_ce_weight}")
        print(f"JS weight / temp:   {args.js_weight} / {args.js_temperature}")
        print("=" * 70 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.save_name)

    raw_model = TopKConsistencySuperPosWrapper(
        args.src_model,
        prompt_len=args.prompt_len,
        num_labels=NUM_LABELS,
        m=args.superpos_m,
        temperature=args.temperature,
        seed=args.seed,
    ).to(device)
    train_model = (
        DDP(raw_model, device_ids=[local_rank], find_unused_parameters=False)
        if _is_ddp()
        else raw_model
    )

    train_dataset, eval_dataset = load_dataset_by_name(
        args.dataset,
        raw_model.tokenizer,
    )
    if _is_ddp():
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            seed=args.seed,
        )
        train_loader = build_dataloader(
            train_dataset,
            args.prompt_batch_size,
            shuffle=False,
            sampler=train_sampler,
        )
    else:
        train_sampler = None
        train_loader = build_dataloader(
            train_dataset,
            args.prompt_batch_size,
            shuffle=True,
            seed=args.seed,
        )

    eval_loader = build_dataloader(
        eval_dataset,
        args.prompt_batch_size * 2,
        shuffle=False,
    )
    train_eval_subset = torch.utils.data.Subset(
        train_dataset,
        range(min(1000, len(train_dataset))),
    )
    train_eval_loader = build_dataloader(
        train_eval_subset,
        args.prompt_batch_size * 2,
        shuffle=False,
    )

    trainable_parameters = [
        parameter for parameter in train_model.parameters() if parameter.requires_grad
    ]
    optimizer = AdamW(
        trainable_parameters,
        lr=args.prompt_lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.prompt_epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps,
    )

    topk_rng = random.Random(args.seed)
    best_accuracy = -1.0
    total_start = time.time()

    for epoch in range(args.prompt_epochs):
        epoch_start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        metrics = train_one_epoch(
            train_model,
            train_loader,
            optimizer,
            scheduler,
            device,
            args.topk_values,
            topk_rng,
            args.full_ce_weight,
            args.topk_ce_weight,
            args.js_weight,
            args.js_temperature,
            is_main,
            epoch,
            args.prompt_epochs,
        )

        train_accuracy = evaluate_model(
            raw_model,
            train_eval_loader,
            device,
            desc="Eval train",
        )
        eval_accuracy = evaluate_model(
            raw_model,
            eval_loader,
            device,
            desc="Eval val",
        )

        if is_main:
            best_marker = ""
            if eval_accuracy > best_accuracy:
                best_accuracy = eval_accuracy
                save_checkpoint(raw_model, save_path, args, best_accuracy)
                best_marker = "  [*BEST]"

            print(
                f"Epoch {epoch + 1:2d}/{args.prompt_epochs} | "
                f"loss={metrics['loss']:.4f} | "
                f"full_ce={metrics['full_ce']:.4f} | "
                f"topk_ce={metrics['topk_ce']:.4f} | "
                f"js={metrics['js']:.4f} | "
                f"train_acc={train_accuracy:.4f} | "
                f"eval_acc={eval_accuracy:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"time={format_time(time.time() - epoch_start)}{best_marker}"
            )

        _barrier()

    if is_main:
        print(
            f"\nTotal time: {format_time(time.time() - total_start)} | "
            f"Best source eval_acc: {best_accuracy:.4f}"
        )
        print(f"Saved: {save_path}\n")

    _barrier()
    if _is_ddp():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
