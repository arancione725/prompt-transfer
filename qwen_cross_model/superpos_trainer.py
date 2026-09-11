"""Train source prompts, including the SuperPos prompt variant, with DDP support."""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from .config import (
    PROMPT_LR, PROMPT_EPOCHS, PROMPT_BATCH_SIZE,
    PROMPT_WARMUP_RATIO, PROMPT_WEIGHT_DECAY, PROMPT_ANCHOR_WEIGHT,
    PROMPT_INIT_TEXT, NUM_LABELS, OUTPUT_DIR,
    SUPERPOS_TEMPERATURE, DEFAULT_SEED,
)
from .utils import (
    PromptQwenWrapper,
    load_dataset_by_name,
    save_prompt,
    save_prompt_checkpoint,
    load_prompt_checkpoint,
    format_time,
    train_one_epoch,
    evaluate_model,
    build_dataloader,
    set_seed,
)


def _is_ddp():
    return dist.is_available() and dist.is_initialized()


def _barrier():
    if _is_ddp():
        dist.barrier()


def train_prompt(
    model_name,
    dataset_name="sst2",
    prompt_len=100,
    learning_rate=PROMPT_LR,
    num_epochs=PROMPT_EPOCHS,
    batch_size=PROMPT_BATCH_SIZE,
    warmup_ratio=PROMPT_WARMUP_RATIO,
    weight_decay=PROMPT_WEIGHT_DECAY,
    anchor_weight=PROMPT_ANCHOR_WEIGHT,
    init_text=None,
    output_dir=OUTPUT_DIR,
    device=None,
    num_gpus=0,
    resume_from=None,
    seed=DEFAULT_SEED,
):
    set_seed(seed)
    # ---- DDP setup ----
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        is_main = (local_rank == 0)
    else:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    world_size = dist.get_world_size() if _is_ddp() else 1

    if is_main:
        print(f"\n{'='*60}")
        print(f"Step 1: Prompt Tuning on {model_name}")
        print(f"Dataset: {dataset_name}  |  Prompt len: {prompt_len}")
        print(f"GPUs: {world_size} (DDP)  |  per-GPU batch: {batch_size}")
        print(f"Epochs: {num_epochs}  |  LR: {learning_rate}")
        if resume_from:
            print(f"Resuming from: {resume_from}")
        print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"prompt_{model_name.replace('/', '_')}_{dataset_name}.pt")
    ckpt_path = os.path.join(output_dir, f"prompt_checkpoint.pt")

    # 1. Build model on this GPU
    if init_text is None:
        init_text = PROMPT_INIT_TEXT.get(dataset_name)
    wrapper = PromptQwenWrapper(model_name, prompt_len=prompt_len, init_text=init_text)
    wrapper.to(device)

    # DDP wrap (backbone frozen → find_unused_parameters=True)
    _raw = wrapper  # underlying module for .tokenizer access
    if _is_ddp():
        wrapper = DDP(wrapper, device_ids=[local_rank], find_unused_parameters=True)
        train_model = wrapper
    else:
        train_model = wrapper

    # 2. Load data (with DistributedSampler in DDP mode)
    train_ds, eval_ds = load_dataset_by_name(dataset_name, _raw.tokenizer)

    if _is_ddp():
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
        train_loader = build_dataloader(train_ds, batch_size, shuffle=False, sampler=train_sampler)
    else:
        train_loader = build_dataloader(train_ds, batch_size, shuffle=True, seed=seed)

    eval_loader = build_dataloader(eval_ds, batch_size * 2, shuffle=False)
    train_eval_subset = torch.utils.data.Subset(train_ds, range(min(1000, len(train_ds))))
    train_eval_loader = build_dataloader(train_eval_subset, batch_size * 2, shuffle=False)

    # 3. Optimizer & scheduler
    optimizer = AdamW(wrapper.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 4. Resume
    start_epoch = 0
    best_acc = 0.0

    if resume_from and os.path.exists(resume_from):
        resume_model = wrapper.module if isinstance(wrapper, DDP) else wrapper
        ckpt_epoch, ckpt_best = load_prompt_checkpoint(
            resume_from, resume_model, optimizer, scheduler, device
        )
        start_epoch = ckpt_epoch
        best_acc = ckpt_best
        if is_main:
            print(f"  Resumed from epoch {start_epoch} (best acc so far: {best_acc:.4f})")

        remaining_steps = (num_epochs - start_epoch) * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=remaining_steps)

    # 5. Training
    total_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        if _is_ddp():
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(train_model, train_loader, optimizer, scheduler, device,
                                     desc=f"Epoch {epoch+1}/{num_epochs}", anchor_weight=anchor_weight)

        # Eval — all ranks participate (use underlying module)
        eval_model = wrapper.module if isinstance(wrapper, DDP) else wrapper
        train_acc = evaluate_model(eval_model, train_eval_loader, device, desc="Eval train")
        eval_acc = evaluate_model(eval_model, eval_loader, device, desc="Eval val")

        if is_main:
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            best_marker = ""
            if eval_acc > best_acc:
                best_acc = eval_acc
                save_prompt(eval_model, save_path)
                best_marker = "  [*BEST]"

            # Report cos-sim for monitoring anchor effect
            with torch.no_grad():
                emb = eval_model.backbone.model.embed_tokens.weight.float()
                emb_norm = F.normalize(emb, dim=1)
                prompt_norm = F.normalize(eval_model.prompt_embeddings.weight, dim=1)
                cos_sim = (prompt_norm @ emb_norm.T).max(dim=1).values.mean()

            print(f"Epoch {epoch+1:2d}/{num_epochs}  |  "
                  f"loss={train_loss:.4f}  |  "
                  f"train_acc={train_acc:.4f}  |  "
                  f"eval_acc={eval_acc:.4f}  |  "
                  f"cos-sim={cos_sim:.4f}  |  "
                  f"lr={current_lr:.2e}  |  "
                  f"time={format_time(epoch_time)}{best_marker}")

            save_prompt_checkpoint(ckpt_path, epoch + 1, eval_model, optimizer, scheduler, best_acc)

        _barrier()

    if is_main:
        total_time = time.time() - total_start
        print(f"\nTotal time: {format_time(total_time)}  |  Best eval_acc: {best_acc:.4f}")
        print(f"Saved: {save_path}\n")

    metrics = {"best_eval_accuracy": best_acc if is_main else 0.0, "save_path": save_path}
    return wrapper.module if isinstance(wrapper, DDP) else wrapper, metrics


def train_superpos_prompt(
    model_name,
    dataset_name="sst2",
    prompt_len=100,
    m=128,
    learning_rate=PROMPT_LR,
    num_epochs=PROMPT_EPOCHS,
    batch_size=PROMPT_BATCH_SIZE,
    warmup_ratio=PROMPT_WARMUP_RATIO,
    weight_decay=PROMPT_WEIGHT_DECAY,
    output_dir=OUTPUT_DIR,
    device=None,
    num_gpus=0,
    resume_from=None,
    save_name=None,
    seed=DEFAULT_SEED,
):
    """Train SuperPos soft prompt: each prompt vector = weighted sum of m token embeddings.

    Anchor loss is NOT needed — prompt vectors are hard-constrained to the token manifold
    by construction (convex combination of real token embeddings).
    """
    import os
    from .utils import SuperPosPromptWrapper, save_superpos_prompt, train_one_epoch

    set_seed(seed)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        is_main = (local_rank == 0)
    else:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    world_size = dist.get_world_size() if _is_ddp() else 1

    if is_main:
        print(f"\n{'='*60}")
        print(f"SuperPos Prompt Tuning on {model_name}")
        print(f"Dataset: {dataset_name}  |  Prompt len: {prompt_len}  |  m: {m}")
        print(f"GPUs: {world_size}  |  per-GPU batch: {batch_size}")
        print(f"Epochs: {num_epochs}  |  LR: {learning_rate}")
        print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)
    if save_name:
        save_path = os.path.join(output_dir, save_name)
    else:
        save_path = os.path.join(output_dir, f"prompt_superpos_{model_name.replace('/', '_')}_{dataset_name}.pt")
    ckpt_path = os.path.join(output_dir, f"superpos_checkpoint.pt")

    # Build SuperPos wrapper
    wrapper = SuperPosPromptWrapper(
        model_name,
        prompt_len=prompt_len,
        num_labels=NUM_LABELS,
        m=m,
        temperature=SUPERPOS_TEMPERATURE,
        seed=seed,
    )
    wrapper.to(device)
    _raw = wrapper

    if _is_ddp():
        wrapper = DDP(wrapper, device_ids=[local_rank], find_unused_parameters=True)
        train_model = wrapper
    else:
        train_model = wrapper

    train_ds, eval_ds = load_dataset_by_name(dataset_name, _raw.tokenizer)
    if _is_ddp():
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
        train_loader = build_dataloader(train_ds, batch_size, shuffle=False, sampler=train_sampler)
    else:
        train_loader = build_dataloader(train_ds, batch_size, shuffle=True, seed=seed)
    eval_loader = build_dataloader(eval_ds, batch_size * 2, shuffle=False)
    train_eval_subset = torch.utils.data.Subset(train_ds, range(min(1000, len(train_ds))))
    train_eval_loader = build_dataloader(train_eval_subset, batch_size * 2, shuffle=False)

    optimizer = AdamW(train_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    start_epoch = 0
    best_acc = 0.0
    total_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        if _is_ddp():
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(train_model, train_loader, optimizer, scheduler, device,
                                     desc=f"Epoch {epoch+1}/{num_epochs}", anchor_weight=0.0)

        eval_model = wrapper.module if isinstance(wrapper, DDP) else wrapper
        train_acc = evaluate_model(eval_model, train_eval_loader, device, desc="Eval train")
        eval_acc = evaluate_model(eval_model, eval_loader, device, desc="Eval val")

        if is_main:
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start
            best_marker = ""
            if eval_acc > best_acc:
                best_acc = eval_acc
                save_superpos_prompt(eval_model, save_path)
                best_marker = "  [*BEST]"
            print(f"Epoch {epoch+1:2d}/{num_epochs}  |  "
                  f"loss={train_loss:.4f}  |  train_acc={train_acc:.4f}  |  "
                  f"eval_acc={eval_acc:.4f}  |  lr={current_lr:.2e}  |  "
                  f"time={format_time(epoch_time)}{best_marker}")

        _barrier()

    if is_main:
        total_time = time.time() - total_start
        print(f"\nTotal time: {format_time(total_time)}  |  Best eval_acc: {best_acc:.4f}")
        print(f"Saved: {save_path}\n")

    metrics = {"best_eval_accuracy": best_acc if is_main else 0.0, "save_path": save_path}
    return wrapper.module if isinstance(wrapper, DDP) else wrapper, metrics
