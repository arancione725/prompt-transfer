"""Cross-model prompt projector and training with DDP (single-module DDP wrapping)."""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from .config import (
    PROJECTOR_LR, PROJECTOR_EPOCHS, PROJECTOR_BATCH_SIZE,
    PROJECTOR_WARMUP_RATIO, PROJECTOR_WEIGHT_DECAY, OUTPUT_DIR,
    PROMPT_LEN, NUM_LABELS, DEFAULT_SEED,
)
from .utils import (
    PromptQwenWrapper,
    load_dataset_by_name,
    save_prompt,
    save_projector_checkpoint,
    load_projector_checkpoint,
    format_time,
    evaluate_model,
    build_dataloader,
    activate_superpos_weights,
    get_verbalizer_ids,
    set_prompt_tensor,
    set_seed,
)
from .align import get_embedding_matrix


def _is_ddp():
    return dist.is_available() and dist.is_initialized()


def _barrier():
    if _is_ddp():
        dist.barrier()


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------

class PromptProjector(nn.Module):
    """Map a soft prompt from source-model embedding space to target-model space.

    Architecture: flatten -> Linear(768) -> LeakyReLU -> Linear(target_dim) -> reshape -> clamp(-10,10)
    """

    def __init__(self, prompt_len, src_hidden, tgt_hidden):
        super().__init__()
        in_dim = prompt_len * src_hidden
        out_dim = prompt_len * tgt_hidden
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 768),
            nn.LeakyReLU(),
            nn.Linear(768, out_dim),
        )
        self.prompt_len = prompt_len
        self.tgt_hidden = tgt_hidden

    def forward(self, src_prompt_emb):
        flat = src_prompt_emb.flatten()
        out = self.proj(flat)
        out = out.clamp(-10, 10)
        return out.view(self.prompt_len, self.tgt_hidden)


# ---------------------------------------------------------------------------
# Pipeline (projector + target model) for single-module DDP wrapping
# ---------------------------------------------------------------------------

class ProjectorPipeline(nn.Module):
    """Bundles projector + target model so DDP wraps all trainable params in one module.

    DDP calls prepare_for_forward() on this wrapper, ensuring proper gradient sync.
    """

    def __init__(self, projector, tgt_wrapper, src_prompt_emb):
        super().__init__()
        self.projector = projector
        self.backbone = tgt_wrapper.backbone
        self.classification_head = tgt_wrapper.classification_head
        self.prompt_len = tgt_wrapper.prompt_len
        self.backbone_dtype = tgt_wrapper.backbone.dtype
        # src_prompt_emb is a buffer so DDP handles device placement
        self.register_buffer("src_prompt_emb", src_prompt_emb.clone())

    def forward(self, input_ids, attention_mask):
        B = input_ids.size(0)

        # 1. Project source prompt -> target space
        projected = self.projector(self.src_prompt_emb)  # [L, H_tgt] fp32
        projected = projected.to(dtype=self.backbone_dtype)  # fp16

        # 2. Build prompt batch
        projected_batch = projected.unsqueeze(0).expand(B, -1, -1)  # [B, L, H_tgt]

        # 3. Word embeddings
        word_embeds = self.backbone.model.embed_tokens(input_ids)  # [B, N, H_tgt]

        # 4. Concatenate [prompt | text]
        inputs_embeds = torch.cat([projected_batch, word_embeds], dim=1)

        # 5. Attention mask
        prompt_mask = torch.ones(B, self.prompt_len, dtype=attention_mask.dtype, device=attention_mask.device)
        extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # 6. Backbone forward
        outputs = self.backbone(inputs_embeds=inputs_embeds, attention_mask=extended_mask)

        # 7. Last text position hidden -> classification
        last_pos_hidden = outputs.hidden_states[-1][:, -1, :]
        logits = self.classification_head(last_pos_hidden.float())

        return logits


# ---------------------------------------------------------------------------
# Projector training
# ---------------------------------------------------------------------------

def train_projector(
    src_model_name,
    tgt_model_name,
    src_prompt_path,
    dataset_name="sst2",
    prompt_len=PROMPT_LEN,
    num_labels=NUM_LABELS,
    learning_rate=PROJECTOR_LR,
    num_epochs=PROJECTOR_EPOCHS,
    batch_size=PROJECTOR_BATCH_SIZE,
    warmup_ratio=PROJECTOR_WARMUP_RATIO,
    weight_decay=PROJECTOR_WEIGHT_DECAY,
    output_dir=OUTPUT_DIR,
    device=None,
    num_gpus=0,
    resume_from=None,
):
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
        print(f"Step 2: Cross-Model Projector Training")
        print(f"Source: {src_model_name}  ->  Target: {tgt_model_name}")
        print(f"GPUs: {world_size} (DDP)  |  per-GPU batch: {batch_size}")
        print(f"Epochs: {num_epochs}  |  LR: {learning_rate}")
        if resume_from:
            print(f"Resuming from: {resume_from}")
        print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)
    proj_save_path = os.path.join(output_dir, "projector.pt")
    prompt_save_path = os.path.join(output_dir, "projected_prompt.pt")
    ckpt_path = os.path.join(output_dir, "projector_checkpoint.pt")

    # 1. Load source prompt
    src_checkpoint = torch.load(src_prompt_path, map_location="cpu")
    src_prompt_emb = src_checkpoint["prompt_embeddings"]["weight"].clone()
    src_hidden = src_prompt_emb.size(1)

    if is_main:
        print(f"  src_prompt: shape={src_prompt_emb.shape}  min={src_prompt_emb.min().item():.4f}  "
              f"max={src_prompt_emb.max().item():.4f}  has_nan={torch.isnan(src_prompt_emb).any().item()}")

    # 2. Build target model (GPU)
    tgt_wrapper = PromptQwenWrapper(tgt_model_name, prompt_len=prompt_len, num_labels=num_labels)
    tgt_wrapper.to(device)
    tgt_hidden = tgt_wrapper.hidden_size

    # Freeze backbone (already frozen, just to be safe)
    for p in tgt_wrapper.backbone.parameters():
        p.requires_grad = False

    # 3. Build projector (fp32)
    projector = PromptProjector(prompt_len, src_hidden, tgt_hidden).to(device=device)

    # 4. Build pipeline and wrap with DDP
    pipeline = ProjectorPipeline(projector, tgt_wrapper, src_prompt_emb).to(device)
    if _is_ddp():
        train_model = DDP(pipeline, device_ids=[local_rank], find_unused_parameters=True)
    else:
        train_model = pipeline

    # 5. Load data
    train_ds, eval_ds = load_dataset_by_name(dataset_name, tgt_wrapper.tokenizer)

    if _is_ddp():
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
        train_loader = build_dataloader(train_ds, batch_size, shuffle=False, sampler=train_sampler)
    else:
        train_loader = build_dataloader(train_ds, batch_size, shuffle=True)

    eval_loader = build_dataloader(eval_ds, batch_size * 2, shuffle=False)
    train_eval_subset = torch.utils.data.Subset(train_ds, range(min(1000, len(train_ds))))
    train_eval_loader = build_dataloader(train_eval_subset, batch_size * 2, shuffle=False)

    # 6. Optimizer & scheduler (over DDP params)
    optimizer = AdamW(train_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps) if warmup_steps > 0 else None

    # 7. Resume
    start_epoch = 0
    best_acc = 0.0

    if resume_from and os.path.exists(resume_from):
        ckpt_epoch, ckpt_best, _ = load_projector_checkpoint(
            resume_from, projector, tgt_wrapper.classification_head, optimizer, scheduler, device
        )
        start_epoch = ckpt_epoch
        best_acc = ckpt_best
        if is_main:
            print(f"  Resumed from epoch {start_epoch} (best acc so far: {best_acc:.4f})")
        remaining_steps = (num_epochs - start_epoch) * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=remaining_steps)

    # 8. Training loop
    loss_fn = nn.CrossEntropyLoss()
    total_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        train_model.train()
        total_loss = 0.0

        if _is_ddp():
            train_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, disable=not is_main)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = train_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()

            # Gradient clipping
            trainable_params = [p for p in train_model.parameters() if p.requires_grad]
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            if is_main:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        # Eval — all ranks participate
        eval_model = pipeline  # use underlying module (non-DDP) for eval
        train_acc = evaluate_model(eval_model, train_eval_loader, device, desc="Eval train")
        eval_acc = evaluate_model(eval_model, eval_loader, device, desc="Eval val")

        if is_main:
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            best_marker = ""
            if eval_acc > best_acc:
                best_acc = eval_acc
                torch.save(projector.state_dict(), proj_save_path)
                save_prompt(tgt_wrapper, os.path.join(output_dir, "target_head.pt"))
                best_marker = "  [*BEST]"

            print(f"Epoch {epoch+1:2d}/{num_epochs}  |  "
                  f"loss={avg_loss:.4f}  |  "
                  f"train_acc={train_acc:.4f}  |  "
                  f"eval_acc={eval_acc:.4f}  |  "
                  f"lr={current_lr:.2e}  |  "
                  f"time={format_time(epoch_time)}{best_marker}")

            save_projector_checkpoint(
                ckpt_path, epoch + 1, projector, tgt_wrapper.classification_head,
                optimizer, scheduler, best_acc, src_prompt_emb
            )

        _barrier()

    if is_main:
        total_time = time.time() - total_start
        print(f"\nTotal time: {format_time(total_time)}  |  Best cross-model eval_acc: {best_acc:.4f}")
        print(f"Projector: {proj_save_path}\n")

    metrics = {"best_cross_eval_accuracy": best_acc if is_main else 0.0, "projector_path": proj_save_path}
    return projector, metrics


# ---------------------------------------------------------------------------
# SuperPos -> target LM-head projector
# ---------------------------------------------------------------------------

class SuperPosProjector(nn.Module):
    """Map a source SuperPos prompt from source to target embedding space."""

    def __init__(self, prompt_len, src_dim, tgt_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(prompt_len * src_dim, 768),
            nn.LeakyReLU(),
            nn.Linear(768, prompt_len * tgt_dim),
        )
        self.prompt_len = prompt_len
        self.tgt_dim = tgt_dim

    def forward(self, src_prompt_emb):
        return self.proj(src_prompt_emb.flatten()).view(self.prompt_len, self.tgt_dim)


class ProjectorLMPipeline(nn.Module):
    """SuperPos projector with a frozen target backbone and native LM Head."""

    def __init__(self, projector, tgt_backbone, src_prompt_emb,
                 pos_ids, neg_ids, prompt_len):
        super().__init__()
        self.projector = projector
        self.backbone = tgt_backbone
        self.backbone_dtype = next(tgt_backbone.parameters()).dtype
        self.prompt_len = prompt_len
        self.register_buffer("src_prompt_emb", src_prompt_emb.clone())
        self.register_buffer("pos_ids", torch.tensor(sorted(pos_ids), dtype=torch.long))
        self.register_buffer("neg_ids", torch.tensor(sorted(neg_ids), dtype=torch.long))

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        projected = self.projector(self.src_prompt_emb).to(dtype=self.backbone_dtype)
        word_embeds = self.backbone.model.embed_tokens(input_ids)
        prompt_batch = projected.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([prompt_batch, word_embeds], dim=1)
        prompt_mask = torch.ones(
            batch_size,
            self.prompt_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        outputs = self.backbone(inputs_embeds=inputs_embeds, attention_mask=extended_mask)
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        lm_logits = self.backbone.lm_head(last_hidden)
        pos_score = lm_logits[:, self.pos_ids].max(dim=-1).values
        neg_score = lm_logits[:, self.neg_ids].max(dim=-1).values
        return torch.stack([neg_score, pos_score], dim=1)


@torch.no_grad()
def evaluate_projector_lmhead(model, dataloader, device, desc="Eval"):
    model.eval()
    correct = 0
    total = 0
    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        predictions = model(input_ids=input_ids, attention_mask=attention_mask).argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


def main_superpos_projector():
    """Train the optional MLP projector from a SuperPos checkpoint."""
    parser = argparse.ArgumentParser(description="Train SuperPos -> target LM-head projector")
    parser.add_argument(
        "--src-prompt",
        default="outputs_qwen/superpos_prompt_transfer_main_clean/prompt_transfer_main_clean.pt",
    )
    parser.add_argument("--src-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--tgt-model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--dataset", default="sst2")
    parser.add_argument("--prompt-len", type=int, default=PROMPT_LEN)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--weight-mode", default="softmax", choices=["softmax", "relu"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--pos-word", default="positive")
    parser.add_argument("--neg-word", default="negative")
    parser.add_argument(
        "--save-name",
        default="projector_superpos_lmhead.pt",
        help="Filename for the best projector state dict",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    set_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        is_main = local_rank == 0
    else:
        device = torch.device(args.device) if args.device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        is_main = True
    world_size = dist.get_world_size() if _is_ddp() else 1

    if is_main:
        print(f"\n{'='*60}")
        print("Projector Training — SuperPos -> Target (LM Head)")
        print(f"Source prompt:  {args.src_prompt}")
        print(f"Source model:   {args.src_model}")
        print(f"Target model:   {args.tgt_model}")
        print(f"Dataset:        {args.dataset}")
        print(f"GPUs: {world_size} (DDP)  |  per-GPU batch: {args.batch_size}")
        print(f"Epochs: {args.epochs}  |  LR: {args.lr}")
        print(f"Weight mode:    {args.weight_mode}")
        print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    proj_save_path = os.path.join(args.output_dir, args.save_name)
    ckpt = torch.load(args.src_prompt, map_location="cpu")
    sampled_ids = ckpt["sampled_ids"]
    if args.weight_mode == "softmax":
        weights, temperature = activate_superpos_weights(ckpt)
    else:
        weights = F.relu(ckpt["prompt_weights"].float())
        temperature = None

    src_emb_full, src_hidden = get_embedding_matrix(args.src_model)
    src_prompt_emb = weights @ src_emb_full[sampled_ids].float()
    if is_main:
        print(f"  Weights:      {weights.shape}")
        if temperature is not None:
            print(f"  Temperature:  {temperature}")
        print(f"  src_prompt:   shape={src_prompt_emb.shape}  "
              f"norm={src_prompt_emb.norm(p=2, dim=-1).mean():.4f}")

    tgt_backbone = AutoModelForCausalLM.from_pretrained(
        args.tgt_model, torch_dtype=torch.float16
    )
    tgt_backbone.config.output_hidden_states = True
    tgt_backbone.gradient_checkpointing_enable()
    for parameter in tgt_backbone.parameters():
        parameter.requires_grad = False
    tgt_backbone.to(device)
    tgt_hidden = tgt_backbone.config.hidden_size

    tokenizer = AutoTokenizer.from_pretrained(args.tgt_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    pos_ids = get_verbalizer_ids(tokenizer, args.pos_word)
    neg_ids = get_verbalizer_ids(tokenizer, args.neg_word)

    projector = SuperPosProjector(args.prompt_len, src_hidden, tgt_hidden)
    pipeline = ProjectorLMPipeline(
        projector, tgt_backbone, src_prompt_emb,
        pos_ids, neg_ids, args.prompt_len,
    ).to(device)
    train_model = DDP(pipeline, device_ids=[local_rank], find_unused_parameters=True) \
        if _is_ddp() else pipeline

    train_ds, eval_ds = load_dataset_by_name(args.dataset, tokenizer)
    if _is_ddp():
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=local_rank, shuffle=True
        )
        train_loader = build_dataloader(
            train_ds, args.batch_size, shuffle=False, sampler=train_sampler
        )
    else:
        train_sampler = None
        train_loader = build_dataloader(
            train_ds, args.batch_size, shuffle=True, seed=args.seed
        )
    eval_loader = build_dataloader(eval_ds, args.batch_size * 2, shuffle=False)
    train_subset = torch.utils.data.Subset(train_ds, range(min(1000, len(train_ds))))
    train_eval_loader = build_dataloader(train_subset, args.batch_size * 2, shuffle=False)

    optimizer = AdamW(train_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = (
        get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        if warmup_steps > 0 else None
    )
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0
    total_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_model.train()
        total_loss = 0.0
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            leave=False,
            disable=not is_main,
        )
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = train_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in train_model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            if is_main:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_acc = evaluate_projector_lmhead(
            pipeline, train_eval_loader, device, desc="Eval train"
        )
        eval_acc = evaluate_projector_lmhead(
            pipeline, eval_loader, device, desc="Eval val"
        )
        if is_main:
            best_marker = ""
            if eval_acc > best_acc:
                best_acc = eval_acc
                torch.save(projector.state_dict(), proj_save_path)
                best_marker = "  [*BEST]"
            print(
                f"Epoch {epoch + 1:2d}/{args.epochs}  | "
                f"loss={total_loss / len(train_loader):.4f}  | "
                f"train_acc={train_acc:.4f}  | eval_acc={eval_acc:.4f}  | "
                f"time={format_time(time.time() - epoch_start)}{best_marker}"
            )
        _barrier()

    if is_main:
        print(
            f"\nTotal time: {format_time(time.time() - total_start)}  | "
            f"Best eval_acc: {best_acc:.4f}"
        )
        print(f"Projector: {proj_save_path}\n")


# ---------------------------------------------------------------------------
# Legacy standard-projector evaluation helpers, kept in this module so the
# projector implementation and its evaluation code have one source of truth.
# ---------------------------------------------------------------------------

def evaluate_direct(model_name, prompt_path, dataset_name="sst2", device=None):
    """Evaluate a standard prompt directly on its source model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = load_prompt(model_name, prompt_path, device=device)
    _, eval_ds = load_dataset_by_name(dataset_name, wrapper.tokenizer)
    acc = evaluate_model(wrapper, build_dataloader(eval_ds, 32, shuffle=False), device)
    print(f"  Accuracy: {acc:.4f}")
    return acc


def evaluate_cross_model(
    src_model_name,
    tgt_model_name,
    src_prompt_path,
    projector_path,
    dataset_name="sst2",
    device=None,
):
    """Evaluate the ordinary classification-head projector on the target model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_ckpt = torch.load(src_prompt_path, map_location="cpu")
    src_prompt_emb = src_ckpt["prompt_embeddings"]["weight"].clone()
    proj_ckpt = torch.load(projector_path, map_location="cpu")
    tgt_wrapper = PromptQwenWrapper(
        tgt_model_name, prompt_len=src_prompt_emb.size(0), num_labels=NUM_LABELS
    ).to(device)
    head_path = projector_path.replace("projector.pt", "target_head.pt")
    head_ckpt = torch.load(head_path, map_location=device)
    if "classification_head" in head_ckpt:
        tgt_wrapper.classification_head.load_state_dict(head_ckpt["classification_head"])
    projector = PromptProjector(
        src_prompt_emb.size(0), src_prompt_emb.size(1), tgt_wrapper.hidden_size
    ).to(device)
    projector.load_state_dict(proj_ckpt)
    with torch.no_grad():
        set_prompt_tensor(tgt_wrapper, projector(src_prompt_emb.to(device)))
    _, eval_ds = load_dataset_by_name(dataset_name, tgt_wrapper.tokenizer)
    acc = evaluate_model(
        tgt_wrapper, build_dataloader(eval_ds, 32, shuffle=False), device
    )
    print(f"  Accuracy: {acc:.4f}")
    return acc


def quick_compare(
    model_1_5b,
    model_7b,
    prompt_1_5b_path,
    projector_path=None,
    dataset_name="sst2",
    device=None,
):
    """Print standard prompt/projector evaluation results."""
    from tabulate import tabulate

    rows = []
    if prompt_1_5b_path and model_1_5b:
        rows.append([
            "1.5B direct",
            f"{evaluate_direct(model_1_5b, prompt_1_5b_path, dataset_name, device):.4f}",
        ])
    if projector_path and model_7b:
        rows.append([
            "1.5B -> 7B (projected)",
            f"{evaluate_cross_model(model_1_5b, model_7b, prompt_1_5b_path, projector_path, dataset_name, device):.4f}",
        ])
    print("\n" + tabulate(rows, headers=["Setting", "Accuracy"], tablefmt="grid"))
    return rows


if __name__ == "__main__":
    main_superpos_projector()
