#!/usr/bin/env python3
"""Train projector from SuperPos prompt (1.5B) to target model (7B) — LM Head version.

Key differences from the old train_projector.py:
  1. NO nn.Linear classification head — uses target model's native LM Head for
     "positive"/"negative" token scoring (same principle as the 0.8440 result)
  2. Source prompt embedding is registered as a buffer in the pipeline —
     computation graph is intact, projector gets proper gradients
  3. Projector only learns the embedding-space mapping; the LM Head handles
     classification, which is pre-trained and near-perfect

Supports DDP multi-GPU via torchrun.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from qwen_cross_model.config import SUPERPOS_TEMPERATURE, OUTPUT_DIR, PROMPT_LEN
from qwen_cross_model.utils import load_dataset_by_name, build_dataloader, format_time
from qwen_cross_model.align import get_embedding_matrix


def _is_ddp():
    return dist.is_available() and dist.is_initialized()


def _barrier():
    if _is_ddp():
        dist.barrier()


# ---------------------------------------------------------------------------
# Projector (same architecture as PromptProjector)
# ---------------------------------------------------------------------------

class Projector(nn.Module):
    """Map prompt from source to target embedding space.

    Architecture: flatten → Linear(768) → LeakyReLU → Linear(target_dim) → reshape
    """

    def __init__(self, prompt_len, src_dim, tgt_dim):
        super().__init__()
        in_dim = prompt_len * src_dim
        out_dim = prompt_len * tgt_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 768),
            nn.LeakyReLU(),
            nn.Linear(768, out_dim),
        )
        self.prompt_len = prompt_len
        self.tgt_dim = tgt_dim

    def forward(self, src_prompt_emb):
        flat = src_prompt_emb.flatten()
        out = self.proj(flat)
        return out.view(self.prompt_len, self.tgt_dim)


# ---------------------------------------------------------------------------
# Pipeline: projector + frozen target backbone + LM Head scoring
# ---------------------------------------------------------------------------

class ProjectorLMPipeline(nn.Module):
    """Bundles projector + frozen target backbone for DDP training.

    The projector maps src_prompt_emb [L, d_src] → [L, d_tgt].
    The projected prompt is prepended to text input and fed through the frozen
    target backbone. Classification uses the backbone's native LM Head to score
    "positive" vs "negative" token logits — NO randomly initialized nn.Linear head.
    """

    def __init__(self, projector, tgt_backbone, src_prompt_emb,
                 pos_ids, neg_ids, prompt_len):
        super().__init__()
        self.projector = projector
        self.backbone = tgt_backbone
        self.backbone_dtype = next(tgt_backbone.parameters()).dtype
        self.prompt_len = prompt_len

        # src_prompt_emb as buffer — part of module state, DDP handles device placement.
        # self.projector(self.src_prompt_emb) creates a computation graph through
        # the projector's trainable parameters → gradients flow correctly.
        self.register_buffer("src_prompt_emb", src_prompt_emb.clone())

        # Token IDs for LM Head scoring
        self.register_buffer("pos_ids", torch.tensor(sorted(pos_ids), dtype=torch.long))
        self.register_buffer("neg_ids", torch.tensor(sorted(neg_ids), dtype=torch.long))

    def forward(self, input_ids, attention_mask):
        B = input_ids.size(0)

        # 1. Project source prompt → target space (gradients flow through projector)
        projected = self.projector(self.src_prompt_emb)                     # [L, d_tgt] fp32
        projected = projected.to(dtype=self.backbone_dtype)                  # fp16

        # 2. Word embeddings
        word_embeds = self.backbone.model.embed_tokens(input_ids)            # [B, N, d_tgt]

        # 3. [prompt | text]
        prompt_batch = projected.unsqueeze(0).expand(B, -1, -1)             # [B, L, d_tgt]
        inputs_embeds = torch.cat([prompt_batch, word_embeds], dim=1)        # [B, L+N, d_tgt]

        # 4. Attention mask
        prompt_mask = torch.ones(B, self.prompt_len,
                                 dtype=attention_mask.dtype, device=attention_mask.device)
        extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # 5. Frozen backbone forward
        outputs = self.backbone(inputs_embeds=inputs_embeds, attention_mask=extended_mask)
        last_hidden = outputs.hidden_states[-1][:, -1, :]                    # [B, d_tgt]

        # 6. LM Head scoring: "positive" vs "negative"
        lm_logits = self.backbone.lm_head(last_hidden)                      # [B, vocab_size]
        pos_score = lm_logits[:, self.pos_ids].max(dim=-1).values            # [B]
        neg_score = lm_logits[:, self.neg_ids].max(dim=-1).values            # [B]

        # Return as [B, 2] for CrossEntropyLoss (neg, pos order matches SST-2: 0=neg, 1=pos)
        logits = torch.stack([neg_score, pos_score], dim=1)                  # [B, 2]
        return logits


# ---------------------------------------------------------------------------
# LM Head evaluation function (replaces evaluate_model for this pipeline)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_pipeline(model, dataloader, device, desc="Eval"):
    model.eval()
    correct = 0
    total = 0
    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Projector (SuperPos → 7B, LM Head)")
    parser.add_argument("--src-prompt", default="outputs_qwen/superpos/prompt_superpos_T0.5.pt")
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
    parser.add_argument("--weight-mode", default="softmax", choices=["softmax", "relu"],
                        help="How to activate prompt_weights from checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--pos-word", default="positive")
    parser.add_argument("--neg-word", default="negative")
    args = parser.parse_args()

    # ---- DDP setup ----
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        is_main = (local_rank == 0)
    else:
        device = torch.device(args.device) if args.device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    world_size = dist.get_world_size() if _is_ddp() else 1

    if is_main:
        print(f"\n{'='*60}")
        print(f"Projector Training — SuperPos → Target (LM Head)")
        print(f"Source prompt:  {args.src_prompt}")
        print(f"Source model:   {args.src_model}")
        print(f"Target model:   {args.tgt_model}")
        print(f"Dataset:        {args.dataset}")
        print(f"GPUs: {world_size} (DDP)  |  per-GPU batch: {args.batch_size}")
        print(f"Epochs: {args.epochs}  |  LR: {args.lr}")
        print(f"Weight mode:    {args.weight_mode}")
        print(f"Classification: LM Head ('{args.pos_word}' vs '{args.neg_word}')")
        print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    proj_save_path = os.path.join(args.output_dir, "projector_superpos_lmhead.pt")

    # =========================================================================
    # 1. Load SuperPos checkpoint → compute src_prompt_emb
    # =========================================================================
    if is_main:
        print("--- [1/5] Loading SuperPos checkpoint ---")
    ckpt = torch.load(args.src_prompt, map_location="cpu")
    sampled_ids = ckpt["sampled_ids"]
    logits = ckpt["prompt_weights"].float()

    if args.weight_mode == "softmax":
        weights = F.softmax(logits / SUPERPOS_TEMPERATURE, dim=-1)
    else:
        weights = F.relu(logits)

    src_emb_full, src_hidden = get_embedding_matrix(args.src_model)
    src_emb_sampled = src_emb_full[sampled_ids].float()
    src_prompt_emb = weights @ src_emb_sampled                       # [L, d_src]

    if is_main:
        print(f"  Weights:      {weights.shape}")
        print(f"  src_prompt:   shape={src_prompt_emb.shape}  "
              f"norm={src_prompt_emb.norm(p=2, dim=-1).mean():.4f}")

    # =========================================================================
    # 2. Build frozen target backbone (raw, no PromptQwenWrapper)
    # =========================================================================
    if is_main:
        print("--- [2/5] Loading target backbone (frozen) ---")
    tgt_backbone = AutoModelForCausalLM.from_pretrained(
        args.tgt_model, torch_dtype=torch.float16
    )
    tgt_backbone.config.output_hidden_states = True
    tgt_backbone.gradient_checkpointing_enable()
    for p in tgt_backbone.parameters():
        p.requires_grad = False
    tgt_backbone.to(device)
    tgt_hidden = tgt_backbone.config.hidden_size

    # =========================================================================
    # 3. Identify pos/neg token IDs for LM Head scoring
    # =========================================================================
    if is_main:
        print("--- [3/5] Finding pos/neg token IDs ---")
    tokenizer = AutoTokenizer.from_pretrained(args.tgt_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    pos_ids = set()
    neg_ids = set()
    for prefix in ["", " "]:
        pos_ids.update(tokenizer.encode(prefix + args.pos_word, add_special_tokens=False))
        neg_ids.update(tokenizer.encode(prefix + args.neg_word, add_special_tokens=False))
    if is_main:
        print(f"  pos_ids: {sorted(pos_ids)}  → {[tokenizer.decode([i]) for i in sorted(pos_ids)]}")
        print(f"  neg_ids: {sorted(neg_ids)}  → {[tokenizer.decode([i]) for i in sorted(neg_ids)]}")

    # =========================================================================
    # 4. Build projector + pipeline
    # =========================================================================
    if is_main:
        print("--- [4/5] Building projector + pipeline ---")
    projector = Projector(args.prompt_len, src_hidden, tgt_hidden)
    pipeline = ProjectorLMPipeline(
        projector, tgt_backbone, src_prompt_emb,
        pos_ids, neg_ids, args.prompt_len,
    ).to(device)

    if _is_ddp():
        train_model = DDP(pipeline, device_ids=[local_rank], find_unused_parameters=True)
    else:
        train_model = pipeline

    n_params = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    if is_main:
        print(f"  Projector params: {n_params:,}")

    # =========================================================================
    # 5. Data
    # =========================================================================
    train_ds, eval_ds = load_dataset_by_name(args.dataset, tokenizer)
    if _is_ddp():
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
        train_loader = build_dataloader(train_ds, args.batch_size, shuffle=False, sampler=train_sampler)
    else:
        train_loader = build_dataloader(train_ds, args.batch_size, shuffle=True)
    eval_loader = build_dataloader(eval_ds, args.batch_size * 2, shuffle=False)
    train_eval_subset = torch.utils.data.Subset(train_ds, range(min(1000, len(train_ds))))
    train_eval_loader = build_dataloader(train_eval_subset, args.batch_size * 2, shuffle=False)

    # =========================================================================
    # 6. Optimizer
    # =========================================================================
    optimizer = AdamW(train_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps) if warmup_steps > 0 else None

    # =========================================================================
    # 7. Training
    # =========================================================================
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0
    total_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_model.train()
        total_loss = 0.0

        if _is_ddp():
            train_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    leave=False, disable=not is_main)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = train_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()

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

        # Evaluate
        eval_model = pipeline
        train_acc = evaluate_pipeline(eval_model, train_eval_loader, device, desc="Eval train")
        eval_acc = evaluate_pipeline(eval_model, eval_loader, device, desc="Eval val")

        if is_main:
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start
            best_marker = ""
            if eval_acc > best_acc:
                best_acc = eval_acc
                torch.save(projector.state_dict(), proj_save_path)
                best_marker = "  [*BEST]"

            print(f"Epoch {epoch+1:2d}/{args.epochs}  |  "
                  f"loss={avg_loss:.4f}  |  train_acc={train_acc:.4f}  |  "
                  f"eval_acc={eval_acc:.4f}  |  lr={current_lr:.2e}  |  "
                  f"time={format_time(epoch_time)}{best_marker}")

        _barrier()

    if is_main:
        total_time = time.time() - total_start
        print(f"\nTotal time: {format_time(total_time)}  |  Best eval_acc: {best_acc:.4f}")
        print(f"Projector: {proj_save_path}\n")


if __name__ == "__main__":
    main()
