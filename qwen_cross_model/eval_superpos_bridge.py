#!/usr/bin/env python3
"""SuperPos + Discrete Bridge — LM Head scoring (no classification_head required).

  --bridge-mode direct:  top-K weights → target embedding lookup
  --bridge-mode soft:    compute prompt embeddings → soft_bridge_prompt (full vocab search
                         + sink mask + dynamic temp + local norm + direct ID)

Unlike the old version, this uses the target model's native LM Head to score
"positive"/"negative" tokens — no trained classification head needed.
"""

import os
import sys
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .config import DEFAULT_SEED
from .utils import (
    load_dataset_by_name,
    build_dataloader,
    activate_superpos_weights,
    assert_sampled_ids_aligned,
    get_verbalizer_ids,
    set_seed,
)

SRC_PROMPT = "outputs_qwen/superpos/prompt_superpos_Qwen_Qwen2.5-1.5B_sst2.pt"
TGT_MODEL = "Qwen/Qwen2.5-7B"
SRC_MODEL = "Qwen/Qwen2.5-1.5B"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--src-prompt", default=SRC_PROMPT)
parser.add_argument("--src-model", default=SRC_MODEL)
parser.add_argument("--tgt-model", default=TGT_MODEL)
parser.add_argument("--bridge-mode", default="soft", choices=["direct", "soft"],
                    help="direct=pure target lookup; soft=basis-constrained bridge")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--bridge-topk", type=int, default=5,
                    help="TopK for soft bridge vocab search")
parser.add_argument("--bridge-temp", type=float, default=0.05)
parser.add_argument("--direct-temp", type=float, default=1.0,
                    help="Temperature for direct bridge weight scaling (lower=sharper)")
parser.add_argument("--direct-norm", action="store_true", default=False,
                    help="Deprecated compatibility flag; direct bridge is always pure")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--pos-word", default="positive")
parser.add_argument("--neg-word", default="negative")
parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
args = parser.parse_args()
set_seed(args.seed)

print(f"SuperPos + Bridge (LM Head)  bridge_mode={args.bridge_mode}")
print(f"Source prompt: {args.src_prompt}")
print(f"Target model:  {args.tgt_model}")

# =========================================================================
# 1. Load SuperPos checkpoint
# =========================================================================
print("\n[1/5] Loading SuperPos checkpoint...")
ckpt = torch.load(args.src_prompt, map_location="cpu")
sampled_ids = ckpt["sampled_ids"]          # [128]
# Convert raw logits with the temperature recorded at training time.
weights, temperature = activate_superpos_weights(ckpt)
sampled_ids = sampled_ids.long()
print(f"  Weights: {weights.shape}")
print(f"  Checkpoint temperature: {temperature}")

# =========================================================================
# 2. Compute source prompt embeddings (lightweight — embedding only)
# =========================================================================
print("[2/5] Computing source prompt embeddings...")
from .align import get_embedding_matrix
src_emb_full, _ = get_embedding_matrix(args.src_model)
src_emb_sampled = src_emb_full[sampled_ids].float()  # [128, 1536]
src_prompt_emb = weights @ src_emb_sampled            # [100, 1536]

src_emb_sampled_norm = F.normalize(src_emb_sampled, dim=1)
prompt_norm = F.normalize(src_prompt_emb, dim=1)
nn_sim = (prompt_norm @ src_emb_sampled_norm.T).max(dim=1).values  # [100]
print(f"  Prompt shape: {src_prompt_emb.shape}")
print(f"  Nearest-neighbor cos-sim (mean/min/max): "
      f"{nn_sim.mean().item():.4f} / {nn_sim.min().item():.4f} / {nn_sim.max().item():.4f}")

# =========================================================================
# 3. Load target model + set up LM Head scoring
# =========================================================================
print("[3/5] Loading target model...")
tgt_model = AutoModelForCausalLM.from_pretrained(
    args.tgt_model, torch_dtype=torch.float16, device_map=args.device
)
tgt_model.eval()

tokenizer = AutoTokenizer.from_pretrained(args.tgt_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
src_tokenizer = AutoTokenizer.from_pretrained(args.src_model)
assert_sampled_ids_aligned(sampled_ids, src_tokenizer, tokenizer)

pos_ids = get_verbalizer_ids(tokenizer, args.pos_word)
neg_ids = get_verbalizer_ids(tokenizer, args.neg_word)
print(f"  Positive tokens: {pos_ids}  -> {[tokenizer.decode([i]) for i in pos_ids]}")
print(f"  Negative tokens: {neg_ids}  -> {[tokenizer.decode([i]) for i in neg_ids]}")


def score_pos_neg(logits):
    pos_score = logits[:, pos_ids].max(dim=1).values
    neg_score = logits[:, neg_ids].max(dim=1).values
    return pos_score - neg_score


# =========================================================================
# 4. Evaluation loop (shared)
# =========================================================================
_, eval_ds = load_dataset_by_name("sst2", tokenizer)
loader = build_dataloader(eval_ds, args.batch_size, shuffle=False)


def evaluate_with_lm_head(projected_prompt, desc="Eval"):
    """Run LM Head evaluation given [L, d_tgt] projected prompt."""
    correct = 0
    total = 0
    prompt_emb = projected_prompt.to(args.device).to(dtype=torch.float16)
    L = prompt_emb.size(0)

    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["label"].to(args.device)

        B = input_ids.size(0)
        word_embeds = tgt_model.model.embed_tokens(input_ids)
        pb = prompt_emb.unsqueeze(0).expand(B, -1, -1)
        inputs_embeds = torch.cat([pb, word_embeds], dim=1)

        prompt_mask = torch.ones(B, L, dtype=attention_mask.dtype, device=args.device)
        extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        with torch.no_grad():
            outputs = tgt_model(inputs_embeds=inputs_embeds, attention_mask=extended_mask)
            last_logits = outputs.logits[:, -1, :].float()
            diff = score_pos_neg(last_logits)
            preds = (diff > 0).long()
            correct += (preds == labels).sum().item()
            total += B

    return correct / total if total > 0 else 0.0


# =====================================================================
# 5a. Mode: Direct — top-K weight lookup
# =====================================================================
if args.bridge_mode == "direct":
    tgt_emb = tgt_model.model.embed_tokens.weight[sampled_ids].float().cpu()  # [128, d_tgt]

    print("[5/5] Direct bridge K sweep...")
    k_list = [1, 3, 5, 6, 7, 8, 10, 20, 50, 128]
    for k in k_list:
        if k > weights.size(1):
            continue

        # Top-K weights (renormalized to sum=1)
        topk_vals, topk_idx = weights.topk(k, dim=1)            # [100, K]
        topk_w = topk_vals / topk_vals.sum(dim=1, keepdim=True)  # [100, K]

        projected = torch.zeros(100, tgt_emb.size(1))
        for i in range(100):
            pos_vec = (topk_w[i].unsqueeze(1) * tgt_emb[topk_idx[i]]).sum(dim=0)
            projected[i] = pos_vec

        acc = evaluate_with_lm_head(projected, desc=f"Direct K={k}")
        proj_norm = projected.norm(p=2, dim=-1).mean().item()
        print(f"SuperPos-direct K={k}: {acc:.4f}  (proj_norm={proj_norm:.4f})")


# =====================================================================
# 5b. Mode: Soft bridge — SuperPos embeddings -> full vocab search -> bridge
# =====================================================================
else:
    from .discrete_bridge import soft_bridge_prompt

    k_list = [1, 3, 5, 6, 7, 8, 10, 20]
    for k in k_list:
        print(f"\n[5/5] Soft bridge K={k}...")
        projected, diag = soft_bridge_prompt(
            src_prompt_emb, args.src_model, args.tgt_model,
            topk=k,
            temperature=args.bridge_temp,
            dynamic_temp=True,
            local_norm=True,
            use_direct_id=True,
            mask_sink_tokens=True,
            sampled_ids=sampled_ids,
            prompt_weights=weights,
            weight_mode="softmax",
            weight_temperature=temperature,
        )

        acc = evaluate_with_lm_head(projected, desc=f"Soft K={k}")
        print(f"SuperPos-soft-bridge K={k:2d}: acc={acc:.4f}  "
              f"(top1_sim={diag['mean_top1_sim']:.4f}, norm={diag['projected_norm']:.4f})")
