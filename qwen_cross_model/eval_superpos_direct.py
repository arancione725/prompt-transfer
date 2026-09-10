#!/usr/bin/env python3
"""SuperPos + LM Head — direct bridge (same token IDs, target embedding lookup).

Loads the SuperPos checkpoint, applies weights to target model's embedding table
at the SAME sampled_ids, prepends prompt to input, and uses the 7B's native
LM Head to score "positive" vs "negative" token completions.

This is the direct mode: no vocab search, no decode→encode round-trip.
"""

import os
import sys
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from qwen_cross_model.config import DEFAULT_SEED
from qwen_cross_model.utils import (
    load_dataset_by_name,
    build_dataloader,
    activate_superpos_weights,
    assert_sampled_ids_aligned,
    get_verbalizer_ids,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(description="SuperPos + LM Head eval (direct bridge)")
    parser.add_argument("--src-prompt", default="outputs_qwen/superpos/prompt_superpos_T0.5.pt")
    parser.add_argument("--src-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--tgt-model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--dataset", default="sst2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bridge-k", type=int, default=128,
                        help="Top-K weights for direct bridge (default 128 = all basis tokens)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pos-word", default="positive", help="Word used for positive class")
    parser.add_argument("--neg-word", default="negative", help="Word used for negative class")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"SuperPos + LM Head Evaluation (direct bridge)")
    print(f"Source prompt: {args.src_prompt}")
    print(f"Target model:  {args.tgt_model}")
    print(f"Bridge K:      {args.bridge_k}")
    print(f"Pos/Neg words: '{args.pos_word}' / '{args.neg_word}'")

    # --- 1. Load SuperPos checkpoint ---
    print("\n[1/5] Loading SuperPos checkpoint...")
    ckpt = torch.load(args.src_prompt, map_location="cpu")
    sampled_ids = ckpt["sampled_ids"]            # [128]
    prob, temperature = activate_superpos_weights(ckpt)
    sampled_ids = sampled_ids.long()
    if not 1 <= args.bridge_k <= prob.size(1):
        raise ValueError(f"--bridge-k must be in [1, {prob.size(1)}], got {args.bridge_k}")
    print(f"  Checkpoint temperature: {temperature}")

    # --- 2. Direct bridge: same sampled_ids → target embedding lookup ---
    print("[2/5] Direct bridge: same token IDs, target embedding lookup...")
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.tgt_model, torch_dtype=torch.float16, device_map=args.device
    )
    tgt_model.eval()
    src_tokenizer = AutoTokenizer.from_pretrained(args.src_model)
    tgt_emb = tgt_model.model.embed_tokens.weight[sampled_ids].float().cpu()  # [128, 3584]

    if args.bridge_k < prob.size(1):
        topk_vals, topk_idx = prob.topk(args.bridge_k, dim=1)
        topk_w = topk_vals / topk_vals.sum(dim=1, keepdim=True)
        projected = torch.zeros(prob.size(0), tgt_emb.size(1))
        for i in range(prob.size(0)):
            projected[i] = (topk_w[i].unsqueeze(1) * tgt_emb[topk_idx[i]]).sum(dim=0)
    else:
        projected = prob @ tgt_emb  # [100, 3584]

    print(f"  Projected prompt: {projected.shape}")
    print(f"  Prompt norm:      {projected.norm(p=2, dim=-1).mean():.4f}")

    # --- 3. Identify pos/neg token IDs ---
    print("[3/5] Finding pos/neg token IDs...")
    tokenizer = AutoTokenizer.from_pretrained(args.tgt_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    assert_sampled_ids_aligned(sampled_ids, src_tokenizer, tokenizer)

    pos_ids = get_verbalizer_ids(tokenizer, args.pos_word)
    neg_ids = get_verbalizer_ids(tokenizer, args.neg_word)
    print(f"  Positive tokens: {pos_ids}  → {[tokenizer.decode([i]) for i in pos_ids]}")
    print(f"  Negative tokens: {neg_ids}  → {[tokenizer.decode([i]) for i in neg_ids]}")

    # --- 4. Scoring function ---
    def score_pos_neg(logits):
        pos_score = logits[:, pos_ids].max(dim=1).values
        neg_score = logits[:, neg_ids].max(dim=1).values
        return pos_score - neg_score

    # --- 5. Load data ---
    print("[4/5] Loading dataset...")
    _, eval_ds = load_dataset_by_name(args.dataset, tokenizer)
    loader = build_dataloader(eval_ds, args.batch_size, shuffle=False)

    # --- 6. Evaluate ---
    print("[5/5] Evaluating...")
    correct = 0
    total = 0

    prompt_emb = projected.to(args.device).to(dtype=torch.float16)

    for batch in tqdm(loader, desc="LM Head eval"):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["label"].to(args.device)

        B = input_ids.size(0)
        word_embeds = tgt_model.model.embed_tokens(input_ids)
        pb = prompt_emb.unsqueeze(0).expand(B, -1, -1)
        inputs_embeds = torch.cat([pb, word_embeds], dim=1)

        prompt_mask = torch.ones(B, projected.size(0), dtype=attention_mask.dtype, device=args.device)
        extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        with torch.no_grad():
            outputs = tgt_model(inputs_embeds=inputs_embeds, attention_mask=extended_mask)
            next_logits = outputs.logits[:, -1, :].float()
            diff = score_pos_neg(next_logits)
            preds = (diff > 0).long()
            correct += (preds == labels).sum().item()
            total += B

    acc = correct / total
    print(f"\nSuperPos + Direct Bridge + LM Head (K={args.bridge_k}): {acc:.4f}")
    return acc


if __name__ == "__main__":
    main()
