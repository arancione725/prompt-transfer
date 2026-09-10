#!/usr/bin/env python3
"""SuperPos + Soft Bridge + LM Head — weight-decomposed bridge within basis tokens.

Key insight: the mixed prompt vector is a weighted sum of m=128 basis tokens.
Searching the full 150K vocab for nearest neighbors is wrong — it finds unrelated
tokens and discards the learned weights. The correct approach:

  1. Constrain NN search to the 128 basis tokens only (not full vocab)
  2. Use original prompt_weights as primary weights (no re-computed softmax)
  3. Cosine similarity acts only as a per-position quality gate
  4. Decompose: top-K basis tokens by weight → bridge individually → recombine
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
from qwen_cross_model.align import get_embedding_matrix
from qwen_cross_model.discrete_bridge import soft_bridge_prompt


def main():
    parser = argparse.ArgumentParser(description="SuperPos + Soft Bridge + LM Head eval")
    parser.add_argument("--src-prompt", default="outputs_qwen/superpos/prompt_superpos_T0.5.pt")
    parser.add_argument("--src-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--tgt-model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--dataset", default="sst2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bridge-topk", type=int, default=8,
                        help="Top-K basis tokens per position (selected by original weight)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pos-word", default="positive", help="Word used for positive class")
    parser.add_argument("--neg-word", default="negative", help="Word used for negative class")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-local-norm", action="store_true",
                        help="Disable local norm calibration")
    parser.add_argument("--no-direct-id", action="store_true",
                        help="Disable direct ID mapping (use decode→encode)")
    args = parser.parse_args()
    set_seed(args.seed)

    print(f"SuperPos + Soft Bridge + LM Head Evaluation")
    print(f"Source prompt:   {args.src_prompt}")
    print(f"Source model:    {args.src_model}")
    print(f"Target model:    {args.tgt_model}")
    print(f"Bridge TopK:     {args.bridge_topk}")
    print(f"Local norm:      {not args.no_local_norm}")
    print(f"Direct ID:       {not args.no_direct_id}")
    print(f"Pos/Neg words:   '{args.pos_word}' / '{args.neg_word}'")

    # --- 1. Load SuperPos checkpoint ---
    print("\n[1/5] Loading SuperPos checkpoint...")
    ckpt = torch.load(args.src_prompt, map_location="cpu")
    sampled_ids = ckpt["sampled_ids"]              # [128]
    prob, temperature = activate_superpos_weights(ckpt)
    sampled_ids = sampled_ids.long()
    if not 1 <= args.bridge_topk <= prob.size(1):
        raise ValueError(f"--bridge-topk must be in [1, {prob.size(1)}], got {args.bridge_topk}")
    print(f"  Checkpoint temperature: {temperature}")

    if not args.no_direct_id:
        src_tokenizer = AutoTokenizer.from_pretrained(args.src_model)
        tgt_tokenizer = AutoTokenizer.from_pretrained(args.tgt_model)
        assert_sampled_ids_aligned(sampled_ids, src_tokenizer, tgt_tokenizer)

    # Weight sparsity stats
    total_w = prob.numel()
    near_zero = (prob < 0.001).sum().item()
    print(f"  Weights: {prob.shape}, near-zero (<0.001): {near_zero}/{total_w} ({100*near_zero/total_w:.1f}%)")

    # --- 2. Compute source prompt embeddings ---
    print("[2/5] Computing source prompt embeddings...")
    src_emb_full, src_hidden = get_embedding_matrix(args.src_model)
    src_emb_sampled = src_emb_full[sampled_ids].float()  # [128, 1536]
    src_prompt_emb = prob @ src_emb_sampled               # [100, 1536]
    print(f"  Source prompt shape: {src_prompt_emb.shape}")

    # --- 3. Soft bridge: constrained to basis tokens, original weights ---
    print("[3/5] Soft bridge (basis-constrained, weight-decomposed)...")
    projected, diag = soft_bridge_prompt(
        src_prompt_emb, args.src_model, args.tgt_model,
        topk=args.bridge_topk,
        local_norm=not args.no_local_norm,
        use_direct_id=not args.no_direct_id,
        sampled_ids=sampled_ids,
        prompt_weights=prob,
        weight_mode="softmax",
        weight_temperature=temperature,
    )

    print(f"  Mode:             {diag['mode']}")
    print(f"  Projected shape:  {projected.shape}")
    print(f"  Gate (mean/min/max): {diag['gate_mean']:.4f} / {diag['gate_min']:.4f} / {diag['gate_max']:.4f}")
    print(f"  Top-1 cos-sim:    {diag['mean_top1_sim']:.4f}")
    print(f"  Projected norm:   {diag['projected_norm']:.4f}")
    print(f"  Sample (first 5 positions):")
    for t in diag["sample_texts"]:
        print(f"    {t}")

    # --- 4. Load target model + eval ---
    print("[4/5] Loading target model...")
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.tgt_model, torch_dtype=torch.float16, device_map=args.device
    )
    tgt_model.eval()

    print("[5/5] Evaluating...")
    tokenizer = AutoTokenizer.from_pretrained(args.tgt_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    pos_ids = get_verbalizer_ids(tokenizer, args.pos_word)
    neg_ids = get_verbalizer_ids(tokenizer, args.neg_word)
    print(f"  Positive tokens: {pos_ids}  → {[tokenizer.decode([i]) for i in pos_ids]}")
    print(f"  Negative tokens: {neg_ids}  → {[tokenizer.decode([i]) for i in neg_ids]}")

    def score_pos_neg(logits):
        pos_score = logits[:, pos_ids].max(dim=1).values
        neg_score = logits[:, neg_ids].max(dim=1).values
        return pos_score - neg_score

    _, eval_ds = load_dataset_by_name(args.dataset, tokenizer)
    loader = build_dataloader(eval_ds, args.batch_size, shuffle=False)

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
    print(f"\nSuperPos + Soft Bridge + LM Head (K={args.bridge_topk}): {acc:.4f}")
    return acc


if __name__ == "__main__":
    main()
