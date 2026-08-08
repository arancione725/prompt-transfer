#!/usr/bin/env python3
"""SuperPos weight diagnostics — diagnose per-position sparsity & diversity.

Usage:
    python diagnose_weights.py
    python diagnose_weights.py --ckpt path/to/checkpoint.pt --temp 0.5
    python diagnose_weights.py --ckpt outputs_qwen/superpos/old/prompt_superpos_...pt  # old code
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="SuperPos weight diagnostics")
    parser.add_argument("--ckpt", default="outputs_qwen/prompt_superpos_Qwen_Qwen2.5-1.5B_sst2.pt")
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    logits = ckpt["prompt_weights"].float()  # [L, m]
    ids = ckpt["sampled_ids"]                 # [m]
    L, m = logits.shape

    prob = F.softmax(logits / args.temp, dim=-1)

    tok = AutoTokenizer.from_pretrained(args.model)

    # --- Per-position stats ---
    top1_val, top1_idx = prob.topk(1, dim=-1)
    top1_val = top1_val.squeeze(-1)
    top1_idx = top1_idx.squeeze(-1)

    top3_val, top3_idx = prob.topk(3, dim=-1)
    top3_sum = top3_val.sum(-1)

    entropy = -(prob * prob.log().clamp_min(1e-12)).sum(-1)

    print(f"Checkpoint: {args.ckpt}")
    print(f"Temperature: {args.temp}")
    print(f"Shape: [{L}, {m}]  (prompt_len={L}, basis={m})")
    print()

    # Top-N entropy by position
    print(f"{'pos':>4s}  {'entropy':>7s}  {'top1':>7s}  {'top3_sum':>9s}  top-3 tokens")
    print("-" * 85)
    sample_positions = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    for i in sample_positions:
        if i >= L:
            continue
        t3 = [tok.decode([ids[t].item()]) for t in top3_idx[i]]
        print(f"{i:4d}  {entropy[i]:7.4f}  {top1_val[i].item():7.4f}  {top3_sum[i].item():9.4f}  {t3}")

    # --- Aggregate ---
    print(f"\n{'='*60}")
    print(f"Aggregate")
    print(f"{'='*60}")
    print(f"Entropy:       mean={entropy.mean():.4f}  min={entropy.min():.4f}  max={entropy.max():.4f}")
    print(f"Top-1 ratio:   mean={top1_val.float().mean():.4f}  min={top1_val.float().min():.4f}  max={top1_val.float().max():.4f}")
    print(f"Top-3 sum:     mean={top3_sum.mean():.4f}  min={top3_sum.min():.4f}  max={top3_sum.max():.4f}")

    n_unique = len(set(ids[top1_idx].tolist()))
    print(f"Unique top-1 tokens: {n_unique} / {m} basis tokens used")

    dead = (top1_val > 0.99).sum().item()
    dense = (top1_val < 0.5).sum().item()
    print(f"Dead  positions (top1>0.99): {dead}/{L}")
    print(f"Dense positions (top1<0.5):  {dense}/{L}")

    # --- Verdict ---
    print(f"\n{'='*60}")
    print(f"Verdict")
    print(f"{'='*60}")
    if entropy.mean() < 0.5:
        print(f"[!] T={args.temp} too SMALL — weights collapsed to one-hot")
        print(f"    > Try T={args.temp * 3:.1f}~{args.temp * 5:.1f}")
    elif entropy.mean() > 4.5:
        print(f"[!] T={args.temp} too LARGE — weights nearly uniform")
        print(f"    > Try T={args.temp * 0.3:.1f}~{args.temp * 0.5:.1f}")
    elif 1.5 <= entropy.mean() <= 3.5 and dead < 10 and dense < 80:
        print(f"[*] Weights look HEALTHY at T={args.temp}")
        print(f"    > Proceed to bridge eval")
    else:
        print(f"[~] Weights at T={args.temp} are borderline")
        top1_avg = top1_val.float().mean().item()
        if top1_avg > 0.7:
            print(f"    > Slightly too sparse (top1 mean={top1_avg:.3f}), consider raising T")
        elif top1_avg < 0.1:
            print(f"    > Slightly too uniform (top1 mean={top1_avg:.3f}), consider lowering T")


if __name__ == "__main__":
    main()
