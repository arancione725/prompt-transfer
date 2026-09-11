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


DEFAULT_K_VALUES = [1, 3, 5, 6, 7, 8, 10, 20, 50, 128]


def project_prompt(prob, target_embeddings, bridge_k):
    """Project SuperPos probabilities onto target embeddings using top-K."""
    if bridge_k == prob.size(1):
        return prob @ target_embeddings

    topk_values, topk_indices = prob.topk(bridge_k, dim=1)
    topk_weights = topk_values / topk_values.sum(dim=1, keepdim=True).clamp_min(1e-12)
    selected_embeddings = target_embeddings[topk_indices]
    return (topk_weights.unsqueeze(-1) * selected_embeddings).sum(dim=1)


def main():
    parser = argparse.ArgumentParser(description="SuperPos + LM Head eval (direct bridge)")
    parser.add_argument(
        "--src-prompt",
        default="outputs_qwen/superpos_prompt_transfer_main_clean/prompt_transfer_main_clean.pt",
    )
    parser.add_argument("--src-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--tgt-model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--dataset", default="sst2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--bridge-k",
        type=int,
        nargs="+",
        default=DEFAULT_K_VALUES,
        help="K values to evaluate in one run (default: 1 3 5 6 7 8 10 20 50 128)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pos-word", default="positive", help="Word used for positive class")
    parser.add_argument("--neg-word", default="negative", help="Word used for negative class")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"SuperPos + LM Head Evaluation (direct bridge)")
    print(f"Source prompt: {args.src_prompt}")
    print(f"Target model:  {args.tgt_model}")
    print(f"Bridge K sweep: {args.bridge_k}")
    print(f"Pos/Neg words: '{args.pos_word}' / '{args.neg_word}'")

    # --- 1. Load SuperPos checkpoint ---
    print("\n[1/5] Loading SuperPos checkpoint...")
    ckpt = torch.load(args.src_prompt, map_location="cpu")
    sampled_ids = ckpt["sampled_ids"]            # [128]
    prob, temperature = activate_superpos_weights(ckpt)
    sampled_ids = sampled_ids.long()
    print(f"  Checkpoint temperature: {temperature}")

    # 2. Load the target model and construct the direct bridge basis once.
    print("[2/5] Direct bridge: same token IDs, target embedding lookup...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.tgt_model,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    target_model.eval()
    source_tokenizer = AutoTokenizer.from_pretrained(args.src_model)
    target_embeddings = target_model.model.embed_tokens.weight[sampled_ids].float().cpu()

    if not args.bridge_k or any(k < 1 or k > prob.size(1) for k in args.bridge_k):
        raise ValueError(
            f"--bridge-k values must be in [1, {prob.size(1)}], got {args.bridge_k}"
        )

    # 3. Identify positive and negative token IDs.
    print("[3/5] Finding pos/neg token IDs...")
    tokenizer = AutoTokenizer.from_pretrained(args.tgt_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    assert_sampled_ids_aligned(sampled_ids, source_tokenizer, tokenizer)

    pos_ids = get_verbalizer_ids(tokenizer, args.pos_word)
    neg_ids = get_verbalizer_ids(tokenizer, args.neg_word)
    print(f"  Positive tokens: {pos_ids} -> {[tokenizer.decode([i]) for i in pos_ids]}")
    print(f"  Negative tokens: {neg_ids} -> {[tokenizer.decode([i]) for i in neg_ids]}")

    # 4. Load the dataset once and reuse the same examples for every K.
    print("[4/5] Loading dataset...")
    _, eval_dataset = load_dataset_by_name(args.dataset, tokenizer)
    dataloader = build_dataloader(eval_dataset, args.batch_size, shuffle=False)

    def evaluate_projected_prompt(model, projected, bridge_k):
        correct = 0
        total = 0
        prompt_emb = projected.to(device=args.device, dtype=torch.float16)

        def score_pos_neg(logits):
            positive_score = logits[:, pos_ids].max(dim=1).values
            negative_score = logits[:, neg_ids].max(dim=1).values
            return positive_score - negative_score

        for batch in tqdm(dataloader, desc=f"LM Head eval K={bridge_k}"):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["label"].to(args.device)

            batch_size = input_ids.size(0)
            word_embeds = model.model.embed_tokens(input_ids)
            prompt_batch = prompt_emb.unsqueeze(0).expand(batch_size, -1, -1)
            inputs_embeds = torch.cat([prompt_batch, word_embeds], dim=1)

            prompt_mask = torch.ones(
                batch_size,
                projected.size(0),
                dtype=attention_mask.dtype,
                device=args.device,
            )
            extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)

            with torch.no_grad():
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=extended_mask,
                )
                next_logits = outputs.logits[:, -1, :].float()
                diff = score_pos_neg(next_logits)
                predictions = (diff > 0).long()
                correct += (predictions == labels).sum().item()
                total += batch_size

        return correct / total if total else 0.0

    # 5. Evaluate all requested K values without reloading model or data.
    print(f"[5/5] Evaluating K sweep: {args.bridge_k}")
    results = {}

    for bridge_k in args.bridge_k:
        projected = project_prompt(prob, target_embeddings, bridge_k)
        prompt_norm = projected.norm(p=2, dim=-1).mean().item()
        accuracy = evaluate_projected_prompt(target_model, projected, bridge_k)
        results[bridge_k] = accuracy
        print(
            f"SuperPos + Direct Bridge + LM Head "
            f"(K={bridge_k}): {accuracy:.4f} (proj_norm={prompt_norm:.4f})"
        )

    print("\nK sweep summary:")
    for bridge_k in args.bridge_k:
        print(f"  K={bridge_k:>3}: acc={results[bridge_k]:.4f}")

    return results


if __name__ == "__main__":
    main()
