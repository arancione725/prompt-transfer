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
import torch.nn.functional as F
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
# Fast token ID resolution (avoids decode→encode fragmentation)
# ---------------------------------------------------------------------------

def _get_min_shared_vocab(src_model_name, tgt_model_name):
    """Return the smaller vocab size — tokens below this are 1:1 across Qwen2.5 series."""
    src_tok = AutoTokenizer.from_pretrained(src_model_name)
    tgt_tok = AutoTokenizer.from_pretrained(tgt_model_name)
    return min(src_tok.vocab_size, tgt_tok.vocab_size)


def _resolve_target_ids(src_tid, src_tok, tgt_tok, min_shared_vocab, use_direct_id=True):
    """Resolve a source token ID to target token ID(s).

    When use_direct_id=True and src_tid < min_shared_vocab, returns [src_tid]
    directly — the Qwen2.5 series share the same tokenizer, so IDs are 1:1 for
    the shared vocabulary. This avoids the decode→encode round-trip that can
    fragment subword tokens (e.g. 'Ġing' → ' ing' → [' ', 'ing']).

    For tokens outside the shared range, falls back to text-based encode.
    """
    if use_direct_id and src_tid < min_shared_vocab:
        return [src_tid]
    text = src_tok.decode([src_tid])
    return tgt_tok.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------------------
# Attention-sink token masking (prevents EOS collapse in soft bridge)
# ---------------------------------------------------------------------------

def _get_sink_token_ids(tokenizer):
    """Return token IDs that act as attention sinks / structural anchors.

    In LLMs, certain tokens (<|endoftext|>, chat format markers, whitespace, etc.)
    serve as "attention sinks" — their hidden states absorb attention mass without
    carrying semantic content. When soft prompt vectors drift near these tokens
    in embedding space, the bridge picks them as nearest neighbors, producing
    semantically empty prompt text.

    Masking these tokens before top-K forces the bridge to choose tokens that carry
    actual linguistic meaning, even if they are geometrically slightly farther away.
    """
    sink_ids = set()

    # 1. All registered special tokens (eos, bos, pad, etc.)
    if tokenizer.all_special_ids:
        sink_ids.update(tokenizer.all_special_ids)

    # 2. Explicit structural / control tokens
    structural_texts = [
        "<|im_start|>", "<|im_end|>",
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
        "<|video_pad|>", "<|audio_pad|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
        "<|grounding|>",
        "\n", "\t",
    ]
    for text in structural_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        sink_ids.update(ids)

    return torch.tensor(sorted(sink_ids))


# ---------------------------------------------------------------------------


@torch.no_grad()
def soft_bridge_prompt(
    src_prompt_emb, src_model_name, tgt_model_name,
    topk=5, temperature=0.05,
    dynamic_temp=True,
    local_norm=True,
    use_direct_id=True,
    mask_sink_tokens=True,
    sampled_ids=None,
    prompt_weights=None,
    weight_mode="softmax",
    weight_temperature=0.5,
    bridge_resources=None,
):
    """Soft discrete bridge: transfer prompt from source to target embedding space.

    Two modes:
    A) WITH sampled_ids + prompt_weights (SuperPos path — the correct approach):
       1. Constrain NN search to the m basis tokens only (not full 150K vocab)
       2. Use original prompt_weights as primary weights — no re-computed softmax
       3. Cosine similarity acts only as a per-position quality gate
       4. Decompose: top-K basis tokens by weight → bridge individually → recombine

    B) WITHOUT sampled_ids (legacy path — for standard soft prompts):
       Full-vocab NN search → softmax re-weighting → bridge.
       Only used when the prompt is a free vector (not a basis combination).

    Args:
        src_prompt_emb:   [L, d_src] mixed prompt embeddings
        src_model_name:    source model name
        tgt_model_name:    target model name
        topk:              number of candidate tokens per position
        temperature:       base softmax temperature (legacy path only)
        dynamic_temp:      adapt temperature per position (legacy path only)
        local_norm:        rescale each position to its weighted token norm
        use_direct_id:     use direct ID mapping for shared-vocab tokens
        mask_sink_tokens:  mask attention-sink tokens (legacy path only)
        sampled_ids:       [m] basis token IDs — when provided, enables SuperPos path
        prompt_weights:    [L, m] ACTIVATED weights — caller must apply softmax/relu first.
                           Only used when sampled_ids is provided.
        weight_mode:       "softmax" or "relu" — for diagnostics only
        weight_temperature: T value used to activate weights — for diagnostics only

    Returns:
        projected:   [prompt_len, d_tgt] target prompt embeddings
        diagnostics: dict
    """
    from .align import get_embedding_matrix

    if bridge_resources is None:
        src_tok = AutoTokenizer.from_pretrained(src_model_name)
        tgt_tok = AutoTokenizer.from_pretrained(tgt_model_name)
        src_emb_full, _ = get_embedding_matrix(src_model_name)
        tgt_emb_full, tgt_hidden = get_embedding_matrix(tgt_model_name)
    else:
        src_tok = bridge_resources["src_tokenizer"]
        tgt_tok = bridge_resources["tgt_tokenizer"]
        src_emb_full = bridge_resources["src_embeddings"]
        tgt_emb_full = bridge_resources["tgt_embeddings"]
        tgt_hidden = bridge_resources["tgt_hidden"]

    prompt_len = src_prompt_emb.size(0)
    prompt_norm = F.normalize(src_prompt_emb.float(), dim=1)

    # =========================================================================
    # SuperPos path: constrained search within basis tokens, original weights
    # =========================================================================
    if sampled_ids is not None and prompt_weights is not None:
        m = sampled_ids.size(0)

        # --- 1. Constrain search space to the m basis tokens only ---
        src_emb_sampled = src_emb_full[sampled_ids].float()                 # [m, d_src]
        src_emb_sampled_norm = F.normalize(src_emb_sampled, dim=1)           # [m, d_src]
        sim = prompt_norm @ src_emb_sampled_norm.T                          # [L, m]

        # --- 2. Select top-K basis tokens by ORIGINAL weight (not cosine sim) ---
        k = min(topk, m)
        if k < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        topk_vals, topk_idx = prompt_weights.topk(k, dim=1)                 # [L, K]
        # Top-K is a sparse approximation of the original convex mixture.
        # Renormalize the retained mass so changing K does not also change
        # the prompt scale.
        topk_weights = topk_vals / topk_vals.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # --- 3. Per-position cosine-similarity gate ---
        # Gate = max cosine sim between mixed vector and its selected basis tokens.
        # High → geometry is clean, weights transfer faithfully.
        # Low  → mixing drifted, dampen this position to prevent noise injection.
        gate_per_pos = torch.zeros(prompt_len)
        for i in range(prompt_len):
            gate_per_pos[i] = sim[i, topk_idx[i]].max()

        # --- 4. Per-position: bridge top-K basis tokens + recombine with original weights ---
        tgt_emb_sampled = tgt_emb_full[sampled_ids].float()                 # [m, d_tgt]
        min_shared_vocab = (
            bridge_resources["min_shared_vocab"]
            if bridge_resources is not None and use_direct_id
            else (_get_min_shared_vocab(src_model_name, tgt_model_name) if use_direct_id else 0)
        )

        projected = torch.zeros(prompt_len, tgt_hidden)
        diag_texts = []

        for i in range(prompt_len):
            gate = gate_per_pos[i].item()
            pos_vec = torch.zeros(tgt_hidden)
            expected_norm = 0.0
            texts_for_pos = []

            for k_idx in range(k):
                basis_idx = topk_idx[i, k_idx].item()       # index into [0, m-1]
                w = topk_weights[i, k_idx].item()           # renormalized weight
                if w <= 0:
                    continue

                src_tid = sampled_ids[basis_idx].item()
                text = src_tok.decode([src_tid])

                # Bridge this basis token to target space
                tgt_ids = _resolve_target_ids(src_tid, src_tok, tgt_tok, min_shared_vocab, use_direct_id)
                if len(tgt_ids) == 0:
                    continue

                tok_emb = tgt_emb_full[torch.tensor(tgt_ids)].mean(dim=0)  # [d_tgt]
                pos_vec += w * tok_emb

                if local_norm:
                    expected_norm += w * tok_emb.norm(p=2, dim=-1)

                texts_for_pos.append(f"{text}(w={w:.3f})")

            # --- Apply cosine-similarity gate ---
            pos_vec = pos_vec * gate

            # --- Local norm calibration (on gated vector) ---
            if local_norm and expected_norm > 0:
                actual_norm = pos_vec.norm(p=2, dim=-1).clamp_min(1e-8)
                pos_vec = pos_vec / actual_norm * expected_norm * gate

            projected[i] = pos_vec
            diag_texts.append(" | ".join(texts_for_pos[:3]))

        diagnostics = {
            "mode": "superpos",
            "topk": min(topk, m),
            "weight_mode": weight_mode,
            "weight_temperature": weight_temperature,
            "local_norm": local_norm,
            "use_direct_id": use_direct_id,
            "gate_mean": gate_per_pos.mean().item(),
            "gate_min": gate_per_pos.min().item(),
            "gate_max": gate_per_pos.max().item(),
            "mean_top1_sim": sim.max(dim=1).values.mean().item(),
            "projected_norm": projected.norm(p=2, dim=-1).mean().item(),
            "sample_texts": diag_texts[:5],
        }

        return projected, diagnostics

    # =========================================================================
    # Legacy path: full-vocab NN search → softmax re-weighting
    # (for standard soft prompts that are free vectors, not basis combinations)
    # =========================================================================

    src_emb_norm = F.normalize(src_emb_full, dim=1)

    sim = prompt_norm @ src_emb_norm.T            # [L, V_src]

    if mask_sink_tokens:
        sink_ids = _get_sink_token_ids(src_tok)
        sim[:, sink_ids] = -float('inf')

    topk_sim, topk_ids = sim.topk(topk, dim=1)    # [L, K], [L, K]

    if dynamic_temp:
        std_sim = topk_sim.std(dim=1, keepdim=True) + 1e-6
        effective_temp = temperature / (std_sim * 10)
    else:
        effective_temp = temperature

    weights = torch.softmax(topk_sim / effective_temp, dim=1)  # [L, K]

    min_shared_vocab = (
        bridge_resources["min_shared_vocab"]
        if bridge_resources is not None and use_direct_id
        else (_get_min_shared_vocab(src_model_name, tgt_model_name) if use_direct_id else 0)
    )

    projected = torch.zeros(prompt_len, tgt_hidden)
    diag_texts = []

    for i in range(prompt_len):
        pos_vec = torch.zeros(tgt_hidden)
        expected_norm = 0.0
        texts_for_pos = []
        for k in range(topk):
            tid = topk_ids[i, k].item()
            w = weights[i, k].item()
            text = src_tok.decode([tid])

            tgt_ids = _resolve_target_ids(tid, src_tok, tgt_tok, min_shared_vocab, use_direct_id)
            if len(tgt_ids) == 0:
                continue

            tgt_emb = tgt_emb_full[torch.tensor(tgt_ids)].mean(dim=0)
            pos_vec += w * tgt_emb

            if local_norm:
                expected_norm += w * tgt_emb.norm(p=2, dim=-1)

            texts_for_pos.append(f"{text}({w:.3f})")

        if local_norm and expected_norm > 0:
            actual_norm = pos_vec.norm(p=2, dim=-1).clamp_min(1e-8)
            pos_vec = pos_vec / actual_norm * expected_norm

        projected[i] = pos_vec
        diag_texts.append(" | ".join(texts_for_pos[:3]))

    if not local_norm:
        target_norm = tgt_emb_full.norm(p=2, dim=-1).mean()
        prompt_norm_val = projected.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        projected = projected / prompt_norm_val * target_norm

    diagnostics = {
        "mode": "legacy",
        "topk": topk,
        "temperature": temperature,
        "dynamic_temp": dynamic_temp,
        "local_norm": local_norm,
        "use_direct_id": use_direct_id,
        "mean_top1_sim": topk_sim[:, 0].mean().item(),
        "mean_topk_sim": topk_sim.mean().item(),
        "projected_norm": projected.norm(p=2, dim=-1).mean().item(),
        "sample_texts": diag_texts[:5],
    }

    return projected, diagnostics





def main():
    parser = argparse.ArgumentParser(
        description="SuperPos + Soft Bridge + LM Head eval with K sweep"
    )
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
        "--bridge-topk",
        dest="bridge_k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 6, 7, 8, 10, 20, 50, 128],
        help="K values to evaluate in one run (default: 1 3 5 6 7 8 10 20 50 128)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pos-word", default="positive", help="Word used for positive class")
    parser.add_argument("--neg-word", default="negative", help="Word used for negative class")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--no-local-norm",
        action="store_true",
        help="Disable local norm calibration",
    )
    parser.add_argument(
        "--no-direct-id",
        action="store_true",
        help="Disable direct ID mapping (use decode->encode)",
    )
    args = parser.parse_args()
    set_seed(args.seed)

    print("SuperPos + Soft Bridge + LM Head Evaluation")
    print(f"Source prompt:   {args.src_prompt}")
    print(f"Source model:    {args.src_model}")
    print(f"Target model:    {args.tgt_model}")
    print(f"Bridge K sweep:  {args.bridge_k}")
    print(f"Local norm:      {not args.no_local_norm}")
    print(f"Direct ID:       {not args.no_direct_id}")
    print(f"Pos/Neg words:   '{args.pos_word}' / '{args.neg_word}'")

    # 1. Load the SuperPos checkpoint and activate its saved weights.
    print("\n[1/5] Loading SuperPos checkpoint...")
    ckpt = torch.load(args.src_prompt, map_location="cpu")
    sampled_ids = ckpt["sampled_ids"].long()
    prob, temperature = activate_superpos_weights(ckpt)

    if not args.bridge_k or any(k < 1 or k > prob.size(1) for k in args.bridge_k):
        raise ValueError(
            f"--bridge-k values must be in [1, {prob.size(1)}], got {args.bridge_k}"
        )
    print(f"  Checkpoint temperature: {temperature}")

    source_tokenizer = AutoTokenizer.from_pretrained(args.src_model)
    target_tokenizer = AutoTokenizer.from_pretrained(args.tgt_model)
    if not args.no_direct_id:
        assert_sampled_ids_aligned(sampled_ids, source_tokenizer, target_tokenizer)

    total_weights = prob.numel()
    near_zero = (prob < 0.001).sum().item()
    print(
        f"  Weights: {prob.shape}, near-zero (<0.001): "
        f"{near_zero}/{total_weights} ({100 * near_zero / total_weights:.1f}%)"
    )

    # 2. Compute source prompt embeddings once.
    print("[2/5] Computing source prompt embeddings...")
    source_embeddings, _ = get_embedding_matrix(args.src_model)
    source_prompt_embeddings = prob @ source_embeddings[sampled_ids].float()
    print(f"  Source prompt shape: {source_prompt_embeddings.shape}")

    print("  Loading target bridge embeddings once...")
    target_embeddings, target_hidden = get_embedding_matrix(args.tgt_model)
    bridge_resources = {
        "src_tokenizer": source_tokenizer,
        "tgt_tokenizer": target_tokenizer,
        "src_embeddings": source_embeddings,
        "tgt_embeddings": target_embeddings,
        "tgt_hidden": target_hidden,
        "min_shared_vocab": min(source_tokenizer.vocab_size, target_tokenizer.vocab_size),
    }

    # 3. Build every soft bridge prompt while reusing the checkpoint and source prompt.
    print("[3/5] Soft bridge (basis-constrained, weight-decomposed)...")
    projected_prompts = {}
    diagnostics_by_k = {}

    for bridge_k in args.bridge_k:
        projected, diagnostics = soft_bridge_prompt(
            source_prompt_embeddings,
            args.src_model,
            args.tgt_model,
            topk=bridge_k,
            local_norm=not args.no_local_norm,
            use_direct_id=not args.no_direct_id,
            sampled_ids=sampled_ids,
            prompt_weights=prob,
            weight_mode="softmax",
            weight_temperature=temperature,
            bridge_resources=bridge_resources,
        )
        projected_prompts[bridge_k] = projected
        diagnostics_by_k[bridge_k] = diagnostics

        print(
            f"  K={bridge_k}: gate={diagnostics['gate_mean']:.4f}, "
            f"top1_sim={diagnostics['mean_top1_sim']:.4f}, "
            f"projected_norm={diagnostics['projected_norm']:.4f}"
        )
        for sample_text in diagnostics["sample_texts"][:2]:
            print(f"    {sample_text}")

    # 4. Load the target model and evaluation data once.
    print("[4/5] Loading target model...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.tgt_model,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    target_model.eval()

    print("[5/5] Evaluating K sweep...")
    tokenizer = AutoTokenizer.from_pretrained(args.tgt_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    pos_ids = get_verbalizer_ids(tokenizer, args.pos_word)
    neg_ids = get_verbalizer_ids(tokenizer, args.neg_word)
    print(f"  Positive tokens: {pos_ids} -> {[tokenizer.decode([i]) for i in pos_ids]}")
    print(f"  Negative tokens: {neg_ids} -> {[tokenizer.decode([i]) for i in neg_ids]}")

    _, eval_dataset = load_dataset_by_name(args.dataset, tokenizer)
    dataloader = build_dataloader(eval_dataset, args.batch_size, shuffle=False)

    def evaluate_projected_prompt(projected, bridge_k):
        correct = 0
        total = 0
        prompt_emb = projected.to(device=args.device, dtype=torch.float16)

        for batch in tqdm(dataloader, desc=f"LM Head eval K={bridge_k}"):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["label"].to(args.device)

            batch_size = input_ids.size(0)
            word_embeds = target_model.model.embed_tokens(input_ids)
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
                outputs = target_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=extended_mask,
                )
                next_logits = outputs.logits[:, -1, :].float()
                positive_score = next_logits[:, pos_ids].max(dim=1).values
                negative_score = next_logits[:, neg_ids].max(dim=1).values
                predictions = ((positive_score - negative_score) > 0).long()
                correct += (predictions == labels).sum().item()
                total += batch_size

        return correct / total if total else 0.0

    results = {}
    for bridge_k in args.bridge_k:
        accuracy = evaluate_projected_prompt(projected_prompts[bridge_k], bridge_k)
        results[bridge_k] = accuracy
        print(
            f"SuperPos + Soft Bridge + LM Head "
            f"(K={bridge_k}): {accuracy:.4f} "
            f"(proj_norm={diagnostics_by_k[bridge_k]['projected_norm']:.4f})"
        )

    print("\nK sweep summary:")
    for bridge_k in args.bridge_k:
        print(f"  K={bridge_k:>3}: acc={results[bridge_k]:.4f}")

    return results


if __name__ == "__main__":
    main()
