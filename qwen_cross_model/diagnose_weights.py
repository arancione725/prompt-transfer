#!/usr/bin/env python3
"""SuperPos weight diagnostics — diagnose per-position sparsity & diversity.

Usage:
    python diagnose_weights.py
    python diagnose_weights.py --ckpt path/to/checkpoint.pt --temp 0.5
    python diagnose_weights.py --ckpt outputs_qwen/superpos/old/prompt_superpos_...pt  # old code
"""

import argparse
import gc
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from qwen_cross_model.utils import (
    activate_superpos_weights,
    assert_sampled_ids_aligned,
)


DEFAULT_K_VALUES = [3, 5, 7, 8, 10, 20]


def _resolve_k_values(requested, checkpoint, basis_size):
    values = requested or checkpoint.get("topk_values") or DEFAULT_K_VALUES
    values = list(dict.fromkeys(int(value) for value in values))
    invalid = [value for value in values if value < 1 or value > basis_size]
    if invalid:
        raise ValueError(
            f"K values must be in [1, {basis_size}], got invalid values {invalid}"
        )
    return values


def _topk_stats(probabilities, k):
    selected, indices = probabilities.topk(k, dim=-1)
    mass = selected.sum(dim=-1)
    if k < probabilities.size(1):
        boundary = probabilities.topk(k + 1, dim=-1).values
        gap = boundary[:, k - 1] - boundary[:, k]
    else:
        gap = torch.full_like(mass, float("nan"))
    return mass, gap, indices


def _row_js(first, second):
    mixture = 0.5 * (first + second)
    first_log = first.clamp_min(1e-12).log()
    second_log = second.clamp_min(1e-12).log()
    mixture_log = mixture.clamp_min(1e-12).log()
    return 0.5 * (
        (first * (first_log - mixture_log)).sum(dim=-1)
        + (second * (second_log - mixture_log)).sum(dim=-1)
    )


def main():
    parser = argparse.ArgumentParser(description="SuperPos weight diagnostics")
    parser.add_argument(
        "--ckpt",
        default="outputs_qwen/superpos_prompt_transfer_main_clean/prompt_transfer_main_clean.pt",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=None,
        help="Override checkpoint temperature; default reads the checkpoint",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Source tokenizer; default reads source_model from the checkpoint",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=None,
        help="K values for mass and boundary diagnostics",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Optional second checkpoint for run-to-run stability comparison",
    )
    parser.add_argument(
        "--target-model",
        default=None,
        help="Optionally load target embeddings and report direct-transfer geometry",
    )
    parser.add_argument(
        "--show-positions",
        type=int,
        default=5,
        help="Number of lowest-mass prompt positions to display",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    prob, temperature = activate_superpos_weights(ckpt, temperature=args.temp)
    sampled_ids = ckpt["sampled_ids"].long()
    prompt_len, basis_size = prob.shape
    k_values = _resolve_k_values(args.k_values, ckpt, basis_size)
    focus_k = max((k for k in k_values if k <= 10), default=min(k_values))
    source_model = args.model or ckpt.get("source_model") or "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(source_model)

    entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
    effective_support = entropy.exp()
    top1_value, top1_index = prob.max(dim=-1)
    top3_indices = prob.topk(min(3, basis_size), dim=-1).indices
    stats_by_k = {
        k: _topk_stats(prob, k)
        for k in k_values
    }

    print("=" * 72)
    print("Checkpoint metadata")
    print("=" * 72)
    print(f"Checkpoint:              {args.ckpt}")
    print(f"Source model:            {source_model}")
    print(f"Shape:                   [{prompt_len}, {basis_size}]")
    print(f"Temperature:             {temperature}")
    print(f"Seed:                    {ckpt.get('seed', 'not saved')}")
    print(f"Training objective:      {ckpt.get('training_objective', 'standard CE or legacy')}")
    print(f"Training K values:       {ckpt.get('topk_values', 'not saved')}")
    print(f"Full CE weight:          {ckpt.get('full_ce_weight', 'not saved')}")
    print(f"Top-K CE weight:         {ckpt.get('topk_ce_weight', 'not saved')}")
    print(f"JS weight / temperature: {ckpt.get('js_weight', 'not saved')} / "
          f"{ckpt.get('js_temperature', 'not saved')}")
    print(f"Best source eval acc:    {ckpt.get('best_source_eval_accuracy', 'not saved')}")

    print("\n" + "=" * 72)
    print("Weight concentration")
    print("=" * 72)
    print(
        f"Entropy:          mean={entropy.mean():.4f}  "
        f"min={entropy.min():.4f}  max={entropy.max():.4f}"
    )
    print(
        f"Effective tokens: mean={effective_support.mean():.2f}  "
        f"min={effective_support.min():.2f}  max={effective_support.max():.2f}"
    )
    print(
        f"Top-1 weight:     mean={top1_value.mean():.4f}  "
        f"min={top1_value.min():.4f}  max={top1_value.max():.4f}"
    )
    unique_top1 = len(set(sampled_ids[top1_index].tolist()))
    near_one_hot = (top1_value >= 0.95).sum().item()
    print(f"Unique Top-1 tokens: {unique_top1}/{prompt_len} prompt positions")
    print(f"Near one-hot positions (Top-1 >= 0.95): {near_one_hot}/{prompt_len}")

    print("\nK     mass_mean  mass_min  tail_mean  boundary_gap_mean  boundary_gap_median")
    print("-" * 78)
    for k in k_values:
        mass, gap, _ = stats_by_k[k]
        if torch.isnan(gap).all():
            gap_mean = "n/a"
            gap_median = "n/a"
        else:
            gap_mean = f"{gap.mean().item():.6f}"
            gap_median = f"{gap.median().item():.6f}"
        print(
            f"{k:>2}    {mass.mean():.4f}     {mass.min():.4f}    "
            f"{(1 - mass).mean():.4f}     {gap_mean:>10}           {gap_median:>10}"
        )

    focus_mass, focus_gap, focus_indices = stats_by_k[focus_k]
    unique_focus = len(set(sampled_ids[focus_indices.reshape(-1)].tolist()))
    low_mass_count = (focus_mass < 0.80).sum().item()
    print(f"\nFocus K={focus_k}: unique basis tokens={unique_focus}/{basis_size}")
    print(f"Positions with mass@{focus_k} < 0.80: {low_mass_count}/{prompt_len}")

    show_count = max(0, min(args.show_positions, prompt_len))
    if show_count:
        lowest_mass_positions = focus_mass.topk(show_count, largest=False).indices
        print(f"\nLowest mass@{focus_k} positions")
        print(f"{'pos':>4}  {'effective':>9}  {'top1':>7}  "
              f"{'mass':>7}  {'gap':>9}  top-3 tokens")
        print("-" * 82)
        for position in lowest_mass_positions.tolist():
            tokens = [
                tokenizer.decode([sampled_ids[index].item()])
                for index in top3_indices[position]
            ]
            print(
                f"{position:4d}  {effective_support[position]:9.2f}  "
                f"{top1_value[position]:7.4f}  {focus_mass[position]:7.4f}  "
                f"{focus_gap[position]:9.6f}  {tokens}"
            )

    if args.compare:
        other_ckpt = torch.load(args.compare, map_location="cpu")
        other_prob, other_temperature = activate_superpos_weights(other_ckpt)
        other_ids = other_ckpt["sampled_ids"].long()

        print("\n" + "=" * 72)
        print("Checkpoint comparison")
        print("=" * 72)
        print(f"Other checkpoint: {args.compare}")
        print(f"Other temperature: {other_temperature}")
        print(f"Same shape: {tuple(prob.shape) == tuple(other_prob.shape)}")
        same_basis = torch.equal(sampled_ids, other_ids)
        print(f"Same sampled_ids and order: {same_basis}")

        if prob.shape != other_prob.shape or not same_basis:
            print("Weight similarity skipped because the SuperPos bases are not aligned.")
        else:
            row_js = _row_js(prob, other_prob)
            mean_l1 = (prob - other_prob).abs().sum(dim=-1).mean()
            top1_agreement = (prob.argmax(dim=-1) == other_prob.argmax(dim=-1)).float().mean()
            print(f"Mean row JS:        {row_js.mean():.6f}")
            print(f"Max row JS:         {row_js.max():.6f}")
            print(f"Mean row L1:        {mean_l1:.6f}")
            print(f"Top-1 agreement:    {top1_agreement:.2%}")
            print("Top-K set overlap:")
            for k in k_values:
                first_topk = prob.topk(k, dim=-1).indices
                second_topk = other_prob.topk(k, dim=-1).indices
                overlap = []
                for position in range(prompt_len):
                    first_set = set(first_topk[position].tolist())
                    second_set = set(second_topk[position].tolist())
                    overlap.append(len(first_set & second_set) / k)
                print(f"  K={k:>2}: {sum(overlap) / len(overlap):.2%}")

    if args.target_model:
        from qwen_cross_model.align import get_embedding_matrix

        target_tokenizer = AutoTokenizer.from_pretrained(args.target_model)
        assert_sampled_ids_aligned(sampled_ids, tokenizer, target_tokenizer)
        target_embedding_matrix, _ = get_embedding_matrix(args.target_model)
        target_basis = target_embedding_matrix[sampled_ids].float().clone()
        del target_embedding_matrix
        gc.collect()

        full_projected = prob @ target_basis
        print("\n" + "=" * 72)
        print(f"Direct-transfer geometry: {args.target_model}")
        print("=" * 72)
        print(
            f"Full prompt norm: "
            f"{full_projected.norm(p=2, dim=-1).mean().item():.4f}"
        )
        print("K     prompt_norm  cancellation  cosine_to_full")
        print("-" * 56)
        for k in k_values:
            values, indices = prob.topk(k, dim=-1)
            weights = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            selected_embeddings = target_basis[indices]
            projected = (weights.unsqueeze(-1) * selected_embeddings).sum(dim=1)
            expected_norm = (
                weights * selected_embeddings.norm(p=2, dim=-1)
            ).sum(dim=-1).clamp_min(1e-12)
            cancellation = projected.norm(p=2, dim=-1) / expected_norm
            cosine_to_full = F.cosine_similarity(
                projected,
                full_projected,
                dim=-1,
            )
            print(
                f"{k:>2}    {projected.norm(p=2, dim=-1).mean():11.4f}  "
                f"{cancellation.mean():12.4f}  {cosine_to_full.mean():14.4f}"
            )


if __name__ == "__main__":
    main()
