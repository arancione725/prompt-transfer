#!/usr/bin/env python3
"""Main entry point for Qwen cross-model prompt transfer pipeline.

Usage:
    python -m qwen_cross_model.train_projector --src Qwen/Qwen2.5-1.5B --tgt Qwen/Qwen2.5-7B --dataset sst2

Steps:
    1. Train soft prompt on source model
    2. Train projector (source-prompt -> target-model space)
    3. Evaluate all settings

If you already have a trained source prompt, pass --src-prompt <path> to skip step 1.
"""

import os
import sys
import argparse
import torch

# Ensure the parent directory is importable
PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from qwen_cross_model.config import (
    MODEL_1_5B, MODEL_7B,
    PROMPT_LEN, PROMPT_LR, PROMPT_EPOCHS, PROMPT_BATCH_SIZE,
    PROMPT_ANCHOR_WEIGHT, SUPERPOS_M, NUM_LABELS,
    PROJECTOR_LR, PROJECTOR_EPOCHS, PROJECTOR_BATCH_SIZE,
    OUTPUT_DIR, DEFAULT_SEED,
)
from qwen_cross_model.prompt_tuner import train_prompt, train_superpos_prompt
from qwen_cross_model.projector import train_projector
from qwen_cross_model.utils import (
    load_prompt, evaluate_model, build_dataloader, load_dataset_by_name,
    PromptQwenWrapper, SuperPosPromptWrapper,
    save_superpos_prompt, load_superpos_prompt, transfer_superpos,
    set_seed,
)
from qwen_cross_model.eval import quick_compare


def main():
    parser = argparse.ArgumentParser(description="Qwen Cross-Model Prompt Transfer")
    parser.add_argument("--src", type=str, default=MODEL_1_5B, help="Source model name")
    parser.add_argument("--tgt", type=str, default=MODEL_7B, help="Target model name")
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--prompt-len", type=int, default=PROMPT_LEN)

    parser.add_argument("--src-prompt", type=str, default=None, help="Path to pre-trained source prompt (skip step 1)")
    parser.add_argument("--skip-projector", action="store_true", help="Skip projector training")

    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)

    # Training hyperparameters
    parser.add_argument("--prompt-lr", type=float, default=PROMPT_LR)
    parser.add_argument("--prompt-epochs", type=int, default=PROMPT_EPOCHS)
    parser.add_argument("--prompt-batch-size", type=int, default=PROMPT_BATCH_SIZE)
    parser.add_argument("--anchor-weight", type=float, default=PROMPT_ANCHOR_WEIGHT,
                        help="Cosine-sim regularizer: 0=off, 0.05=default, higher=stronger constraint")
    parser.add_argument("--init-text", type=str, default=None,
                        help="Semantic text for prompt initialization (prevents collapse)")
    parser.add_argument("--superpos", action="store_true",
                        help="Use SuperPos parametrization (prompt = weights @ token_embeddings)")
    parser.add_argument("--superpos-m", type=int, default=SUPERPOS_M,
                        help="Number of sampled tokens for SuperPos (default 128)")
    parser.add_argument("--projector-lr", type=float, default=PROJECTOR_LR)
    parser.add_argument("--projector-epochs", type=int, default=PROJECTOR_EPOCHS)
    parser.add_argument("--projector-batch-size", type=int, default=PROJECTOR_BATCH_SIZE)

    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs for DataParallel (0 = auto-detect all, 1 = single GPU)")
    parser.add_argument("--resume-prompt", type=str, default=None,
                        help="Resume prompt training from checkpoint (e.g. outputs_qwen/prompt_checkpoint.pt)")
    parser.add_argument("--resume-projector", type=str, default=None,
                        help="Resume projector training from checkpoint (e.g. outputs_qwen/projector_checkpoint.pt)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Set by torchrun for DDP")
    parser.add_argument("--save-name", type=str, default=None, help="Custom checkpoint filename")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed used for reproducible prompt training")

    args = parser.parse_args()

    set_seed(args.seed)

    # DDP mode: launched by torchrun
    is_ddp = args.local_rank >= 0 or "LOCAL_RANK" in os.environ
    if is_ddp:
        device = None  # projector.py detects DDP via LOCAL_RANK and sets device internally
        num_gpus = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    else:
        num_gpus = args.num_gpus if args.num_gpus > 0 else torch.cuda.device_count()
        if num_gpus == 0:
            num_gpus = 1
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Device: {device}  |  GPUs: {num_gpus}")

    # ------------------------------------------------------------------
    # SuperPos mode: train on source, transfer to target (training-free)
    # ------------------------------------------------------------------
    if args.superpos:
        if args.src_prompt:
            src_prompt_path = args.src_prompt
            print(f"\n>>> Using existing SuperPos prompt: {src_prompt_path}")
        else:
            _, metrics = train_superpos_prompt(
                model_name=args.src,
                dataset_name=args.dataset,
                prompt_len=args.prompt_len,
                m=args.superpos_m,
                learning_rate=args.prompt_lr,
                num_epochs=args.prompt_epochs,
                batch_size=args.prompt_batch_size,
                output_dir=args.output_dir,
                device=device,
                num_gpus=num_gpus,
                resume_from=args.resume_prompt,
                save_name=args.save_name,
                seed=args.seed,
            )
            src_prompt_path = metrics["save_path"]

        # Only rank 0 runs transfer + eval; other DDP processes exit
        is_main = int(os.environ.get("LOCAL_RANK", -1)) <= 0
        if not is_main:
            return

        # Transfer to target (training-free)
        print(f"\n{'='*60}")
        print(f"SuperPos Transfer (Training-Free)")
        print(f"Source prompt: {src_prompt_path}")
        print(f"Target model: {args.tgt}")
        print(f"{'='*60}")

        if device is None:
            device = torch.device("cuda:0")

        projected, ckpt = transfer_superpos(src_prompt_path, args.tgt)
        print(f"  Projected prompt: {projected.shape}")

        # LM Head mode: no classification head to project — use eval_superpos_lmhead.py
        if "classification_head" not in ckpt:
            print(f"\n  LM Head checkpoint detected — no classification head to project.")
            print(f"  Use eval_superpos_lmhead.py for evaluation.")
            return

        # Build target wrapper + project head
        tgt_wrapper = PromptQwenWrapper(args.tgt, prompt_len=args.prompt_len, num_labels=NUM_LABELS)
        tgt_wrapper.to(device)
        with torch.no_grad():
            tgt_wrapper.prompt_embeddings.weight.copy_(projected.to(device))

        # Project head via ridge
        from qwen_cross_model.align import align_embeddings_by_token_text, compute_mapping, project_head
        src_aligned, tgt_aligned, _, _, _, _ = align_embeddings_by_token_text(args.src, args.tgt)
        W, src_mean, tgt_mean, _ = compute_mapping(src_aligned, tgt_aligned, reg_lambda=1e-3)
        head_w = ckpt["classification_head"]["weight"].clone()
        head_b = ckpt["classification_head"]["bias"].clone()
        thw, thb = project_head(head_w, head_b, W, src_mean=src_mean, tgt_mean=tgt_mean)
        with torch.no_grad():
            tgt_wrapper.classification_head.weight.copy_(thw.to(device))
            tgt_wrapper.classification_head.bias.copy_(thb.to(device))

        # Evaluate
        _, eval_ds = load_dataset_by_name(args.dataset, tgt_wrapper.tokenizer)
        eval_loader = build_dataloader(eval_ds, args.prompt_batch_size * 2, shuffle=False)

        # Source direct
        src_w = load_superpos_prompt(args.src, src_prompt_path, device=device)
        src_acc = evaluate_model(src_w, eval_loader, device, desc="Src direct")
        # Transferred
        tgt_acc = evaluate_model(tgt_wrapper, eval_loader, device, desc="SuperPos transfer")

        from tabulate import tabulate
        print("\n" + tabulate([
            ["Source direct (SuperPos)", f"{src_acc:.4f}"],
            ["SuperPos transfer", f"{tgt_acc:.4f}"],
        ], headers=["Setting", "Accuracy"], tablefmt="grid"))
        return

    # ------------------------------------------------------------------
    # Standard mode: train prompt + projector
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
