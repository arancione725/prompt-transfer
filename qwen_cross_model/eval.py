"""Evaluation: direct, cross-model, and cross-task."""

import torch
from tabulate import tabulate

from .config import OUTPUT_DIR, PROMPT_LEN, NUM_LABELS
from .utils import (
    PromptQwenWrapper,
    load_dataset_by_name,
    load_prompt,
    get_prompt_tensor,
    set_prompt_tensor,
    evaluate_model,
    build_dataloader,
)
from .projector import PromptProjector


def evaluate_direct(model_name, prompt_path, dataset_name="sst2", device=None):
    """Evaluate a prompt directly on its source model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[Eval] Direct: {model_name} + {dataset_name}")
    wrapper = load_prompt(model_name, prompt_path, device=device)
    _, eval_ds = load_dataset_by_name(dataset_name, wrapper.tokenizer)
    eval_loader = build_dataloader(eval_ds, 32, shuffle=False)
    acc = evaluate_model(wrapper, eval_loader, device)
    print(f"  Accuracy: {acc:.4f}")
    return acc


def evaluate_cross_model(src_model_name, tgt_model_name, src_prompt_path, projector_path, dataset_name="sst2", device=None):
    """Evaluate a projected prompt on the target model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[Eval] Cross-model: {src_model_name} -> {tgt_model_name} + {dataset_name}")

    # Load source prompt embedding
    src_ckpt = torch.load(src_prompt_path, map_location="cpu")
    src_prompt_emb = src_ckpt["prompt_embeddings"]["weight"].clone()

    # Load projector
    proj_ckpt = torch.load(projector_path, map_location="cpu")
    src_hidden = src_prompt_emb.size(1)

    # Build target model and load its classification head
    head_path = projector_path.replace("projector.pt", "target_head.pt")
    tgt_wrapper = PromptQwenWrapper(tgt_model_name, prompt_len=src_prompt_emb.size(0), num_labels=NUM_LABELS)
    tgt_wrapper.to(device)

    head_ckpt = torch.load(head_path, map_location=device)
    if "classification_head" in head_ckpt:
        tgt_wrapper.classification_head.load_state_dict(head_ckpt["classification_head"])

    projector = PromptProjector(src_prompt_emb.size(0), src_hidden, tgt_wrapper.hidden_size).to(device)
    projector.load_state_dict(proj_ckpt)

    # Project and inject
    with torch.no_grad():
        projected = projector(src_prompt_emb.to(device))
        set_prompt_tensor(tgt_wrapper, projected)

    _, eval_ds = load_dataset_by_name(dataset_name, tgt_wrapper.tokenizer)
    eval_loader = build_dataloader(eval_ds, 32, shuffle=False)
    acc = evaluate_model(tgt_wrapper, eval_loader, device)
    print(f"  Accuracy: {acc:.4f}")
    return acc


def quick_compare(
    model_1_5b, model_7b,
    prompt_1_5b_path,
    projector_path=None,
    dataset_name="sst2",
    device=None,
):
    """Print a comparison table of all evaluation scenarios."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []

    # 1. 1.5B direct
    if prompt_1_5b_path and model_1_5b:
        acc = evaluate_direct(model_1_5b, prompt_1_5b_path, dataset_name, device)
        rows.append(["1.5B direct", f"{acc:.4f}"])

    # 3. Cross-model (if projector available)
    if projector_path and model_7b:
        acc = evaluate_cross_model(
            model_1_5b, model_7b, prompt_1_5b_path, projector_path, dataset_name, device
        )
        rows.append(["1.5B -> 7B (projected)", f"{acc:.4f}"])

    print("\n" + tabulate(rows, headers=["Setting", "Accuracy"], tablefmt="grid"))
    return rows
