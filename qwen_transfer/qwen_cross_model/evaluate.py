#!/usr/bin/env python3
"""Post-hoc labeled target evaluation, deliberately isolated from refinement."""

import argparse
import csv
import json
from pathlib import Path
import torch

from .config import DATASET, MAX_LENGTH, TARGET_MODEL
from .data import build_loader, load_classification_data
from .rfa import forward_next_token_logits, load_model, load_tokenizer, validate_sampled_ids
from .superpos import file_sha256, load_checkpoint, prompt_from_coordinates, verbalizer_ids


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate direct and RFA-refined target prompts")
    p.add_argument("--source-checkpoint", required=True)
    p.add_argument("--refined-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-model", default=TARGET_MODEL)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default=None)
    return p.parse_args(argv)


@torch.inference_mode()
def _accuracy(model, loader, prompt, negative_ids, positive_ids):
    input_device = model.get_input_embeddings().weight.device
    correct = total = 0
    for batch in loader:
        logits = forward_next_token_logits(model, batch["input_ids"].to(input_device),
                                           batch["attention_mask"].to(input_device), prompt)
        scores = torch.stack([logits[:, negative_ids].max(-1).values,
                              logits[:, positive_ids].max(-1).values], -1)
        labels = batch["label"]
        correct += int((scores.argmax(-1).cpu() == labels).sum())
        total += labels.numel()
    return correct / max(total, 1)


def run(args):
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = out / "target_evaluation.json", out / "target_evaluation.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("Refusing to overwrite an existing evaluation report")
    source = load_checkpoint(args.source_checkpoint)
    refined = torch.load(Path(args.refined_checkpoint), map_location="cpu", weights_only=True)
    if refined.get("type") != "rfa_refined_target_prompt":
        raise ValueError("Not an RFA-refined target checkpoint")
    if refined.get("target_model") != args.target_model:
        raise ValueError("Refined checkpoint target_model does not match --target-model")
    if refined.get("source_checkpoint_sha256") != file_sha256(source.path):
        raise ValueError("Refined checkpoint source content hash does not match")
    target_revision = refined.get("target_model_revision")
    tokenizer = load_tokenizer(args.target_model, target_revision)
    source_revision = source.payload.get("source_tokenizer_revision") or source.payload.get("source_model_revision")
    validate_sampled_ids(source.sampled_ids,
                         load_tokenizer(source.source_model, source_revision), tokenizer)
    _, validation = load_classification_data(args.dataset, tokenizer, args.max_length)
    loader = build_loader(validation, args.batch_size)
    model = load_model(args.target_model, args.dtype, args.device, args.device_map,
                       target_revision)
    expected_revision = refined.get("target_model_revision")
    actual_revision = getattr(model.config, "_commit_hash", None)
    if expected_revision and actual_revision and expected_revision != actual_revision:
        raise ValueError("Target model revision differs from refinement checkpoint")
    direct_prompt = prompt_from_coordinates(source.coordinates, source.sampled_ids,
                                            model.get_input_embeddings().weight)
    refined_logits = refined.get("target_prompt_logits")
    if not torch.is_tensor(refined_logits) or refined_logits.shape != source.prompt_logits.shape:
        raise ValueError("Refined target_prompt_logits have an invalid shape")
    if not torch.isfinite(refined_logits).all():
        raise ValueError("Refined target_prompt_logits contain NaN or Inf")
    refined_coordinates = torch.softmax(refined_logits.float() / source.temperature, dim=-1)
    refined_prompt = prompt_from_coordinates(refined_coordinates, source.sampled_ids,
                                             model.get_input_embeddings().weight)
    neg, pos = verbalizer_ids(tokenizer, "negative"), verbalizer_ids(tokenizer, "positive")
    direct = _accuracy(model, loader, direct_prompt, neg, pos)
    rfa = _accuracy(model, loader, refined_prompt, neg, pos)
    report = {"source_checkpoint": str(source.path),
              "refined_checkpoint": str(Path(args.refined_checkpoint).resolve()),
              "target_model": args.target_model, "dataset": args.dataset,
              "direct_accuracy": direct, "refined_accuracy": rfa,
              "accuracy_change": rfa - direct,
              "best_refinement_objective": float(refined["best_objective"]),
              "best_rfa": float(refined["best_rfa"]),
              "best_stay_kl": float(refined["best_stay_kl"]),
              "probe_fingerprint": refined["probe_bank"]["fingerprint"],
              "anchor_mode": refined["anchors"]["mode"],
              "note": "Labels are used only in this post-hoc evaluation."}
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=report.keys()); writer.writeheader(); writer.writerow(report)
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
