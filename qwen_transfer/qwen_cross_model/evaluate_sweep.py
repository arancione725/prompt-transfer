#!/usr/bin/env python3
"""Post-hoc labeled evaluation for a completed label-free K sweep."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from .config import DATASET, MAX_LENGTH
from .data import build_loader, load_classification_data
from .evaluate import _accuracy
from .refine_target import build_topk_support, prompt_from_topk_coordinates
from .rfa import load_model, load_tokenizer, validate_sampled_ids
from .superpos import file_sha256, load_checkpoint, runtime_versions, verbalizer_ids
from .sweep_k import CACHE_NAME, SELECTION_JSON, SWEEP_JSON, validate_cache


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate every direct/refined K candidate after label-free selection"
    )
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-checkpoint",
        help="Optional path override when the cached absolute source path moved",
    )
    parser.add_argument("--target-model", help="Defaults to the model recorded in the cache")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-length", type=int,
        help="Defaults to the max length recorded by the sweep cache",
    )
    parser.add_argument(
        "--dtype", choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or (args.max_length is not None and args.max_length < 1):
        parser.error("--batch-size and --max-length must be positive")
    return args


def _validate_candidate(payload, basis_k, source, cache_sha256, target_model):
    if payload.get("type") != "rfa_refined_target_prompt_topk":
        raise ValueError(f"K={basis_k}: unsupported refined checkpoint type")
    if payload.get("format_version") != 2:
        raise ValueError(f"K={basis_k}: unsupported checkpoint format")
    if int(payload.get("basis_k", -1)) != basis_k:
        raise ValueError(f"K={basis_k}: checkpoint K mismatch")
    if payload.get("source_checkpoint_sha256") != file_sha256(source.path):
        raise ValueError(f"K={basis_k}: source checkpoint hash mismatch")
    if payload.get("cache_sha256") != cache_sha256:
        raise ValueError(f"K={basis_k}: cache hash mismatch")
    if payload.get("target_model") != target_model:
        raise ValueError(f"K={basis_k}: target model mismatch")
    support = payload.get("support_token_ids")
    direct = payload.get("source_topk_coordinates")
    refined = payload.get("target_topk_coordinates")
    if not all(torch.is_tensor(value) for value in (support, direct, refined)):
        raise ValueError(f"K={basis_k}: missing support or coordinates")
    expected = (source.prompt_logits.size(0), basis_k)
    if support.shape != expected or direct.shape != expected or refined.shape != expected:
        raise ValueError(f"K={basis_k}: invalid support/coordinate shape")
    if (
        support.min() < 0
        or not torch.isfinite(direct).all()
        or not torch.isfinite(refined).all()
        or direct.min() < 0
        or refined.min() < 0
    ):
        raise ValueError(f"K={basis_k}: invalid support/coordinate values")
    if not torch.allclose(
        direct.sum(-1), torch.ones(direct.size(0)), atol=1e-6, rtol=1e-6
    ) or not torch.allclose(
        refined.sum(-1), torch.ones(refined.size(0)), atol=1e-6, rtol=1e-6
    ):
        raise ValueError(f"K={basis_k}: coordinates are not row-normalized")
    expected_support = build_topk_support(source, basis_k)
    if not torch.equal(support.long(), expected_support.token_ids):
        raise ValueError(f"K={basis_k}: support is not the source prompt Top-K")
    if not torch.allclose(
        direct.float(), expected_support.source_coordinates, atol=1e-7, rtol=1e-7
    ):
        raise ValueError(f"K={basis_k}: direct coordinates do not match source Top-K")
    target_logits = payload.get("target_active_logits")
    if not torch.is_tensor(target_logits) or target_logits.shape != direct.shape:
        raise ValueError(f"K={basis_k}: invalid target active logits")
    reconstructed = torch.softmax(target_logits.float() / source.temperature, dim=-1)
    if not torch.allclose(refined.float(), reconstructed, atol=1e-7, rtol=1e-7):
        raise ValueError(f"K={basis_k}: refined coordinates do not match saved logits")
    return support.long(), direct.float(), refined.float()


def _average_ranks(values):
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = (start + 1 + stop) / 2.0
        for index in order[start:stop]:
            ranks[index] = average
        start = stop
    return ranks


def spearman(values_a, values_b):
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    rank_a, rank_b = _average_ranks(values_a), _average_ranks(values_b)
    mean_a = sum(rank_a) / len(rank_a)
    mean_b = sum(rank_b) / len(rank_b)
    numerator = sum(
        (first - mean_a) * (second - mean_b)
        for first, second in zip(rank_a, rank_b)
    )
    denominator = math.sqrt(
        sum((value - mean_a) ** 2 for value in rank_a)
        * sum((value - mean_b) ** 2 for value in rank_b)
    )
    return numerator / denominator if denominator else None


def run(args):
    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "k_sweep_evaluation.json"
    csv_path = output_dir / "k_sweep_evaluation.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("Refusing to overwrite an existing sweep evaluation")

    cache_path = sweep_dir / CACHE_NAME
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    source_path = args.source_checkpoint or cache.get("source_checkpoint")
    if not source_path:
        raise ValueError("Source checkpoint path is unavailable")
    source = load_checkpoint(source_path)
    target_model_name = args.target_model or cache.get("target_model")
    if not target_model_name:
        raise ValueError("Target model is unavailable")
    cache_sha256 = validate_cache(
        cache, cache_path, source, target_model_name
    )
    evaluation_max_length = args.max_length or int(cache.get("max_length", MAX_LENGTH))

    sweep_report = json.loads((sweep_dir / SWEEP_JSON).read_text(encoding="utf-8"))
    selection = json.loads((sweep_dir / SELECTION_JSON).read_text(encoding="utf-8"))
    if sweep_report.get("cache_sha256") != cache_sha256:
        raise ValueError("Sweep report cache hash mismatch")
    selected_k = int(selection["selected_k"])
    if int(sweep_report.get("selection", {}).get("selected_k", -1)) != selected_k:
        raise ValueError("Selection file and sweep report disagree")
    target_revision = cache.get("target_model_revision")
    target_tokenizer = load_tokenizer(
        target_model_name, cache.get("target_tokenizer_revision") or target_revision
    )
    source_revision = (
        source.payload.get("source_tokenizer_revision")
        or source.payload.get("source_model_revision")
    )
    source_tokenizer = load_tokenizer(source.source_model, source_revision)
    validate_sampled_ids(source.sampled_ids, source_tokenizer, target_tokenizer)
    _, validation_data = load_classification_data(
        args.dataset, target_tokenizer, evaluation_max_length
    )
    loader = build_loader(validation_data, args.batch_size)
    model = load_model(
        target_model_name, args.dtype, args.device, args.device_map, target_revision
    )
    actual_revision = getattr(model.config, "_commit_hash", None)
    if target_revision and actual_revision and target_revision != actual_revision:
        raise ValueError("Evaluation target model revision differs from the sweep cache")
    negative_ids = verbalizer_ids(target_tokenizer, "negative")
    positive_ids = verbalizer_ids(target_tokenizer, "positive")

    sweep_rows = sweep_report.get("rows", [])
    if not sweep_rows:
        raise ValueError("Sweep report contains no K rows")
    rows = []
    for sweep_row in sorted(sweep_rows, key=lambda row: int(row["basis_k"])):
        basis_k = int(sweep_row["basis_k"])
        checkpoint_path = (
            sweep_dir / f"k_{basis_k:03d}" / f"refined_target_k{basis_k:03d}.pt"
        )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        support, direct_coordinates, refined_coordinates = _validate_candidate(
            payload, basis_k, source, cache_sha256, target_model_name
        )
        embedding = model.get_input_embeddings().weight
        direct_prompt = prompt_from_topk_coordinates(
            direct_coordinates, support, embedding
        )
        refined_prompt = prompt_from_topk_coordinates(
            refined_coordinates, support, embedding
        )
        direct_accuracy = _accuracy(
            model, loader, direct_prompt, negative_ids, positive_ids
        )
        refined_accuracy = _accuracy(
            model, loader, refined_prompt, negative_ids, positive_ids
        )
        rows.append({
            "basis_k": basis_k,
            "selected_by_rfa": basis_k == selected_k,
            "direct_validation_rfa": float(sweep_row["direct_validation_rfa"]),
            "refined_validation_rfa": float(sweep_row["refined_validation_rfa"]),
            "direct_accuracy": direct_accuracy,
            "refined_accuracy": refined_accuracy,
            "accuracy_change": refined_accuracy - direct_accuracy,
            "checkpoint": str(checkpoint_path),
        })

    direct_oracle = max(rows, key=lambda row: (row["direct_accuracy"], -row["basis_k"]))
    refined_oracle = max(rows, key=lambda row: (row["refined_accuracy"], -row["basis_k"]))
    selected_rows = [row for row in rows if row["selected_by_rfa"]]
    if len(selected_rows) != 1:
        raise ValueError("Selected K is missing from, or duplicated in, the sweep rows")
    selected = selected_rows[0]
    correlation = spearman(
        [-row["refined_validation_rfa"] for row in rows],
        [row["refined_accuracy"] for row in rows],
    )
    summary = {
        "selected_k": selected_k,
        "selected_refined_accuracy": selected["refined_accuracy"],
        "direct_oracle_k": direct_oracle["basis_k"],
        "direct_oracle_accuracy": direct_oracle["direct_accuracy"],
        "refined_oracle_k": refined_oracle["basis_k"],
        "refined_oracle_accuracy": refined_oracle["refined_accuracy"],
        "top1_selection_regret": (
            refined_oracle["refined_accuracy"] - selected["refined_accuracy"]
        ),
        "spearman_negative_rfa_vs_refined_accuracy": correlation,
    }
    report = {
        "type": "rfa_k_sweep_posthoc_evaluation",
        "format_version": 1,
        "source_checkpoint": str(source.path),
        "source_checkpoint_sha256": file_sha256(source.path),
        "cache_sha256": cache_sha256,
        "target_model": target_model_name,
        "dataset": args.dataset,
        "evaluation_config": {
            "batch_size": args.batch_size,
            "max_length": evaluation_max_length,
            "dtype": args.dtype,
            "device": args.device,
            "device_map": args.device_map,
            "target_model_revision": target_revision,
            "target_tokenizer_revision": cache.get("target_tokenizer_revision"),
            "negative_verbalizer_ids": negative_ids,
            "positive_verbalizer_ids": positive_ids,
        },
        "summary": summary,
        "rows": rows,
        "note": "Labels are used only here, after label-free refinement and K selection.",
        "runtime_versions": runtime_versions(),
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    return report


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
