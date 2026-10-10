#!/usr/bin/env python3
"""Label-free fixed-support Top-K refinement and K selection.

The cache contains only unlabeled probe text-derived tensors.  Accuracy is
intentionally unavailable in this module; it belongs in evaluate_sweep.py.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import (
    DEFAULT_SEED,
    K_SWEEP_TRAIN_PROBES,
    K_SWEEP_VALIDATION_PROBES,
    K_SWEEP_VALUES,
    MAX_LENGTH,
    REFINE_BATCH_SIZE,
    REFINE_EPOCHS,
    REFINE_LR,
    RFA_TEMPERATURE,
    STAY_WEIGHT,
    TARGET_MODEL,
)
from .refine_target import _enable_refinement_mode, refine_topk_from_cache
from .rfa import (
    ProbeBank,
    ProbeDataset,
    _fingerprint_texts,
    _read_probe_file,
    build_anchor_plan,
    collect_logits,
    load_model,
    load_sst2_cached_first,
    load_tokenizer,
    set_seed,
    validate_probe_encoding_alignment,
    validate_sampled_ids,
    validate_tokenizer_alignment,
)
from .superpos import file_sha256, load_checkpoint, prompt_from_coordinates, runtime_versions


CACHE_NAME = "rfa_sweep_cache.pt"
SELECTION_JSON = "k_selection.json"
SWEEP_JSON = "k_sweep_rfa.json"
SWEEP_CSV = "k_sweep_rfa.csv"
FORBIDDEN_SELECTION_KEYS = ("accuracy", "label")


def _subprocess_module_name(package_name=None):
    """Reuse the package path used by the parent invocation.

    This supports both `qwen_cross_model.sweep_k` when run from the project
    directory and `qwen_transfer.qwen_cross_model.sweep_k` when run from its
    parent directory.
    """
    package_name = package_name or __package__
    if not package_name:
        raise RuntimeError("sweep_k must be launched as a Python module")
    return f"{package_name}.sweep_k"


def _positive_int(parser, name, value):
    if value < 1:
        parser.error(f"--{name} must be positive")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run label-free fixed-support Top-K RFA refinement and select K"
    )
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-model", default=TARGET_MODEL)
    parser.add_argument(
        "--target-revision",
        help="Immutable target commit/revision; required if the loader cannot resolve one",
    )
    parser.add_argument("--dataset", default="sst2")
    parser.add_argument("--probe-file")
    parser.add_argument("--text-field", default="sentence")
    parser.add_argument("--probe-split", default="train")
    parser.add_argument("--train-probes", type=int, default=K_SWEEP_TRAIN_PROBES)
    parser.add_argument("--validation-probes", type=int, default=K_SWEEP_VALIDATION_PROBES)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--k-values", type=int, nargs="+", default=K_SWEEP_VALUES)
    parser.add_argument("--epochs", type=int, default=REFINE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=REFINE_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=REFINE_LR)
    parser.add_argument("--temperature", type=float, default=RFA_TEMPERATURE)
    parser.add_argument("--stay-weight", type=float, default=STAY_WEIGHT)
    parser.add_argument("--anchor-mode", choices=["full", "source-effect"], default="full")
    parser.add_argument("--anchor-coverage", type=float, default=0.95)
    parser.add_argument("--min-anchor-tokens", type=int, default=512)
    parser.add_argument("--max-anchor-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dtype", choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpus", nargs="*", default=None,
        help="Visible logical CUDA IDs for task-parallel workers, e.g. --gpus 0 1 2 3",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-only", action="store_true")

    # Internal worker arguments.  They are deliberately hidden from normal help.
    parser.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--internal-cache-worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--cache-path", help=argparse.SUPPRESS)
    parser.add_argument("--cache-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--worker-k-values", type=int, nargs="+", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    for name in ("train-probes", "validation-probes", "max-length", "batch-size"):
        _positive_int(parser, name, getattr(args, name.replace("-", "_")))
    if args.epochs < 0:
        parser.error("--epochs must be non-negative")
    for name in ("learning_rate", "temperature"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.stay_weight) or args.stay_weight < 0:
        parser.error("--stay-weight must be finite and non-negative")
    if not 0 < args.anchor_coverage <= 1:
        parser.error("--anchor-coverage must be in (0, 1]")
    if args.min_anchor_tokens < 1 or args.max_anchor_tokens < args.min_anchor_tokens:
        parser.error("anchor bounds must satisfy 1 <= min <= max")
    if len(set(args.k_values)) != len(args.k_values) or any(k < 1 for k in args.k_values):
        parser.error("--k-values must be unique positive integers")
    if args.gpus == []:
        parser.error("--gpus requires at least one GPU ID")
    if args.gpus is not None:
        if any(not str(gpu).strip() for gpu in args.gpus):
            parser.error("--gpus cannot contain empty IDs")
        if any(not str(gpu).isdigit() for gpu in args.gpus):
            parser.error("--gpus accepts non-negative numeric CUDA IDs only")
        if len(set(args.gpus)) != len(args.gpus):
            parser.error("--gpus cannot contain duplicate IDs")
    if args.internal_worker and not (
        args.cache_path and args.cache_sha256 and args.worker_k_values
    ):
        parser.error("internal worker requires cache path, hash, and K values")
    if args.internal_cache_worker and not args.cache_path:
        parser.error("internal cache worker requires --cache-path")
    return args


def split_probe_texts(texts, train_count, validation_count, seed, source):
    """Create deterministic, disjoint unlabeled train/validation probe banks."""
    total = int(train_count) + int(validation_count)
    if train_count < 1 or validation_count < 1:
        raise ValueError("Both probe partitions must be non-empty")
    unique_indices = []
    seen_texts = set()
    for index, value in enumerate(texts):
        text = str(value)
        if text not in seen_texts:
            unique_indices.append(index)
            seen_texts.add(text)
    if total > len(unique_indices):
        raise ValueError(
            f"Requested {total} distinct probes but {source} contains only "
            f"{len(unique_indices)} distinct texts"
        )
    sampled = random.Random(int(seed)).sample(unique_indices, total)
    train_indices = sorted(sampled[:train_count])
    validation_indices = sorted(sampled[train_count:])

    def bank(indices, suffix):
        selected = [str(texts[index]) for index in indices]
        return ProbeBank(selected, indices, f"{source}:{suffix}", _fingerprint_texts(selected))

    return bank(train_indices, "refine_train"), bank(validation_indices, "rfa_validation")


def _probe_request(args):
    if args.probe_file:
        path = Path(args.probe_file).expanduser().resolve()
        return {
            "kind": "file",
            "path": str(path),
            "sha256": file_sha256(path),
            "text_field": args.text_field,
        }
    return {
        "kind": "dataset",
        "dataset": args.dataset,
        "split": args.probe_split,
        "text_column": "sentence" if args.dataset == "sst2" else args.text_field,
    }


def _load_probe_partitions(args):
    if args.probe_file:
        texts, source = _read_probe_file(args.probe_file, args.text_field)
    else:
        if args.dataset != "sst2":
            raise ValueError("Only sst2 is built in; use --probe-file for other tasks")
        split = load_sst2_cached_first(args.probe_split)
        # Index only the input column.  The label column is never materialized here.
        texts = [str(value) for value in split["sentence"]]
        source = f"datasets:glue/sst2:{args.probe_split}:sentence"
    return split_probe_texts(
        texts, args.train_probes, args.validation_probes, args.seed, source
    )


def _stack_probe_rows(dataset):
    return {
        "input_ids": torch.stack([row["input_ids"] for row in dataset.rows]),
        "attention_mask": torch.stack([row["attention_mask"] for row in dataset.rows]),
    }


def _bank_payload(bank):
    return {
        "source": bank.source,
        "indices": list(bank.indices),
        "fingerprint": bank.fingerprint,
        "num_probes": len(bank.texts),
        "texts": list(bank.texts),
    }


def _anchor_payload(plan, start, stop, aligned_token_ids):
    if plan is None:
        return None
    local_indices = [value.cpu() for value in plan.indices[start:stop]]
    return {
        "indices": local_indices,
        "selected_token_ids": [
            aligned_token_ids.index_select(0, value).cpu() for value in local_indices
        ],
        "selected_counts": plan.selected_counts[start:stop],
        "achieved_coverages": plan.achieved_coverages[start:stop],
        "aggregate_source_js": plan.aggregate_source_js[start:stop],
        "other_bucket": True,
    }


def _resolve_cache_device(args):
    if args.gpus:
        return f"cuda:{args.gpus[0]}"
    return args.device


def _local_model_manifest(path):
    """Cheap identity for local models: full small-file hashes plus weight stats."""
    root = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Local target model directory contains no files: {root}")
    for item in files:
        relative = item.relative_to(root).as_posix()
        stat = item.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        # Config/tokenizer/index files are small enough to hash completely.
        if stat.st_size <= 16 * 1024 * 1024:
            digest.update(file_sha256(item).encode("ascii"))
    return {
        "kind": "local_manifest_v1",
        "path": str(root),
        "sha256": digest.hexdigest(),
    }


def _model_identity(model_name, resolved_revision):
    local_path = Path(model_name).expanduser()
    if local_path.is_dir():
        return _local_model_manifest(local_path)
    if not resolved_revision:
        raise ValueError(
            "Target model revision could not be resolved. Pass --target-revision "
            "so cached reference logits cannot be mixed with different weights."
        )
    return {
        "kind": "hub_revision",
        "model": model_name,
        "revision": str(resolved_revision),
    }


def prepare_cache(args, cache_path):
    """Compute common source/target tensors exactly once, without target labels."""
    cache_path = Path(cache_path).resolve()
    if cache_path.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {cache_path}")
    checkpoint = load_checkpoint(args.source_checkpoint)
    if max(args.k_values) > checkpoint.sampled_ids.numel():
        raise ValueError(
            f"Requested K={max(args.k_values)} exceeds basis size "
            f"{checkpoint.sampled_ids.numel()}"
        )
    source_revision = checkpoint.payload.get("source_model_revision")
    source_tokenizer_revision = (
        checkpoint.payload.get("source_tokenizer_revision") or source_revision
    )
    if source_revision is None:
        warnings.warn(
            "Source checkpoint has no model revision; exact remote-weight reproducibility "
            "cannot be guaranteed.", RuntimeWarning,
        )
    source_tokenizer = load_tokenizer(checkpoint.source_model, source_tokenizer_revision)
    target_tokenizer = load_tokenizer(args.target_model, args.target_revision)
    target_tokenizer_revision = (
        getattr(target_tokenizer, "init_kwargs", {}).get("_commit_hash")
        or args.target_revision
    )
    vocabulary = validate_tokenizer_alignment(source_tokenizer, target_tokenizer)
    validate_sampled_ids(checkpoint.sampled_ids, source_tokenizer, target_tokenizer)

    train_bank, validation_bank = _load_probe_partitions(args)
    all_texts = train_bank.texts + validation_bank.texts
    validate_probe_encoding_alignment(
        all_texts, source_tokenizer, target_tokenizer, args.max_length
    )
    dataset = ProbeDataset(all_texts, source_tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    stacked = _stack_probe_rows(dataset)
    token_ids = vocabulary.token_ids
    device = _resolve_cache_device(args)

    source_model = load_model(
        checkpoint.source_model, args.dtype, device, None, source_revision
    )
    source_prompt = prompt_from_coordinates(
        checkpoint.coordinates,
        checkpoint.sampled_ids,
        source_model.get_input_embeddings().weight,
    )
    source_reference = collect_logits(
        source_model, loader, None, token_ids, "cache source/reference"
    )
    source_prompted = collect_logits(
        source_model, loader, source_prompt, token_ids, "cache source/full-prompt"
    )
    source_effect = (source_prompted - source_reference).cpu()
    anchor_plan = None
    if args.anchor_mode == "source-effect":
        source_mass = F.softmax(source_effect.double() / args.temperature, dim=-1)
        anchor_plan = build_anchor_plan(
            source_mass,
            args.anchor_coverage,
            args.min_anchor_tokens,
            args.max_anchor_tokens,
        )
        minimum_coverage = min(anchor_plan.achieved_coverages)
        if minimum_coverage + 1e-12 < args.anchor_coverage:
            warnings.warn(
                "Sparse anchor cap prevented requested coverage: "
                f"minimum={minimum_coverage:.6f}", RuntimeWarning,
            )
        del source_mass
    del source_prompt, source_reference, source_prompted, source_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target_model = load_model(
        args.target_model, args.dtype, device, None, target_tokenizer_revision
    )
    target_model.config.use_cache = False
    target_reference = collect_logits(
        target_model, loader, None, token_ids, "cache target/reference"
    ).cpu()
    target_model_revision = (
        getattr(target_model.config, "_commit_hash", None)
        or target_tokenizer_revision
        or args.target_revision
    )
    if target_tokenizer_revision and target_model_revision:
        if target_tokenizer_revision != target_model_revision:
            raise ValueError("Target tokenizer and model resolved to different revisions")

    del target_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_stop = len(train_bank.texts)
    cache = {
        "type": "rfa_k_sweep_cache",
        "format_version": 1,
        "source_checkpoint": str(checkpoint.path),
        "source_checkpoint_sha256": file_sha256(checkpoint.path),
        "source_model": checkpoint.source_model,
        "source_model_revision": source_revision,
        "source_tokenizer_revision": source_tokenizer_revision,
        "target_model": args.target_model,
        "requested_target_revision": args.target_revision,
        "target_model_revision": target_model_revision,
        "target_tokenizer_revision": target_tokenizer_revision,
        "target_model_identity": _model_identity(
            args.target_model, target_model_revision
        ),
        "position_id_mode": checkpoint.position_id_mode,
        "basis_size": checkpoint.sampled_ids.numel(),
        "sampled_ids": checkpoint.sampled_ids,
        "token_ids": token_ids,
        "aligned_vocabulary_size": token_ids.numel(),
        "excluded_special_ids": vocabulary.excluded_special_ids,
        "rfa_temperature": float(args.temperature),
        "anchor_mode": args.anchor_mode,
        "anchor_config": {
            "coverage": args.anchor_coverage,
            "min_tokens": args.min_anchor_tokens,
            "max_tokens": args.max_anchor_tokens,
            "source_only": True,
            "fixed_before_optimization": True,
            "other_bucket": anchor_plan is not None,
        },
        "max_length": args.max_length,
        "dtype": args.dtype,
        "seed": args.seed,
        "probe_request": _probe_request(args),
        "train": {
            "input_ids": stacked["input_ids"][:train_stop],
            "attention_mask": stacked["attention_mask"][:train_stop],
            "source_effect": source_effect[:train_stop],
            "target_reference": target_reference[:train_stop],
            "probe_bank": _bank_payload(train_bank),
            "anchor_plan": _anchor_payload(anchor_plan, 0, train_stop, token_ids),
        },
        "validation": {
            "input_ids": stacked["input_ids"][train_stop:],
            "attention_mask": stacked["attention_mask"][train_stop:],
            "source_effect": source_effect[train_stop:],
            "target_reference": target_reference[train_stop:],
            "probe_bank": _bank_payload(validation_bank),
            "anchor_plan": _anchor_payload(
                anchor_plan, train_stop, len(all_texts), token_ids
            ),
        },
        "runtime_versions": runtime_versions(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"cache={cache_path}")
    return cache


def validate_cache(cache, cache_path, checkpoint, target_model, expected_sha256=None):
    if cache.get("type") != "rfa_k_sweep_cache" or cache.get("format_version") != 1:
        raise ValueError("Unsupported RFA K-sweep cache")
    actual_hash = file_sha256(cache_path)
    if expected_sha256 and actual_hash != expected_sha256:
        raise ValueError("Cache content hash does not match the orchestrator")
    if cache.get("source_checkpoint_sha256") != file_sha256(checkpoint.path):
        raise ValueError("Cache source checkpoint hash does not match")
    if cache.get("target_model") != target_model:
        raise ValueError("Cache target_model does not match --target-model")
    current_identity = _model_identity(target_model, cache.get("target_model_revision"))
    if cache.get("target_model_identity") != current_identity:
        raise ValueError("Current target model files/revision differ from the cache")
    if int(cache.get("basis_size", -1)) != checkpoint.sampled_ids.numel():
        raise ValueError("Cache basis size does not match source checkpoint")
    if not torch.equal(cache.get("sampled_ids"), checkpoint.sampled_ids):
        raise ValueError("Cache sampled token IDs do not match source checkpoint")
    for split_name in ("train", "validation"):
        split = cache.get(split_name)
        if not isinstance(split, dict):
            raise ValueError(f"Cache is missing {split_name} partition")
        count = split["input_ids"].size(0)
        if count < 1 or split["source_effect"].size(0) != count:
            raise ValueError(f"Cache {split_name} tensors have inconsistent rows")
        if split["target_reference"].shape != split["source_effect"].shape:
            raise ValueError(f"Cache {split_name} effects have inconsistent shapes")
    return actual_hash


def validate_cache_arguments(cache, args):
    """Prevent a resumed run from silently changing cache-defining settings."""
    expected = {
        "max_length": int(args.max_length),
        "seed": int(args.seed),
        "rfa_temperature": float(args.temperature),
        "anchor_mode": args.anchor_mode,
        "dtype": args.dtype,
        "requested_target_revision": args.target_revision,
    }
    for key, value in expected.items():
        cached = cache.get(key)
        if cached != value:
            raise ValueError(
                f"Resume argument mismatch for {key}: cache={cached!r}, CLI={value!r}"
            )
    if cache["train"]["input_ids"].size(0) != args.train_probes:
        raise ValueError("Resume --train-probes differs from the cache")
    if cache["validation"]["input_ids"].size(0) != args.validation_probes:
        raise ValueError("Resume --validation-probes differs from the cache")
    current_probe_request = _probe_request(args)
    if cache.get("probe_request") != current_probe_request:
        raise ValueError(
            "Resume probe source differs from the cache: "
            f"cache={cache.get('probe_request')!r}, CLI={current_probe_request!r}"
        )
    anchor = cache.get("anchor_config", {})
    anchor_expected = {
        "coverage": args.anchor_coverage,
        "min_tokens": args.min_anchor_tokens,
        "max_tokens": args.max_anchor_tokens,
    }
    for key, value in anchor_expected.items():
        if anchor.get(key) != value:
            raise ValueError(
                f"Resume anchor argument mismatch for {key}: "
                f"cache={anchor.get(key)!r}, CLI={value!r}"
            )


def assign_k_values(k_values, worker_count):
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    assignments = [[] for _ in range(worker_count)]
    for index, basis_k in enumerate(k_values):
        assignments[index % worker_count].append(int(basis_k))
    return [values for values in assignments if values]


def _json_write_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_k_outputs(k_dir, checkpoint_payload, result, history):
    k_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = k_dir / f"refined_target_k{int(result['basis_k']):03d}.pt"
    result_path = k_dir / "rfa_result.json"
    history_path = k_dir / "refinement_history.json"
    existing = [path for path in (checkpoint_path, result_path, history_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite K outputs: {existing}")
    torch.save(checkpoint_payload, checkpoint_path)
    _json_write_new(result_path, result)
    _json_write_new(history_path, history)
    return checkpoint_path


def _completed_k_is_valid(k_dir, basis_k, cache_sha256, refinement_config):
    result_path = k_dir / "rfa_result.json"
    checkpoint_path = k_dir / f"refined_target_k{basis_k:03d}.pt"
    history_path = k_dir / "refinement_history.json"
    if not (
        result_path.is_file() and checkpoint_path.is_file() and history_path.is_file()
    ):
        return False
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not (
        int(result.get("basis_k", -1)) == basis_k
        and result.get("cache_sha256") == cache_sha256
        and result.get("refinement_config") == refinement_config
    ):
        return False
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return (
        payload.get("type") == "rfa_refined_target_prompt_topk"
        and int(payload.get("basis_k", -1)) == basis_k
        and payload.get("cache_sha256") == cache_sha256
        and payload.get("refinement_config") == refinement_config
    )


def _refinement_config(args, cache):
    return {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "stay_weight": args.stay_weight,
        "rfa_temperature": cache["rfa_temperature"],
        "dtype": args.dtype,
        "seed": args.seed,
    }


def run_worker(args):
    set_seed(args.seed)
    cache_path = Path(args.cache_path).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    checkpoint = load_checkpoint(args.source_checkpoint)
    validate_cache(
        cache, cache_path, checkpoint, args.target_model, args.cache_sha256
    )
    if max(args.worker_k_values) > checkpoint.sampled_ids.numel():
        raise ValueError("A worker K exceeds the source basis size")
    target_model = load_model(
        args.target_model,
        args.dtype,
        args.device,
        None,
        cache.get("target_model_revision"),
    )
    actual_revision = getattr(target_model.config, "_commit_hash", None)
    expected_revision = cache.get("target_model_revision")
    if expected_revision and actual_revision and expected_revision != actual_revision:
        raise ValueError("Worker target model revision differs from cache")
    target_model.config.use_cache = False
    _enable_refinement_mode(target_model)
    output_root = Path(args.output_dir).resolve()
    refinement_config = _refinement_config(args, cache)
    for basis_k in args.worker_k_values:
        k_dir = output_root / f"k_{basis_k:03d}"
        if args.resume and _completed_k_is_valid(
            k_dir, basis_k, args.cache_sha256, refinement_config
        ):
            print(f"K={basis_k}: resume skip")
            continue
        if k_dir.exists() and any(k_dir.iterdir()):
            raise FileExistsError(f"K={basis_k}: output directory is not empty: {k_dir}")
        payload, result, history = refine_topk_from_cache(
            target_model,
            checkpoint,
            cache,
            basis_k,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.stay_weight,
            args.cache_sha256,
        )
        payload["refinement_config"] = refinement_config
        result["refinement_config"] = refinement_config
        payload["probe_banks"] = {
            "train": cache["train"]["probe_bank"],
            "validation": cache["validation"]["probe_bank"],
        }
        payload["anchors"] = {
            "mode": cache["anchor_mode"],
            **cache["anchor_config"],
        }
        checkpoint_path = _save_k_outputs(k_dir, payload, result, history)
        print(
            f"K={basis_k}: direct_val={result['direct_validation_rfa']['mean']:.8f} "
            f"refined_val={result['refined_validation_rfa']['mean']:.8f} "
            f"checkpoint={checkpoint_path}"
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del target_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reject_forbidden_keys(value, path="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(forbidden in lowered for forbidden in FORBIDDEN_SELECTION_KEYS):
                raise ValueError(f"Selection input contains forbidden key: {path}.{key}")
            _reject_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, f"{path}[{index}]")


def select_one_se(rows):
    """Select the smallest K within one SE of the lowest held-out mean RFA."""
    if not rows:
        raise ValueError("No K results to select from")
    _reject_forbidden_keys(rows)
    by_mean = min(
        rows,
        key=lambda row: (
            float(row["refined_validation_rfa"]["mean"]), int(row["basis_k"])
        ),
    )
    minimum = float(by_mean["refined_validation_rfa"]["mean"])
    minimum_se = float(by_mean["refined_validation_rfa"]["se"])
    if not math.isfinite(minimum) or not math.isfinite(minimum_se):
        raise ValueError("Held-out RFA mean/SE must be finite")
    threshold = minimum + minimum_se
    eligible = [
        row for row in rows
        if float(row["refined_validation_rfa"]["mean"]) <= threshold + 1e-15
    ]
    selected = min(eligible, key=lambda row: int(row["basis_k"]))
    return {
        "rule": "one_standard_error_smallest_k",
        "metric": "refined held-out mean RFA (lower is better)",
        "minimum_mean_k": int(by_mean["basis_k"]),
        "minimum_mean_rfa": minimum,
        "minimum_mean_se": minimum_se,
        "threshold": threshold,
        "selected_k": int(selected["basis_k"]),
        "selected_rfa": float(selected["refined_validation_rfa"]["mean"]),
    }


def _compact_report_rows(rows, output_dir, selected_k):
    report_rows = []
    for row in rows:
        basis_k = int(row["basis_k"])
        report_rows.append({
            "basis_k": basis_k,
            "optimized": bool(row["optimized"]),
            "direct_train_rfa": float(row["direct_train_rfa"]["mean"]),
            "direct_validation_rfa": float(row["direct_validation_rfa"]["mean"]),
            "refined_train_rfa": float(row["refined_train_rfa"]["mean"]),
            "refined_validation_rfa": float(row["refined_validation_rfa"]["mean"]),
            "refined_validation_se": float(row["refined_validation_rfa"]["se"]),
            "selected": basis_k == selected_k,
            "checkpoint": str(
                output_dir / f"k_{basis_k:03d}" / f"refined_target_k{basis_k:03d}.pt"
            ),
        })
    return report_rows


def collect_and_select(output_dir, k_values, cache_sha256):
    output_dir = Path(output_dir).resolve()
    paths = [output_dir / SWEEP_JSON, output_dir / SWEEP_CSV, output_dir / SELECTION_JSON]
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite selection reports: {existing}")
    rows = []
    for basis_k in sorted(k_values):
        result_path = output_dir / f"k_{basis_k:03d}" / "rfa_result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing completed result for K={basis_k}: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        _reject_forbidden_keys(result)
        if int(result.get("basis_k", -1)) != basis_k:
            raise ValueError(f"Result K mismatch at {result_path}")
        if result.get("cache_sha256") != cache_sha256:
            raise ValueError(f"Result cache hash mismatch at {result_path}")
        checkpoint_path = (
            output_dir / f"k_{basis_k:03d}" / f"refined_target_k{basis_k:03d}.pt"
        )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            payload.get("type") != "rfa_refined_target_prompt_topk"
            or int(payload.get("basis_k", -1)) != basis_k
            or payload.get("cache_sha256") != cache_sha256
        ):
            raise ValueError(f"Checkpoint metadata mismatch at {checkpoint_path}")
        rows.append(result)
    selection = select_one_se(rows)
    refinement_config = rows[0].get("refinement_config")
    if refinement_config is None or any(
        row.get("refinement_config") != refinement_config for row in rows
    ):
        raise ValueError("K results do not share one refinement configuration")
    selected_k = selection["selected_k"]
    report_rows = _compact_report_rows(rows, output_dir, selected_k)
    sweep_payload = {
        "type": "rfa_k_sweep_report",
        "format_version": 1,
        "cache_sha256": cache_sha256,
        "refinement_config": refinement_config,
        "rows": report_rows,
        "selection": selection,
        "note": "Generated exclusively from held-out unlabeled RFA statistics.",
    }
    _json_write_new(output_dir / SWEEP_JSON, sweep_payload)
    _json_write_new(output_dir / SELECTION_JSON, selection)
    with (output_dir / SWEEP_CSV).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)
    return selection, report_rows


def _worker_command(args, cache_path, cache_sha256, k_values, worker_device):
    command = [
        sys.executable,
        "-m",
        _subprocess_module_name(),
        "--source-checkpoint", str(Path(args.source_checkpoint).resolve()),
        "--output-dir", str(Path(args.output_dir).resolve()),
        "--target-model", args.target_model,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--temperature", str(args.temperature),
        "--stay-weight", str(args.stay_weight),
        "--seed", str(args.seed),
        "--dtype", args.dtype,
        "--device", worker_device,
        "--internal-worker",
        "--cache-path", str(cache_path),
        "--cache-sha256", cache_sha256,
        "--worker-k-values", *[str(value) for value in k_values],
    ]
    if args.target_revision:
        command.extend(["--target-revision", args.target_revision])
    if args.resume:
        command.append("--resume")
    return command


def _run_workers(args, cache_path, cache_sha256):
    gpu_ids = list(args.gpus or [])
    worker_count = len(gpu_ids) if gpu_ids else 1
    assignments = assign_k_values(args.k_values, worker_count)
    processes = []
    for index, k_values in enumerate(assignments):
        worker_device = f"cuda:{gpu_ids[index]}" if gpu_ids else args.device
        command = _worker_command(
            args, cache_path, cache_sha256, k_values, worker_device
        )
        print(f"worker={index} K={k_values} device={worker_device}")
        processes.append((index, k_values, subprocess.Popen(command)))
    failures = []
    for index, k_values, process in processes:
        code = process.wait()
        if code != 0:
            failures.append((index, k_values, code))
    if failures:
        raise RuntimeError(f"K-sweep worker failures: {failures}")


def _cache_worker_command(args, cache_path):
    command = [
        sys.executable,
        "-m",
        _subprocess_module_name(),
        "--source-checkpoint", str(Path(args.source_checkpoint).resolve()),
        "--output-dir", str(Path(args.output_dir).resolve()),
        "--target-model", args.target_model,
        "--dataset", args.dataset,
        "--text-field", args.text_field,
        "--probe-split", args.probe_split,
        "--train-probes", str(args.train_probes),
        "--validation-probes", str(args.validation_probes),
        "--max-length", str(args.max_length),
        "--k-values", *[str(value) for value in args.k_values],
        "--batch-size", str(args.batch_size),
        "--temperature", str(args.temperature),
        "--anchor-mode", args.anchor_mode,
        "--anchor-coverage", str(args.anchor_coverage),
        "--min-anchor-tokens", str(args.min_anchor_tokens),
        "--max-anchor-tokens", str(args.max_anchor_tokens),
        "--seed", str(args.seed),
        "--dtype", args.dtype,
        "--device", _resolve_cache_device(args),
        "--internal-cache-worker",
        "--cache-path", str(cache_path),
    ]
    if args.probe_file:
        command.extend(["--probe-file", args.probe_file])
    if args.target_revision:
        command.extend(["--target-revision", args.target_revision])
    return command


def run(args):
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / CACHE_NAME
    if cache_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"Cache already exists; pass --resume to reuse it: {cache_path}"
            )
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        checkpoint = load_checkpoint(args.source_checkpoint)
        cache_sha256 = validate_cache(
            cache, cache_path, checkpoint, args.target_model
        )
        validate_cache_arguments(cache, args)
    else:
        command = _cache_worker_command(args, cache_path)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Cache preparation subprocess failed with code {completed.returncode}"
            )
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        checkpoint = load_checkpoint(args.source_checkpoint)
        cache_sha256 = validate_cache(
            cache, cache_path, checkpoint, args.target_model
        )
        validate_cache_arguments(cache, args)
    print(f"cache_sha256={cache_sha256}")
    if args.cache_only:
        return {"cache": str(cache_path), "cache_sha256": cache_sha256}
    final_paths = [
        output_dir / SWEEP_JSON,
        output_dir / SWEEP_CSV,
        output_dir / SELECTION_JSON,
    ]
    if args.resume and all(path.is_file() for path in final_paths):
        report = json.loads((output_dir / SWEEP_JSON).read_text(encoding="utf-8"))
        selection = json.loads((output_dir / SELECTION_JSON).read_text(encoding="utf-8"))
        recorded_k = sorted(int(row["basis_k"]) for row in report.get("rows", []))
        if report.get("cache_sha256") != cache_sha256:
            raise ValueError("Existing final report cache hash mismatch")
        if recorded_k != sorted(args.k_values):
            raise ValueError("Existing final report K values differ from --k-values")
        expected_config = _refinement_config(args, cache)
        if report.get("refinement_config") != expected_config:
            raise ValueError("Existing final report refinement settings differ from CLI")
        if report.get("selection") != selection:
            raise ValueError("Existing selection file disagrees with the sweep report")
        raw_results = []
        for basis_k in sorted(args.k_values):
            k_dir = output_dir / f"k_{basis_k:03d}"
            if not _completed_k_is_valid(
                k_dir, basis_k, cache_sha256, expected_config
            ):
                raise ValueError(f"Completed sweep has an invalid K={basis_k} artifact set")
            result = json.loads((k_dir / "rfa_result.json").read_text(encoding="utf-8"))
            _reject_forbidden_keys(result)
            raw_results.append(result)
        recomputed = select_one_se(raw_results)
        if recomputed != selection:
            raise ValueError("Existing selection does not match recomputed one-SE result")
        expected_rows = _compact_report_rows(
            raw_results, output_dir, selection["selected_k"]
        )
        if report.get("rows") != expected_rows:
            raise ValueError("Existing compact sweep rows differ from raw K results")
        print("completed sweep found; resume returns the existing immutable reports")
        return {"selection": selection, "rows": report["rows"]}
    if any(path.exists() for path in final_paths):
        raise FileExistsError(
            "Final sweep reports are incomplete or --resume was not supplied; "
            f"refusing to overwrite: {[str(path) for path in final_paths if path.exists()]}"
        )
    _run_workers(args, cache_path, cache_sha256)
    selection, rows = collect_and_select(
        output_dir, args.k_values, cache_sha256
    )
    print(json.dumps(selection, indent=2))
    return {"selection": selection, "rows": rows}


def main(argv=None):
    args = parse_args(argv)
    if args.internal_cache_worker:
        set_seed(args.seed)
        prepare_cache(args, args.cache_path)
    elif args.internal_worker:
        run_worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
