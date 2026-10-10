#!/usr/bin/env python3
"""Shared RFA math plus archived selection helpers.

Each checkpoint contains one source SuperPos prompt.  The same token
coordinates are evaluated on the source embedding table and reconstructed on
the target embedding table, so users do not need to export target tensors by
hand.  For a probe context x and prompt P, the functional effect is

    delta(v) = log p(v | P, x) - log p(v | x).

RFA applies a temperature softmax to delta and averages the Jensen-Shannon
divergence between source and target effect distributions.  Lower is better.
No labels, gradients, accuracy values, or prompt updates are used.
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class ProbeBank:
    texts: List[str]
    indices: List[int]
    source: str
    fingerprint: str


@dataclass
class AlignedVocabulary:
    token_ids: torch.Tensor
    tokenizer_size: int
    excluded_special_ids: List[int]


@dataclass
class SourceCheckpoint:
    checkpoint_id: str
    path: Path
    payload: Mapping[str, Any]
    weights: torch.Tensor
    sampled_ids: torch.Tensor
    weight_mode: str
    activation_temperature: Optional[float]
    seed: Optional[int]


@dataclass
class Candidate:
    candidate_id: str
    source: SourceCheckpoint
    bridge_k: int


@dataclass
class AnchorPlan:
    indices: List[torch.Tensor]
    selected_counts: List[int]
    achieved_coverages: List[float]
    aggregate_source_js: List[float]


class ProbeDataset(Dataset):
    """Tokenized probe text with no representation for labels."""

    def __init__(self, texts, tokenizer, max_length):
        self.rows = []
        for text in texts:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.rows.append(
                {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _validate_superpos_tensors(payload):
    if not isinstance(payload, Mapping):
        raise ValueError("SuperPos checkpoint must be a mapping")
    if "prompt_weights" not in payload or "sampled_ids" not in payload:
        raise ValueError("Checkpoint needs prompt_weights and sampled_ids")
    raw_weights = payload["prompt_weights"]
    sampled_ids = payload["sampled_ids"]
    if not torch.is_tensor(raw_weights) or not torch.is_tensor(sampled_ids):
        raise ValueError("prompt_weights and sampled_ids must be tensors")
    raw_weights = raw_weights.detach().float().cpu()
    sampled_ids = sampled_ids.detach().long().flatten().cpu()
    if raw_weights.ndim != 2 or sampled_ids.ndim != 1:
        raise ValueError("prompt_weights must be 2D and sampled_ids must be 1D")
    if raw_weights.size(1) != sampled_ids.numel() or sampled_ids.numel() == 0:
        raise ValueError("Checkpoint basis size does not match sampled_ids")
    if sampled_ids.min() < 0:
        raise ValueError("Checkpoint contains negative sampled token IDs")
    if not torch.isfinite(raw_weights).all():
        raise ValueError("Checkpoint contains non-finite prompt weights")
    return raw_weights, sampled_ids


def activate_weights(payload, legacy_temperature=None):
    """Reconstruct saved SuperPos coordinates without changing semantics."""
    raw_weights, sampled_ids = _validate_superpos_tensors(payload)
    mode = str(payload.get("weight_mode", "softmax"))
    if mode == "softmax":
        temperature = payload.get("temperature", legacy_temperature)
        if temperature is None:
            raise ValueError(
                "Softmax checkpoint has no temperature metadata; pass "
                "--legacy-temperature explicitly if the old run used a known value"
            )
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("SuperPos temperature must be finite and positive")
        return (
            F.softmax(raw_weights / temperature, dim=-1),
            sampled_ids,
            mode,
            temperature,
        )
    if mode == "raw":
        return raw_weights, sampled_ids, mode, None
    if mode == "l1_normalized_raw":
        epsilon = float(payload.get("normalization_epsilon", 1e-12))
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("normalization_epsilon must be finite and positive")
        denominator = raw_weights.abs().sum(dim=-1, keepdim=True).clamp_min(epsilon)
        return raw_weights / denominator, sampled_ids, mode, None
    raise ValueError(f"Unsupported weight_mode={mode!r}")


def project_prompt(weights, basis_embeddings, bridge_k, weight_mode):
    """Apply the existing direct token-coordinate migration rule."""
    if weights.ndim != 2 or basis_embeddings.ndim != 2:
        raise ValueError("weights and basis_embeddings must be 2D")
    if weights.size(1) != basis_embeddings.size(0):
        raise ValueError("Weight basis does not match embedding basis")
    basis_size = weights.size(1)
    if bridge_k < 1 or bridge_k > basis_size:
        raise ValueError(f"bridge_k must be in [1, {basis_size}]")
    if bridge_k == basis_size:
        return weights @ basis_embeddings

    ranking = weights if weight_mode == "softmax" else weights.abs()
    indices = ranking.topk(bridge_k, dim=1).indices
    selected_weights = weights.gather(1, indices)
    if weight_mode == "softmax":
        selected_weights = selected_weights / selected_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
    elif weight_mode == "l1_normalized_raw":
        selected_weights = selected_weights / selected_weights.abs().sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
    selected_embeddings = basis_embeddings[indices]
    return (selected_weights.unsqueeze(-1) * selected_embeddings).sum(dim=1)


def discover_source_checkpoints(directory, pattern, recursive, legacy_temperature=None):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")
    paths = root.rglob(pattern) if recursive else root.glob(pattern)
    paths = sorted(path.resolve() for path in paths if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No checkpoints matching {pattern!r} in {root}")
    checkpoints = []
    for path in paths:
        payload = _load_checkpoint(path)
        weights, sampled_ids, mode, temperature = activate_weights(
            payload, legacy_temperature=legacy_temperature
        )
        checkpoint_id = path.relative_to(root).as_posix()
        checkpoints.append(
            SourceCheckpoint(
                checkpoint_id=checkpoint_id,
                path=path,
                payload=payload,
                weights=weights,
                sampled_ids=sampled_ids,
                weight_mode=mode,
                activation_temperature=temperature,
                seed=payload.get("seed"),
            )
        )
    return checkpoints


def expand_candidates(checkpoints, requested_k):
    candidates = []
    for checkpoint in checkpoints:
        basis_size = checkpoint.sampled_ids.numel()
        k_values = [basis_size if value == 0 else value for value in requested_k]
        for bridge_k in sorted(set(k_values)):
            if bridge_k < 1 or bridge_k > basis_size:
                raise ValueError(
                    f"{checkpoint.checkpoint_id}: bridge_k={bridge_k} is outside "
                    f"[1, {basis_size}]"
                )
            label = "all" if bridge_k == basis_size else str(bridge_k)
            candidates.append(
                Candidate(
                    candidate_id=f"{checkpoint.checkpoint_id}::k={label}",
                    source=checkpoint,
                    bridge_k=bridge_k,
                )
            )
    return candidates


def validate_tokenizer_alignment(source_tokenizer, target_tokenizer):
    source_size = len(source_tokenizer)
    target_size = len(target_tokenizer)
    if source_size != target_size:
        raise ValueError(
            "RFA requires equal tokenizer sizes, got "
            f"{source_size} and {target_size}"
        )
    for start in range(0, source_size, 4096):
        token_ids = list(range(start, min(start + 4096, source_size)))
        source_tokens = source_tokenizer.convert_ids_to_tokens(token_ids)
        target_tokens = target_tokenizer.convert_ids_to_tokens(token_ids)
        if source_tokens != target_tokens:
            mismatches = [
                (token_id, source_token, target_token)
                for token_id, source_token, target_token in zip(
                    token_ids, source_tokens, target_tokens
                )
                if source_token != target_token
            ]
            raise ValueError(
                "RFA requires identical token semantics at every ID; examples: "
                f"{mismatches[:5]}"
            )
    special_ids = {
        int(token_id)
        for tokenizer in (source_tokenizer, target_tokenizer)
        for token_id in (getattr(tokenizer, "all_special_ids", None) or [])
        if token_id is not None and 0 <= int(token_id) < source_size
    }
    token_ids = torch.tensor(
        [token_id for token_id in range(source_size) if token_id not in special_ids],
        dtype=torch.long,
    )
    if token_ids.numel() == 0:
        raise ValueError("No aligned non-special vocabulary IDs remain")
    return AlignedVocabulary(token_ids, source_size, sorted(special_ids))


def validate_sampled_ids(sampled_ids, source_tokenizer, target_tokenizer):
    mismatches = []
    for token_id in sampled_ids.tolist():
        if token_id < 0 or token_id >= len(source_tokenizer) or token_id >= len(target_tokenizer):
            mismatches.append((token_id, "out_of_range"))
            continue
        source_token = source_tokenizer.convert_ids_to_tokens(token_id)
        target_token = target_tokenizer.convert_ids_to_tokens(token_id)
        if source_token != target_token:
            mismatches.append((token_id, source_token, target_token))
    if mismatches:
        raise ValueError(f"SuperPos sampled IDs are not aligned: {mismatches[:5]}")


def validate_probe_encoding_alignment(texts, source_tokenizer, target_tokenizer, max_length):
    """Verify the same probe text produces identical IDs and masks in both tokenizers."""
    for index, text in enumerate(texts):
        kwargs = dict(truncation=True, max_length=max_length, padding="max_length",
                      return_tensors="pt")
        source = source_tokenizer(text, **kwargs)
        target = target_tokenizer(text, **kwargs)
        for field in ("input_ids", "attention_mask"):
            if not torch.equal(source[field], target[field]):
                raise ValueError(
                    f"Probe {index} has different {field} across source/target tokenizers"
                )


def _fingerprint_texts(texts):
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _select_probe_rows(texts, num_probes, seed, source):
    if num_probes < 1:
        raise ValueError("num_probes must be positive")
    if num_probes > len(texts):
        raise ValueError(f"Requested {num_probes} probes but {source} has {len(texts)}")
    indices = sorted(random.Random(seed).sample(range(len(texts)), num_probes))
    selected = [str(texts[index]) for index in indices]
    return ProbeBank(selected, indices, source, _fingerprint_texts(selected))


def _read_probe_file(path, text_field):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Probe file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".txt":
        texts = [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8").splitlines()]
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        texts = [row if isinstance(row, str) else row[text_field] for row in rows]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Probe JSON must contain a list")
        texts = [row if isinstance(row, str) else row[text_field] for row in payload]
    else:
        raise ValueError("Probe file must be TXT, JSON, or JSONL")
    texts = [str(text) for text in texts if str(text).strip()]
    if not texts:
        raise ValueError("Probe file contains no non-empty texts")
    return texts, str(path)


def load_sst2_cached_first(split=None):
    """Load SST-2 from the local HF cache before attempting any network call.

    Hugging Face's normal cache reuse can still query the Hub for metadata and
    wait for a network timeout.  `local_files_only` makes the common offline
    path fail fast; only a genuine cache miss falls back to the normal loader.
    """
    from datasets import DownloadConfig, load_dataset

    kwargs = {"split": split} if split is not None else {}
    try:
        return load_dataset(
            "glue",
            "sst2",
            download_config=DownloadConfig(local_files_only=True, max_retries=0),
            **kwargs,
        )
    except Exception as cache_error:
        try:
            return load_dataset("glue", "sst2", **kwargs)
        except Exception as network_error:
            raise network_error from cache_error


def load_probe_bank(args):
    """Load only text; target labels are never indexed or materialized."""
    if args.probe_file:
        texts, source = _read_probe_file(args.probe_file, args.text_field)
    else:
        if args.dataset != "sst2":
            raise ValueError("Only sst2 is built in; use --probe-file for other tasks")
        split = load_sst2_cached_first(args.probe_split)
        text_column = "sentence"
        texts = [str(value) for value in split[text_column]]
        source = f"datasets:glue/sst2:{args.probe_split}:{text_column}"
    return _select_probe_rows(texts, args.num_probes, args.seed, source)


def load_tokenizer(model_name, revision=None):
    from transformers import AutoTokenizer

    kwargs = {"revision": revision} if revision else {}
    return configure_tokenizer(AutoTokenizer.from_pretrained(model_name, **kwargs))


def configure_tokenizer(tokenizer):
    """Use one padding contract in source training, refinement, and evaluation."""
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer needs either pad_token or eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def position_ids_from_mask(attention_mask):
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


def last_valid_indices(attention_mask):
    positions = torch.arange(attention_mask.size(1), device=attention_mask.device)
    masked = positions.unsqueeze(0).expand_as(attention_mask).masked_fill(
        attention_mask == 0, -1
    )
    result = masked.max(dim=-1).values
    if (result < 0).any():
        raise ValueError("Every input must contain at least one attended token")
    return result


def _dtype_from_name(name):
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_model(model_name, dtype, device, device_map, revision=None):
    from transformers import AutoModelForCausalLM

    kwargs = {"torch_dtype": _dtype_from_name(dtype)}
    if revision:
        kwargs["revision"] = revision
    if device_map:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not device_map:
        model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def release_model(model):
    import gc

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _supports_logits_to_keep(model):
    import inspect

    try:
        return "logits_to_keep" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def forward_next_token_logits(model, input_ids, attention_mask, prompt):
    word_embeddings = model.get_input_embeddings()(input_ids)
    if prompt is None:
        inputs_embeds = word_embeddings
        extended_mask = attention_mask
    else:
        prompt_batch = prompt.to(
            device=word_embeddings.device, dtype=word_embeddings.dtype
        ).unsqueeze(0).expand(input_ids.size(0), -1, -1)
        inputs_embeds = torch.cat([prompt_batch, word_embeddings], dim=1)
        prompt_mask = torch.ones(
            input_ids.size(0),
            prompt.size(0),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)
    base_model = getattr(model, "model", None)
    output_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if base_model is not None and output_head is not None:
        outputs = base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        indices = last_valid_indices(extended_mask).to(hidden.device)
        final_hidden = hidden[torch.arange(hidden.size(0), device=hidden.device), indices]
        return output_head(final_hidden).float()

    # Compatibility fallback for test doubles/non-Qwen models. It may allocate
    # full sequence logits, but still gathers the true final attended position.
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=extended_mask,
                    use_cache=False)
    indices = last_valid_indices(extended_mask).to(outputs.logits.device)
    return outputs.logits[
        torch.arange(outputs.logits.size(0), device=outputs.logits.device), indices
    ].float()


@torch.inference_mode()
def collect_logits(model, dataloader, prompt, token_ids, description):
    from tqdm import tqdm

    rows = []
    input_device = model.get_input_embeddings().weight.device
    for batch in tqdm(dataloader, desc=description, leave=False):
        if set(batch) != {"input_ids", "attention_mask"}:
            raise ValueError("Probe batches may not contain labels or extra fields")
        input_ids = batch["input_ids"].to(input_device)
        attention_mask = batch["attention_mask"].to(input_device)
        logits = forward_next_token_logits(model, input_ids, attention_mask, prompt)
        if logits.size(1) <= int(token_ids.max()):
            raise ValueError("Model LM head does not cover every aligned token ID")
        rows.append(logits.index_select(1, token_ids.to(logits.device)).cpu())
    if not rows:
        raise ValueError("Probe dataloader produced no rows")
    return torch.cat(rows, dim=0)


def build_prompt(checkpoint, embedding_weight, bridge_k):
    sampled_ids = checkpoint.sampled_ids
    if sampled_ids.max() >= embedding_weight.size(0):
        raise ValueError(f"{checkpoint.checkpoint_id}: sampled ID exceeds embedding rows")
    basis = embedding_weight.index_select(
        0, sampled_ids.to(embedding_weight.device)
    ).float()
    return project_prompt(
        checkpoint.weights.to(basis.device),
        basis,
        bridge_k,
        checkpoint.weight_mode,
    ).detach()


def reference_prompt(reference, prompt):
    if reference == "no-prompt":
        return None
    if reference == "zero":
        return torch.zeros_like(prompt)
    raise ValueError(f"Unsupported reference={reference!r}")


def tokenwise_source_js(prompted_logits, reference_logits, temperature=1.0):
    if prompted_logits.shape != reference_logits.shape or prompted_logits.ndim != 2:
        raise ValueError("Source logits must have equal [probe, vocabulary] shape")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Anchor probability temperature must be positive")
    first_log = F.log_softmax(prompted_logits.float() / temperature, dim=-1)
    second_log = F.log_softmax(reference_logits.float() / temperature, dim=-1)
    log_mixture = torch.logaddexp(first_log, second_log) - math.log(2.0)
    contribution = 0.5 * (
        first_log.exp() * (first_log - log_mixture)
        + second_log.exp() * (second_log - log_mixture)
    )
    return contribution.clamp_min(0.0)


def build_anchor_plan(
    aggregate_contributions,
    coverage,
    min_tokens,
    max_tokens,
):
    if aggregate_contributions.ndim != 2 or aggregate_contributions.size(1) == 0:
        raise ValueError("Aggregate contributions must be [probe, vocabulary]")
    if not 0 < coverage <= 1:
        raise ValueError("Anchor coverage must be in (0, 1]")
    vocabulary_size = aggregate_contributions.size(1)
    if min_tokens < 1 or max_tokens < min_tokens:
        raise ValueError("Anchor bounds must satisfy 1 <= min <= max")
    min_tokens = min(min_tokens, vocabulary_size)
    max_tokens = min(max_tokens, vocabulary_size)
    min_tokens = min(min_tokens, max_tokens)
    indices = []
    counts = []
    achieved = []
    totals = []
    for row in aggregate_contributions:
        order = torch.argsort(row, descending=True, stable=True)
        ordered = row[order]
        total = ordered.sum()
        if total.item() > 0:
            threshold = coverage * total
            required = int(torch.searchsorted(ordered.cumsum(0), threshold).item()) + 1
        else:
            required = min_tokens
        count = max(min_tokens, min(max_tokens, required))
        selected = order[:count].sort().values.cpu()
        covered = row[selected].sum()
        indices.append(selected)
        counts.append(count)
        achieved.append(float((covered / total).item()) if total.item() > 0 else 0.0)
        totals.append(float(total.item()))
    return AnchorPlan(indices, counts, achieved, totals)


def _js_from_probabilities(first, second):
    mixture = 0.5 * (first + second)

    def kl_term(probability):
        positive = probability > 0
        terms = torch.zeros_like(probability)
        terms[positive] = probability[positive] * (
            probability[positive].log() - mixture[positive].log()
        )
        return terms.sum()

    return 0.5 * (kl_term(first) + kl_term(second))


def _collapse_with_other(probability, selected):
    selected = selected.to(probability.device)
    anchor_mass = probability[selected]
    other_mass = (1.0 - anchor_mass.sum()).clamp_min(0.0).unsqueeze(0)
    collapsed = torch.cat([anchor_mass, other_mass])
    return collapsed / collapsed.sum().clamp_min(torch.finfo(collapsed.dtype).tiny)


def rowwise_rfa(source_effects, target_effects, temperature, anchor_plan=None):
    if source_effects.ndim != 2 or source_effects.shape != target_effects.shape:
        raise ValueError("Effects must have equal [probe, vocabulary] shape")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("RFA temperature must be finite and positive")
    if anchor_plan is not None and len(anchor_plan.indices) != source_effects.size(0):
        raise ValueError("Anchor plan must have one set per probe")
    values = []
    for index, (source_row, target_row) in enumerate(zip(source_effects, target_effects)):
        source_q = F.softmax(source_row.double() / temperature, dim=-1)
        target_q = F.softmax(target_row.double() / temperature, dim=-1)
        if anchor_plan is not None:
            selected = anchor_plan.indices[index]
            source_q = _collapse_with_other(source_q, selected)
            target_q = _collapse_with_other(target_q, selected)
        values.append(_js_from_probabilities(source_q, target_q))
    return torch.stack(values)


def coordinate_kl(target_coordinates, source_coordinates, epsilon=1e-12):
    """Mean KL(alpha_target || alpha_source) over prompt positions."""
    if target_coordinates.shape != source_coordinates.shape:
        raise ValueError("Source and target coordinates must have equal shape")
    if target_coordinates.ndim != 2:
        raise ValueError("Coordinates must have [prompt_length, basis_size] shape")
    target = target_coordinates.float().clamp_min(epsilon)
    source = source_coordinates.to(target.device).float().clamp_min(epsilon)
    target = target / target.sum(dim=-1, keepdim=True)
    source = source / source.sum(dim=-1, keepdim=True)
    return (target * (target.log() - source.log())).sum(dim=-1).mean()


def _summary(values):
    return {
        "rfa_score": float(values.mean()),
        "rfa_median": float(values.median()),
        "rfa_std": float(values.std(unbiased=False)),
        "rfa_min": float(values.min()),
        "rfa_max": float(values.max()),
    }


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _anchor_summary(args, plan):
    if plan is None:
        return {
            "mode": "full",
            "scope": "all aligned non-special tokenizer IDs",
            "other_bucket": False,
        }
    return {
        "mode": "source-js",
        "scope": "mean source-only JS contribution across all source checkpoints",
        "fixed_before_target_scoring": True,
        "shared_by_all_candidates": True,
        "coverage": args.anchor_coverage,
        "probability_temperature": args.anchor_probability_temperature,
        "min_tokens": args.min_anchor_tokens,
        "max_tokens": args.max_anchor_tokens,
        "selected_tokens": {
            "min": min(plan.selected_counts),
            "mean": sum(plan.selected_counts) / len(plan.selected_counts),
            "max": max(plan.selected_counts),
        },
        "achieved_coverage": {
            "min": min(plan.achieved_coverages),
            "mean": sum(plan.achieved_coverages) / len(plan.achieved_coverages),
            "max": max(plan.achieved_coverages),
        },
        "other_bucket": True,
    }


def save_reports(args, rows, probes, vocabulary, anchor_plan):
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    json_path = output_dir / f"rfa_selection_{stamp}.json"
    csv_path = output_dir / f"rfa_selection_{stamp}.csv"
    columns = [
        "rank", "selected", "candidate", "checkpoint", "seed", "bridge_k",
        "basis_size", "weight_mode", "activation_temperature", "prompt_length",
        "rfa_score", "rfa_median", "rfa_std", "rfa_min", "rfa_max",
    ]
    with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in columns} for row in rows])
    payload = {
        "method": {
            "name": "relative_functional_alignment",
            "delta": "log p(v|P,x) - log p(v|x)",
            "effect_distribution": "softmax(delta / temperature)",
            "divergence": "mean Jensen-Shannon divergence; lower is better",
            "training_free": True,
            "labels_read": False,
            "anchors": _anchor_summary(args, anchor_plan),
        },
        "settings": _json_safe(vars(args)),
        "probe_bank": {
            "source": probes.source,
            "indices": probes.indices,
            "num_probes": len(probes.texts),
            "sha256": probes.fingerprint,
            "labels_read": False,
        },
        "vocabulary": {
            "tokenizer_size": vocabulary.tokenizer_size,
            "aligned_non_special_size": int(vocabulary.token_ids.numel()),
            "excluded_special_ids": vocabulary.excluded_special_ids,
            "full_id_alignment_verified": True,
        },
        "selected_candidate": rows[0]["candidate"],
        "ranking": rows,
    }
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(_json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
    return json_path, csv_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--checkpoint-glob", default="*.pt")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--source-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--target-model", default="Qwen/Qwen2.5-7B")
    parser.add_argument(
        "--bridge-k", type=int, nargs="+", default=[0],
        help="Target Top-K values; 0 means the full checkpoint basis",
    )
    parser.add_argument(
        "--legacy-temperature", type=float, default=None,
        help="Explicit fallback only for old softmax checkpoints missing temperature",
    )
    parser.add_argument("--dataset", default="sst2")
    parser.add_argument("--probe-split", default="validation")
    parser.add_argument("--probe-file", default=None)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--num-probes", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--anchor-mode", choices=["full", "source-js"], default="full")
    parser.add_argument("--anchor-coverage", type=float, default=0.95)
    parser.add_argument("--min-anchor-tokens", type=int, default=512)
    parser.add_argument("--max-anchor-tokens", type=int, default=8192)
    parser.add_argument("--anchor-probability-temperature", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--reference", choices=["no-prompt", "zero"], default="no-prompt")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dtype", choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/rfa")
    args = parser.parse_args(argv)
    if args.num_probes < 1 or args.max_length < 1 or args.batch_size < 1:
        parser.error("num-probes, max-length, and batch-size must be positive")
    if not args.bridge_k or any(value < 0 for value in args.bridge_k):
        parser.error("--bridge-k values must be non-negative (0 means all)")
    for name in ("temperature", "anchor_probability_temperature"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.legacy_temperature is not None and (
        not math.isfinite(args.legacy_temperature) or args.legacy_temperature <= 0
    ):
        parser.error("--legacy-temperature must be finite and positive")
    if not 0 < args.anchor_coverage <= 1:
        parser.error("--anchor-coverage must be in (0, 1]")
    if args.min_anchor_tokens < 1 or args.max_anchor_tokens < args.min_anchor_tokens:
        parser.error("anchor bounds must satisfy 1 <= min <= max")
    return args


def run(args):
    set_seed(args.seed)
    checkpoints = discover_source_checkpoints(
        args.checkpoints_dir,
        args.checkpoint_glob,
        args.recursive,
        legacy_temperature=args.legacy_temperature,
    )
    prompt_lengths = {checkpoint.weights.size(0) for checkpoint in checkpoints}
    if len(prompt_lengths) != 1:
        raise ValueError(
            "All checkpoints must use the same prompt length for comparable RFA; "
            f"found {sorted(prompt_lengths)}"
        )
    candidates = expand_candidates(checkpoints, args.bridge_k)
    source_tokenizer = load_tokenizer(args.source_model)
    target_tokenizer = load_tokenizer(args.target_model)
    vocabulary = validate_tokenizer_alignment(source_tokenizer, target_tokenizer)
    for checkpoint in checkpoints:
        validate_sampled_ids(checkpoint.sampled_ids, source_tokenizer, target_tokenizer)
        recorded_model = checkpoint.payload.get("source_model")
        if recorded_model and recorded_model != args.source_model:
            raise ValueError(
                f"{checkpoint.checkpoint_id} records source_model={recorded_model!r}, "
                f"but CLI requested {args.source_model!r}"
            )
    for tokenizer in (source_tokenizer, target_tokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    probes = load_probe_bank(args)
    dataset = ProbeDataset(probes.texts, source_tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    token_ids = vocabulary.token_ids
    print(
        f"checkpoints={len(checkpoints)} candidates={len(candidates)} "
        f"probes={len(probes.texts)} aligned_vocab={token_ids.numel()}"
    )

    # Pass 1: source signatures.  One signature is reused by every K generated
    # from the same checkpoint.  Target candidates are not inspected here.
    source_model = load_model(args.source_model, args.dtype, device, args.device_map)
    source_embedding = source_model.get_input_embeddings().weight
    first_prompt = build_prompt(checkpoints[0], source_embedding, checkpoints[0].sampled_ids.numel())
    source_reference = collect_logits(
        source_model,
        dataloader,
        reference_prompt(args.reference, first_prompt),
        token_ids,
        "Source reference",
    )
    source_effects: Dict[str, torch.Tensor] = {}
    contribution_sum = None
    for checkpoint in checkpoints:
        source_prompt = build_prompt(
            checkpoint, source_embedding, checkpoint.sampled_ids.numel()
        )
        prompted = collect_logits(
            source_model,
            dataloader,
            source_prompt,
            token_ids,
            f"Source: {checkpoint.checkpoint_id}",
        )
        source_effects[checkpoint.checkpoint_id] = prompted - source_reference
        if args.anchor_mode == "source-js":
            contribution = tokenwise_source_js(
                prompted,
                source_reference,
                args.anchor_probability_temperature,
            )
            contribution_sum = contribution if contribution_sum is None else contribution_sum + contribution
        del source_prompt, prompted
    del source_reference, source_embedding, first_prompt
    release_model(source_model)

    anchor_plan = None
    if args.anchor_mode == "source-js":
        aggregate = contribution_sum / len(checkpoints)
        anchor_plan = build_anchor_plan(
            aggregate,
            args.anchor_coverage,
            args.min_anchor_tokens,
            args.max_anchor_tokens,
        )
        del aggregate, contribution_sum
        print(
            "fixed source-only anchors: "
            f"mean_count={sum(anchor_plan.selected_counts) / len(anchor_plan.selected_counts):.1f} "
            f"mean_coverage={sum(anchor_plan.achieved_coverages) / len(anchor_plan.achieved_coverages):.4f}"
        )

    # Pass 2: one target model, sequential candidates, no gradients.
    target_model = load_model(args.target_model, args.dtype, device, args.device_map)
    target_embedding = target_model.get_input_embeddings().weight
    target_zero = torch.zeros(
        checkpoints[0].weights.size(0),
        target_embedding.size(1),
        dtype=torch.float32,
        device=target_embedding.device,
    )
    target_reference = collect_logits(
        target_model,
        dataloader,
        reference_prompt(args.reference, target_zero),
        token_ids,
        "Target reference",
    )
    rows = []
    for candidate in candidates:
        prompt = build_prompt(candidate.source, target_embedding, candidate.bridge_k)
        prompted = collect_logits(
            target_model,
            dataloader,
            prompt,
            token_ids,
            f"Target: {candidate.candidate_id}",
        )
        target_effect = prompted - target_reference
        divergences = rowwise_rfa(
            source_effects[candidate.source.checkpoint_id],
            target_effect,
            args.temperature,
            anchor_plan,
        )
        row = {
            "candidate": candidate.candidate_id,
            "checkpoint": str(candidate.source.path),
            "seed": candidate.source.seed,
            "bridge_k": candidate.bridge_k,
            "basis_size": candidate.source.sampled_ids.numel(),
            "weight_mode": candidate.source.weight_mode,
            "activation_temperature": candidate.source.activation_temperature,
            "prompt_length": prompt.size(0),
        }
        row.update(_summary(divergences))
        rows.append(row)
        del prompt, prompted, target_effect, divergences
    del target_reference, target_embedding, target_zero
    release_model(target_model)

    rows.sort(key=lambda row: (row["rfa_score"], row["candidate"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row["selected"] = rank == 1
    json_path, csv_path = save_reports(args, rows, probes, vocabulary, anchor_plan)
    print("\nRFA ranking (lower is better):")
    for row in rows:
        marker = " *" if row["selected"] else ""
        print(f"  {row['rank']:>2}. {row['candidate']} score={row['rfa_score']:.8f}{marker}")
    print(f"\nSelected: {rows[0]['candidate']}")
    print(f"JSON: {json_path}\nCSV:  {csv_path}")
    return rows, json_path, csv_path


def main(argv=None):
    raise SystemExit(
        "The selection CLI is retired. Use `python -m qwen_cross_model.refine_target` "
        "or `python -m qwen_cross_model.pipeline`."
    )


if __name__ == "__main__":
    main()
