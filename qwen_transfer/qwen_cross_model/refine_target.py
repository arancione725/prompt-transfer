#!/usr/bin/env python3
"""Label-free RFA optimization of one target SuperPos prompt."""

import argparse
from dataclasses import dataclass
import gc
import json
import math
import warnings
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import (DEFAULT_SEED, MAX_LENGTH, REFINE_BATCH_SIZE, REFINE_EPOCHS,
                     REFINE_LR, RFA_TEMPERATURE, STAY_WEIGHT, TARGET_MODEL)
from .rfa import (AnchorPlan, ProbeDataset, build_anchor_plan, collect_logits, coordinate_kl,
                  forward_next_token_logits, load_model, load_probe_bank,
                  load_tokenizer, rowwise_rfa, set_seed,
                  validate_probe_encoding_alignment, validate_sampled_ids,
                  validate_tokenizer_alignment)
from .superpos import (file_sha256, load_checkpoint, prompt_from_coordinates,
                       runtime_versions, save_refined_prompt)


@dataclass
class TopKSupport:
    """Fixed per-position source Top-K support for one sweep candidate."""

    basis_indices: torch.Tensor
    token_ids: torch.Tensor
    source_logits: torch.Tensor
    source_coordinates: torch.Tensor

    @property
    def k(self):
        return int(self.basis_indices.size(1))


def build_topk_support(checkpoint, k):
    """Build deterministic per-position support; K=full preserves original order."""
    basis_size = checkpoint.prompt_logits.size(1)
    if not 1 <= int(k) <= basis_size:
        raise ValueError(f"basis K must be in [1, {basis_size}], got {k}")
    k = int(k)
    if k == basis_size:
        indices = torch.arange(basis_size).expand(
            checkpoint.prompt_logits.size(0), -1
        ).clone()
    else:
        # Match the historical direct-transfer implementation exactly,
        # including torch.topk's tie behavior on activated coordinates.
        indices = torch.topk(checkpoint.coordinates, k, dim=1).indices
    active_logits = checkpoint.prompt_logits.gather(1, indices).clone()
    coordinates = F.softmax(active_logits / checkpoint.temperature, dim=-1)
    token_ids = checkpoint.sampled_ids.index_select(0, indices.flatten()).reshape_as(indices)
    return TopKSupport(indices, token_ids, active_logits, coordinates)


def prompt_from_topk_coordinates(coordinates, support_token_ids, embedding_weight):
    """Project [L,K] coordinates through a different K-token basis at each position."""
    if coordinates.shape != support_token_ids.shape or coordinates.ndim != 2:
        raise ValueError("Top-K coordinates and support IDs must have equal [L,K] shape")
    if support_token_ids.min() < 0 or support_token_ids.max() >= embedding_weight.size(0):
        raise ValueError("Top-K support contains an out-of-range token ID")
    device = embedding_weight.device
    token_ids = support_token_ids.to(device)
    weights = coordinates.to(device)
    # The full-K case shares the same ordered basis in every row; matrix
    # multiplication reproduces the existing direct-transfer implementation.
    if torch.equal(token_ids, token_ids[0].unsqueeze(0).expand_as(token_ids)):
        basis = embedding_weight.index_select(0, token_ids[0]).float()
        return weights @ basis
    basis = embedding_weight.index_select(0, token_ids.flatten()).float()
    basis = basis.reshape(token_ids.size(0), token_ids.size(1), -1)
    return torch.einsum("lk,lkh->lh", weights, basis)


def _anchor_plan_from_payload(payload):
    if payload is None:
        return None
    return AnchorPlan(
        payload["indices"], payload["selected_counts"],
        payload["achieved_coverages"], payload["aggregate_source_js"],
    )


def _cached_ranges(split, batch_size):
    size = split["input_ids"].size(0)
    for start in range(0, size, batch_size):
        yield start, min(start + batch_size, size)


def _score_summary(values):
    values = values.double()
    count = values.numel()
    std = values.std(unbiased=count > 1)
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(std),
        "se": float(std / math.sqrt(count)) if count else float("nan"),
        "min": float(values.min()),
        "max": float(values.max()),
        "per_probe": values.tolist(),
    }


@torch.no_grad()
def score_topk_prompt(model, split, token_ids, support, active_logits,
                      coordinate_temperature, rfa_temperature, stay_weight,
                      batch_size):
    coordinates = F.softmax(active_logits / coordinate_temperature, dim=-1)
    prompt = prompt_from_topk_coordinates(
        coordinates, support.token_ids, model.get_input_embeddings().weight
    )
    anchor_plan = _anchor_plan_from_payload(split.get("anchor_plan"))
    rows = []
    input_device = model.get_input_embeddings().weight.device
    for start, stop in _cached_ranges(split, batch_size):
        prompted = forward_next_token_logits(
            model,
            split["input_ids"][start:stop].to(input_device),
            split["attention_mask"][start:stop].to(input_device),
            prompt,
        )
        prompted = prompted.index_select(1, token_ids.to(prompted.device))
        effect = prompted - split["target_reference"][start:stop].to(prompted.device)
        source = split["source_effect"][start:stop].to(prompted.device)
        rows.append(rowwise_rfa(
            source, effect, rfa_temperature,
            _slice_anchor_plan(anchor_plan, start, stop),
        ).cpu())
    values = torch.cat(rows)
    stay = coordinate_kl(coordinates, support.source_coordinates.to(coordinates.device))
    summary = _score_summary(values)
    summary["stay_kl"] = float(stay.cpu())
    summary["objective"] = summary["mean"] + stay_weight * summary["stay_kl"]
    if not all(math.isfinite(summary[name]) for name in ("mean", "stay_kl", "objective")):
        raise FloatingPointError("Non-finite Top-K refinement metric")
    return summary


def refine_topk_from_cache(model, checkpoint, cache, basis_k, epochs, batch_size,
                           learning_rate, stay_weight, cache_sha256):
    """Refine one fixed-support K candidate using a precomputed label-free cache."""
    support = build_topk_support(checkpoint, basis_k)
    device = model.get_input_embeddings().weight.device
    active_logits = torch.nn.Parameter(support.source_logits.clone().to(device))
    token_ids = cache["token_ids"]
    direct_train = score_topk_prompt(
        model, cache["train"], token_ids, support, active_logits,
        checkpoint.temperature, cache["rfa_temperature"], stay_weight, batch_size,
    )
    direct_validation = score_topk_prompt(
        model, cache["validation"], token_ids, support, active_logits,
        checkpoint.temperature, cache["rfa_temperature"], stay_weight, batch_size,
    )
    history = [{"epoch": 0, "objective": direct_train["objective"],
                "rfa": direct_train["mean"], "stay_kl": direct_train["stay_kl"]}]
    best = (direct_train["objective"], active_logits.detach().cpu().clone())

    if support.k > 1 and epochs > 0:
        optimizer = torch.optim.Adam([active_logits], lr=learning_rate)
        train_split = cache["train"]
        anchor_plan = _anchor_plan_from_payload(train_split.get("anchor_plan"))
        source_coordinates = support.source_coordinates.to(device)
        input_device = model.get_input_embeddings().weight.device
        for epoch in range(epochs):
            for start, stop in _cached_ranges(train_split, batch_size):
                optimizer.zero_grad(set_to_none=True)
                coordinates = F.softmax(active_logits / checkpoint.temperature, dim=-1)
                prompt = prompt_from_topk_coordinates(
                    coordinates, support.token_ids, model.get_input_embeddings().weight
                )
                prompted = forward_next_token_logits(
                    model,
                    train_split["input_ids"][start:stop].to(input_device),
                    train_split["attention_mask"][start:stop].to(input_device),
                    prompt,
                )
                prompted = prompted.index_select(1, token_ids.to(prompted.device))
                effect = prompted - train_split["target_reference"][start:stop].to(prompted.device)
                source = train_split["source_effect"][start:stop].to(prompted.device)
                rfa = rowwise_rfa(
                    source, effect, cache["rfa_temperature"],
                    _slice_anchor_plan(anchor_plan, start, stop),
                ).mean()
                stay = coordinate_kl(coordinates, source_coordinates)
                loss = rfa + stay_weight * stay
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"K={support.k}: non-finite objective")
                loss.backward()
                _validate_gradient(active_logits, loss)
                torch.nn.utils.clip_grad_norm_([active_logits], 1.0)
                optimizer.step()
            measured = score_topk_prompt(
                model, train_split, token_ids, support, active_logits,
                checkpoint.temperature, cache["rfa_temperature"], stay_weight,
                batch_size,
            )
            row = {"epoch": epoch + 1, "objective": measured["objective"],
                   "rfa": measured["mean"], "stay_kl": measured["stay_kl"]}
            history.append(row)
            if row["objective"] < best[0]:
                best = (row["objective"], active_logits.detach().cpu().clone())

    best_logits = best[1]
    refined_train = score_topk_prompt(
        model, cache["train"], token_ids, support, best_logits.to(device),
        checkpoint.temperature, cache["rfa_temperature"], stay_weight, batch_size,
    )
    refined_validation = score_topk_prompt(
        model, cache["validation"], token_ids, support, best_logits.to(device),
        checkpoint.temperature, cache["rfa_temperature"], stay_weight, batch_size,
    )
    direct_coordinates = support.source_coordinates
    refined_coordinates = F.softmax(best_logits / checkpoint.temperature, dim=-1)
    direct_prompt = prompt_from_topk_coordinates(
        direct_coordinates, support.token_ids, model.get_input_embeddings().weight
    ).detach().cpu()
    if support.k == checkpoint.sampled_ids.numel():
        expected_full_prompt = prompt_from_coordinates(
            checkpoint.coordinates,
            checkpoint.sampled_ids,
            model.get_input_embeddings().weight,
        ).detach().cpu()
        if not torch.allclose(direct_prompt, expected_full_prompt, atol=1e-6, rtol=1e-6):
            raise RuntimeError("Full-basis Top-K prompt does not reproduce direct transfer")
    refined_prompt = prompt_from_topk_coordinates(
        refined_coordinates, support.token_ids, model.get_input_embeddings().weight
    ).detach().cpu()
    checkpoint_payload = {
        "type": "rfa_refined_target_prompt_topk",
        "format_version": 2,
        "weight_mode": "softmax_topk_fixed_support",
        "outside_support_mass": 0.0,
        "support_tie_policy": "torch.topk_on_source_coordinates",
        "basis_k": support.k,
        "basis_size": checkpoint.sampled_ids.numel(),
        "temperature": checkpoint.temperature,
        "seed": checkpoint.seed,
        "source_model": checkpoint.source_model,
        "target_model": cache["target_model"],
        "target_model_revision": cache.get("target_model_revision"),
        "position_id_mode": checkpoint.position_id_mode,
        "source_checkpoint": str(checkpoint.path),
        "source_checkpoint_sha256": file_sha256(checkpoint.path),
        "cache_sha256": cache_sha256,
        "source_signature": "full_source_prompt",
        "support_basis_indices": support.basis_indices,
        "support_token_ids": support.token_ids,
        "source_active_logits": support.source_logits,
        "source_topk_coordinates": direct_coordinates,
        "target_active_logits": best_logits,
        "target_topk_coordinates": refined_coordinates,
        "direct_target_prompt": direct_prompt,
        "refined_target_prompt": refined_prompt,
        "direct_train_rfa": direct_train,
        "direct_validation_rfa": direct_validation,
        "refined_train_rfa": refined_train,
        "refined_validation_rfa": refined_validation,
        "runtime_versions": runtime_versions(),
    }
    result = {
        "basis_k": support.k,
        "direct_train_rfa": direct_train,
        "direct_validation_rfa": direct_validation,
        "refined_train_rfa": refined_train,
        "refined_validation_rfa": refined_validation,
        "optimized": support.k > 1 and epochs > 0,
        "cache_sha256": cache_sha256,
    }
    return checkpoint_payload, result, history


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Optimize target prompt with label-free RFA")
    p.add_argument("--source-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-model", default=TARGET_MODEL)
    p.add_argument("--dataset", default="sst2")
    p.add_argument("--probe-file")
    p.add_argument("--text-field", default="sentence")
    p.add_argument("--probe-split", default="train")
    p.add_argument("--num-probes", type=int, default=64)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--epochs", type=int, default=REFINE_EPOCHS)
    p.add_argument("--batch-size", type=int, default=REFINE_BATCH_SIZE)
    p.add_argument("--learning-rate", type=float, default=REFINE_LR)
    p.add_argument("--temperature", type=float, default=RFA_TEMPERATURE)
    p.add_argument("--stay-weight", type=float, default=STAY_WEIGHT)
    p.add_argument("--anchor-mode", choices=["full", "source-effect"], default="full")
    p.add_argument("--anchor-coverage", type=float, default=0.95)
    p.add_argument("--min-anchor-tokens", type=int, default=512)
    p.add_argument("--max-anchor-tokens", type=int, default=8192)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default=None)
    args = p.parse_args(argv)
    positive = {"num_probes": args.num_probes, "max_length": args.max_length,
                "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                "temperature": args.temperature}
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0:
            p.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.epochs < 0:
        p.error("--epochs must be non-negative")
    if not math.isfinite(args.stay_weight) or args.stay_weight < 0:
        p.error("--stay-weight must be finite and non-negative")
    if not 0 < args.anchor_coverage <= 1:
        p.error("--anchor-coverage must be in (0, 1]")
    if args.min_anchor_tokens < 1 or args.max_anchor_tokens < args.min_anchor_tokens:
        p.error("anchor bounds must satisfy 1 <= min <= max")
    return args


def _batches(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _slice_anchor_plan(plan, start, stop):
    if plan is None:
        return None
    return type(plan)(plan.indices[start:stop], plan.selected_counts[start:stop],
                      plan.achieved_coverages[start:stop],
                      plan.aggregate_source_js[start:stop])


@torch.no_grad()
def _measure_objective(model, loader, logits_parameter, checkpoint, token_ids,
                       target_reference, source_effect, temperature, stay_weight,
                       anchor_plan):
    coordinates = F.softmax(logits_parameter / checkpoint.temperature, dim=-1)
    prompt = prompt_from_coordinates(coordinates, checkpoint.sampled_ids,
                                     model.get_input_embeddings().weight)
    values = []
    offset = 0
    input_device = model.get_input_embeddings().weight.device
    for batch in loader:
        count = batch["input_ids"].size(0)
        prompted = forward_next_token_logits(model, batch["input_ids"].to(input_device),
                                              batch["attention_mask"].to(input_device), prompt)
        prompted = prompted.index_select(1, token_ids.to(prompted.device))
        target_effect = prompted - target_reference[offset:offset + count].to(prompted.device)
        source_batch = source_effect[offset:offset + count].to(prompted.device)
        values.append(rowwise_rfa(source_batch, target_effect, temperature,
                                  _slice_anchor_plan(anchor_plan, offset, offset + count)).cpu())
        offset += count
    rfa = torch.cat(values).mean()
    stay = coordinate_kl(coordinates, checkpoint.coordinates.to(coordinates.device)).cpu()
    measured = (float(rfa + stay_weight * stay), float(rfa), float(stay))
    if not all(math.isfinite(value) for value in measured):
        raise FloatingPointError(f"Non-finite refinement metrics: {measured}")
    return measured


def _validate_gradient(parameter, loss):
    gradient = parameter.grad
    if gradient is None:
        raise RuntimeError("Target prompt logits received no gradient")
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Target prompt gradient contains NaN or Inf")
    norm = float(gradient.norm().detach())
    if norm == 0.0 and float(loss.detach()) > 1e-12:
        raise RuntimeError("Target prompt gradient is zero for a non-zero objective")
    return norm


def _enable_refinement_mode(model):
    device_map = getattr(model, "hf_device_map", None) or {}
    placements = {str(value) for value in device_map.values()}
    if "disk" in placements or "cpu" in placements:
        raise ValueError("RFA backward does not support CPU/disk device-map offload")
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    # HF checkpointing is active only in train mode. Qwen2.5 has zero dropout;
    # reject a non-zero dropout config so prompted/reference behavior stays fixed.
    for name in ("attention_dropout", "hidden_dropout", "dropout"):
        value = getattr(model.config, name, 0.0)
        if value is not None and float(value) != 0.0:
            raise ValueError(f"Refinement requires deterministic zero dropout; {name}={value}")
    model.train()
    model.requires_grad_(False)


def run(args):
    set_seed(args.seed)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    target_path = out / "refined_target.pt"
    stage_outputs = [target_path, out / "refinement_history.json", out / "probe_bank.json"]
    existing = [path for path in stage_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")
    checkpoint = load_checkpoint(args.source_checkpoint)
    source_revision = checkpoint.payload.get("source_model_revision")
    source_tokenizer_revision = checkpoint.payload.get("source_tokenizer_revision") or source_revision
    if source_revision is None:
        warnings.warn(
            "Source checkpoint has no model revision; exact remote-weight reproducibility "
            "cannot be guaranteed for this legacy checkpoint.", RuntimeWarning
        )
    source_tok = load_tokenizer(checkpoint.source_model, source_tokenizer_revision)
    target_tok = load_tokenizer(args.target_model)
    target_revision = getattr(target_tok, "init_kwargs", {}).get("_commit_hash")
    if target_revision is None:
        warnings.warn(
            "Target tokenizer exposes no commit revision; the checkpoint will still record "
            "the target model revision when available.", RuntimeWarning
        )
    vocabulary = validate_tokenizer_alignment(source_tok, target_tok)
    validate_sampled_ids(checkpoint.sampled_ids, source_tok, target_tok)
    probes = load_probe_bank(args)  # returns texts only; labels are never materialized
    validate_probe_encoding_alignment(
        probes.texts, source_tok, target_tok, args.max_length
    )
    probe_data = ProbeDataset(probes.texts, source_tok, args.max_length)
    loader = _batches(probe_data, args.batch_size)
    token_ids = vocabulary.token_ids

    source = load_model(checkpoint.source_model, args.dtype, args.device, args.device_map,
                        source_revision)
    source_prompt = prompt_from_coordinates(checkpoint.coordinates,
                                             checkpoint.sampled_ids,
                                             source.get_input_embeddings().weight)
    source_reference = collect_logits(source, loader, None, token_ids, "source/reference")
    source_prompted = collect_logits(source, loader, source_prompt, token_ids, "source/prompt")
    source_effect = (source_prompted - source_reference).clone()
    anchor_plan = None
    if args.anchor_mode == "source-effect":
        # Source-only Top-K by mass of the exact RFA source distribution q_s.
        contributions = F.softmax(
            source_effect.double() / args.temperature, dim=-1
        )
        anchor_plan = build_anchor_plan(contributions, args.anchor_coverage,
                                        args.min_anchor_tokens, args.max_anchor_tokens)
    del source_reference, source_prompted, source_prompt, source
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target = load_model(args.target_model, args.dtype, args.device, args.device_map,
                        target_revision)
    target.config.use_cache = False
    target_reference = collect_logits(target, loader, None, token_ids, "target/reference").clone()
    _enable_refinement_mode(target)
    logits_parameter = torch.nn.Parameter(checkpoint.prompt_logits.clone().to(
        target.get_input_embeddings().weight.device))
    optimizer = torch.optim.Adam([logits_parameter], lr=args.learning_rate)
    source_coordinates = checkpoint.coordinates.to(logits_parameter.device)
    initial = _measure_objective(target, loader, logits_parameter, checkpoint, token_ids,
                                 target_reference, source_effect, args.temperature,
                                 args.stay_weight, anchor_plan)
    best = (*initial, logits_parameter.detach().cpu().clone())
    history = [{"epoch": 0, "objective": initial[0], "rfa": initial[1],
                "stay_kl": initial[2]}]
    print(json.dumps(history[0]))
    for epoch in range(args.epochs):
        offset = 0
        for batch in loader:
            count = batch["input_ids"].size(0)
            optimizer.zero_grad(set_to_none=True)
            coordinates = F.softmax(logits_parameter / checkpoint.temperature, dim=-1)
            prompt = prompt_from_coordinates(coordinates, checkpoint.sampled_ids,
                                             target.get_input_embeddings().weight)
            input_device = target.get_input_embeddings().weight.device
            prompted = forward_next_token_logits(target, batch["input_ids"].to(input_device),
                                                   batch["attention_mask"].to(input_device), prompt)
            prompted = prompted.index_select(1, token_ids.to(prompted.device))
            target_effect = prompted - target_reference[offset:offset + count].to(prompted.device)
            source_batch = source_effect[offset:offset + count].to(prompted.device)
            local_plan = _slice_anchor_plan(anchor_plan, offset, offset + count)
            rfa = rowwise_rfa(source_batch, target_effect, args.temperature, local_plan).mean()
            stay = coordinate_kl(coordinates, source_coordinates)
            loss = rfa + args.stay_weight * stay
            if not torch.isfinite(loss):
                raise FloatingPointError("Refinement objective contains NaN or Inf")
            loss.backward()
            _validate_gradient(logits_parameter, loss)
            torch.nn.utils.clip_grad_norm_([logits_parameter], 1.0)
            optimizer.step()
            offset += count
        measured = _measure_objective(target, loader, logits_parameter, checkpoint, token_ids,
                                      target_reference, source_effect, args.temperature,
                                      args.stay_weight, anchor_plan)
        metrics = {"epoch": epoch + 1, "objective": measured[0],
                   "rfa": measured[1], "stay_kl": measured[2]}
        history.append(metrics)
        print(json.dumps(metrics))
        if metrics["objective"] < best[0]:
            best = (metrics["objective"], metrics["rfa"], metrics["stay_kl"],
                    logits_parameter.detach().cpu().clone())

    best_logits = best[3]
    best_coordinates = F.softmax(best_logits / checkpoint.temperature, dim=-1)
    best_prompt = prompt_from_coordinates(best_coordinates, checkpoint.sampled_ids,
                                          target.get_input_embeddings().weight).cpu()
    anchor_meta = {"mode": args.anchor_mode, "other_bucket": anchor_plan is not None,
                   "fixed_before_optimization": True}
    if anchor_plan is not None:
        achieved_min = min(anchor_plan.achieved_coverages)
        if achieved_min + 1e-12 < args.anchor_coverage:
            warnings.warn(
                "Sparse anchor cap prevented the requested coverage: "
                f"min achieved={achieved_min:.6f}, requested={args.anchor_coverage:.6f}",
                RuntimeWarning,
            )
        anchor_meta.update({"coverage": args.anchor_coverage,
                            "coverage_kind": "source RFA effect-distribution mass",
                            "aligned_vocabulary_indices": anchor_plan.indices,
                            "selected_token_ids": [
                                token_ids.index_select(0, indices).cpu()
                                for indices in anchor_plan.indices
                            ],
                            "selected_counts": anchor_plan.selected_counts,
                            "achieved_coverage_min": achieved_min,
                            "achieved_coverage_mean": sum(anchor_plan.achieved_coverages) / len(anchor_plan.achieved_coverages),
                            "achieved_coverage_max": max(anchor_plan.achieved_coverages)})
    probe_meta = {"source": probes.source, "indices": probes.indices,
                  "fingerprint": probes.fingerprint, "num_probes": len(probes.texts),
                  "texts": probes.texts}
    save_refined_prompt(target_path, checkpoint, args.target_model, best_logits, best_prompt,
                        best[0], best[1], best[2], vars(args), probe_meta, anchor_meta,
                        getattr(target.config, "_commit_hash", None))
    (out / "refinement_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "probe_bank.json").write_text(json.dumps(probe_meta, indent=2), encoding="utf-8")
    return target_path


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
