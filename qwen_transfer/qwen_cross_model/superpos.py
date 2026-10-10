"""SuperPos parameterization, checkpoint validation, and direct transfer."""

from dataclasses import dataclass
import hashlib
import importlib.metadata
import math
from pathlib import Path
from typing import Mapping, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    BASIS_SIZE,
    PROMPT_LENGTH,
    SST2_SEED_WORDS,
    SUPERPOS_TEMPERATURE,
)


def verbalizer_ids(tokenizer, word):
    ids = set()
    for prefix in ("", " "):
        encoded = tokenizer.encode(prefix + word, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Verbalizer {prefix + word!r} must be one token, got {encoded}"
            )
        ids.add(encoded[0])
    return sorted(ids)


def _sample_basis(tokenizer, vocabulary_size, basis_size, seed):
    semantic_ids = set()
    for word in SST2_SEED_WORDS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            semantic_ids.add(ids[0])
    semantic = sorted(semantic_ids)
    special = set(getattr(tokenizer, "all_special_ids", None) or [])
    candidates = [
        token_id
        for token_id in range(min(len(tokenizer), vocabulary_size))
        if token_id not in special and token_id not in semantic_ids
    ]
    generator = torch.Generator().manual_seed(int(seed))
    if len(semantic) < basis_size:
        needed = basis_size - len(semantic)
        if needed > len(candidates):
            raise ValueError("Not enough non-special tokens for requested basis size")
        permutation = torch.randperm(len(candidates), generator=generator)[:needed]
        semantic.extend(candidates[index] for index in permutation.tolist())
    return torch.tensor(semantic[:basis_size], dtype=torch.long)


class SuperPosModel(nn.Module):
    """Frozen causal LM with trainable softmax coordinates over token embeddings."""

    def __init__(
        self,
        model_name,
        prompt_length=PROMPT_LENGTH,
        basis_size=BASIS_SIZE,
        temperature=SUPERPOS_TEMPERATURE,
        seed=42,
        sampled_ids=None,
        dtype=torch.float16,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.prompt_length = int(prompt_length)
        self.basis_size = int(basis_size)
        self.temperature = float(temperature)
        self.seed = int(seed)
        if self.prompt_length < 1 or self.basis_size < 1:
            raise ValueError("prompt_length and basis_size must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        )
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        if sampled_ids is None:
            sampled_ids = _sample_basis(
                self.tokenizer,
                self.backbone.get_input_embeddings().weight.size(0),
                self.basis_size,
                self.seed,
            )
        sampled_ids = sampled_ids.detach().long().flatten()
        if sampled_ids.numel() != self.basis_size:
            raise ValueError("sampled_ids size must equal basis_size")
        self.register_buffer("sampled_ids", sampled_ids)

        self.prompt_logits = nn.Parameter(
            torch.randn(self.prompt_length, self.basis_size) * 0.02
        )
        with torch.no_grad():
            for position in range(self.prompt_length):
                self.prompt_logits[position, position % self.basis_size] += 2.0
        self.negative_ids = verbalizer_ids(self.tokenizer, "negative")
        self.positive_ids = verbalizer_ids(self.tokenizer, "positive")

    def coordinates(self):
        return F.softmax(self.prompt_logits / self.temperature, dim=-1)

    def prompt_embeddings(self):
        basis = self.backbone.get_input_embeddings().weight[self.sampled_ids].float()
        return self.coordinates() @ basis

    def forward(self, input_ids, attention_mask):
        from .rfa import last_valid_indices

        prompt = self.prompt_embeddings()
        word_embeddings = self.backbone.get_input_embeddings()(input_ids)
        prompt_batch = prompt.to(word_embeddings.dtype).unsqueeze(0).expand(
            input_ids.size(0), -1, -1
        )
        inputs_embeds = torch.cat([prompt_batch, word_embeddings], dim=1)
        prompt_mask = torch.ones(
            input_ids.size(0),
            self.prompt_length,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        outputs = self.backbone.model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            use_cache=False,
            return_dict=True,
        )
        indices = last_valid_indices(extended_mask)
        hidden = outputs.last_hidden_state[
            torch.arange(input_ids.size(0), device=inputs_embeds.device), indices
        ]
        next_logits = self.backbone.get_output_embeddings()(hidden).float()
        negative = next_logits[:, self.negative_ids].max(dim=-1).values
        positive = next_logits[:, self.positive_ids].max(dim=-1).values
        return torch.stack([negative, positive], dim=-1)


@dataclass
class SuperPosCheckpoint:
    path: Path
    payload: Mapping
    prompt_logits: torch.Tensor
    sampled_ids: torch.Tensor
    temperature: float
    seed: int
    source_model: str
    position_id_mode: str

    @property
    def coordinates(self):
        return F.softmax(self.prompt_logits / self.temperature, dim=-1)


def validate_checkpoint(payload, path="<checkpoint>"):
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: checkpoint must be a mapping")
    if payload.get("weight_mode") != "softmax":
        raise ValueError(
            "RFA gradient refinement requires softmax SuperPos coordinates; "
            f"{path} has weight_mode={payload.get('weight_mode')!r}"
        )
    logits = payload.get("prompt_logits", payload.get("prompt_weights"))
    sampled_ids = payload.get("sampled_ids")
    if not torch.is_tensor(logits) or not torch.is_tensor(sampled_ids):
        raise ValueError(f"{path}: missing prompt_logits/prompt_weights or sampled_ids")
    logits = logits.detach().float().cpu()
    sampled_ids = sampled_ids.detach().long().flatten().cpu()
    if logits.ndim != 2 or logits.size(1) != sampled_ids.numel():
        raise ValueError(f"{path}: invalid prompt coordinate shapes")
    if sampled_ids.unique().numel() != sampled_ids.numel():
        raise ValueError(f"{path}: sampled_ids must be unique")
    if not torch.isfinite(logits).all() or sampled_ids.numel() == 0 or sampled_ids.min() < 0:
        raise ValueError(f"{path}: invalid prompt values")
    temperature = payload.get("temperature")
    if temperature is None:
        raise ValueError(f"{path}: softmax checkpoint must record temperature")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"{path}: invalid temperature")
    source_model = payload.get("source_model")
    if not source_model:
        raise ValueError(f"{path}: checkpoint must record source_model")
    position_id_mode = payload.get("position_id_mode")
    if position_id_mode is None:
        warnings.warn(
            f"{path}: legacy checkpoint has no position_id_mode; assuming the clean "
            "trainer's physical_sequence convention.", RuntimeWarning
        )
        position_id_mode = "physical_sequence"
    if position_id_mode != "physical_sequence":
        raise ValueError(f"{path}: unsupported position_id_mode={position_id_mode!r}")
    return (logits, sampled_ids, temperature, int(payload.get("seed", 0)),
            str(source_model), position_id_mode)


def load_checkpoint(path):
    path = Path(path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    logits, sampled_ids, temperature, seed, source_model, position_id_mode = validate_checkpoint(
        payload, path
    )
    return SuperPosCheckpoint(
        path, payload, logits, sampled_ids, temperature, seed, source_model,
        position_id_mode
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions():
    versions = {"python": __import__("sys").version.split()[0]}
    for package in ("torch", "transformers", "datasets", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def prompt_from_coordinates(coordinates, sampled_ids, embedding_weight):
    if sampled_ids.max() >= embedding_weight.size(0):
        raise ValueError("sampled token ID exceeds embedding rows")
    basis = embedding_weight.index_select(0, sampled_ids.to(embedding_weight.device)).float()
    return coordinates.to(basis.device) @ basis


def save_source_prompt(model, path, best_accuracy, dataset, training_config):
    path = Path(path)
    payload = {
        "type": "superpos_source_prompt",
        "format_version": 3,
        "weight_mode": "softmax",
        "temperature": float(model.temperature),
        "seed": int(model.seed),
        "source_model": model.model_name,
        "position_id_mode": "physical_sequence",
        "source_model_revision": getattr(model.backbone.config, "_commit_hash", None),
        "source_tokenizer_revision": getattr(model.tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "dataset": dataset,
        "sampled_ids": model.sampled_ids.detach().cpu(),
        "prompt_logits": model.prompt_logits.detach().cpu(),
        # Compatibility with the clean project's checkpoint key.
        "prompt_weights": model.prompt_logits.detach().cpu(),
        "prompt_length": model.prompt_length,
        "basis_size": model.basis_size,
        "best_source_accuracy": float(best_accuracy),
        "training_config": dict(training_config),
        "runtime_versions": runtime_versions(),
    }
    torch.save(payload, path)


def save_refined_prompt(
    path,
    source_checkpoint,
    target_model,
    target_logits,
    target_prompt,
    best_objective,
    best_rfa,
    best_stay,
    refinement_config,
    probe_metadata,
    anchor_metadata,
    target_model_revision=None,
):
    payload = {
        "type": "rfa_refined_target_prompt",
        "format_version": 1,
        "weight_mode": "softmax",
        "temperature": source_checkpoint.temperature,
        "seed": source_checkpoint.seed,
        "source_model": source_checkpoint.source_model,
        "target_model": target_model,
        "position_id_mode": source_checkpoint.position_id_mode,
        "target_model_revision": target_model_revision,
        "source_checkpoint": str(source_checkpoint.path),
        "source_checkpoint_sha256": file_sha256(source_checkpoint.path),
        "sampled_ids": source_checkpoint.sampled_ids,
        "source_prompt_logits": source_checkpoint.prompt_logits,
        "target_prompt_logits": target_logits.detach().cpu(),
        "target_coordinates": F.softmax(
            target_logits.detach().cpu() / source_checkpoint.temperature, dim=-1
        ),
        "target_prompt": target_prompt.detach().cpu(),
        "best_objective": float(best_objective),
        "best_rfa": float(best_rfa),
        "best_stay_kl": float(best_stay),
        "refinement_config": dict(refinement_config),
        "probe_bank": dict(probe_metadata),
        "anchors": dict(anchor_metadata),
        "runtime_versions": runtime_versions(),
    }
    torch.save(payload, Path(path))
