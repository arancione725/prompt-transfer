"""Utilities for Qwen prompt tuning: model wrapper, data loading, prompt manipulation."""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .config import (
    MAX_SEQ_LENGTH, NUM_LABELS, PROMPT_LEN,
    PROMPT_INIT_TOKEN, SUPERPOS_TEMPERATURE,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class TextClassificationDataset(Dataset):
    """Wraps a HuggingFace dataset into a simple (input_ids, attention_mask, label) tuple."""

    def __init__(self, hf_dataset, tokenizer, max_length=MAX_SEQ_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        for example in hf_dataset:
            text = example.get("text") or example.get("sentence") or ""
            label = example["label"]
            self.data.append((text, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_sst2(tokenizer):
    """Load SST-2 dataset for training and evaluation."""
    raw = load_dataset("glue", "sst2")
    train_ds = TextClassificationDataset(raw["train"], tokenizer)
    eval_ds = TextClassificationDataset(raw["validation"], tokenizer)
    return train_ds, eval_ds


def load_dataset_by_name(name, tokenizer):
    if name == "sst2":
        return load_sst2(tokenizer)
    raise ValueError(f"Unsupported dataset: {name}")


# ---------------------------------------------------------------------------
# PromptQwenWrapper
# ---------------------------------------------------------------------------

class PromptQwenWrapper(nn.Module):
    """Wrap a Qwen2.5 causal LM for soft-prompt-based classification.

    Input:  {text_tokens}
    Actual: [soft_prompt_tokens] [text_tokens]

    Classification: take the hidden state at the LAST soft-prompt position,
                    feed through a Linear head.

    Freeze strategy:
      - backbone (Qwen model)  : frozen
      - prompt_embeddings       : trainable
      - classification_head     : trainable
    """

    def __init__(self, model_name, prompt_len=PROMPT_LEN, num_labels=NUM_LABELS,
                 init_token_name=PROMPT_INIT_TOKEN, init_text=None):
        super().__init__()
        self.prompt_len = prompt_len
        self.hidden_size = None

        # Backbone
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        self.backbone.config.output_hidden_states = True
        self.backbone.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.hidden_size = self.backbone.config.hidden_size

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.prompt_embeddings = nn.Embedding(prompt_len, self.hidden_size)

        # Initialize: prefer semantic text, fall back to single token
        if init_text is not None:
            init_ids = self.tokenizer.encode(init_text, add_special_tokens=False)
            # Cycle to fill prompt_len positions
            init_ids = (init_ids * (prompt_len // len(init_ids) + 1))[:prompt_len]
        else:
            init_token_id = self.tokenizer.convert_tokens_to_ids(init_token_name)
            if init_token_id is None or init_token_id == self.tokenizer.unk_token_id:
                init_token_id = self.tokenizer.eos_token_id
            init_ids = [init_token_id] * prompt_len

        with torch.no_grad():
            init_emb = self.backbone.model.embed_tokens.weight[torch.tensor(init_ids)].clone().float()
            self.prompt_embeddings.weight.data = init_emb

        # Classification head (fp32 — AdamW requires fp32 params)
        self.classification_head = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, prompt_embeds=None):
        r"""Args:
            input_ids:      [B, N] text token ids
            attention_mask: [B, N]
            prompt_embeds:  optional [B, L, H] — if given, use this prompt instead of lookup.
                            Enables DataParallel-safe prompt injection from external projector.
        Returns:
            logits: [B, num_labels]
        """
        B = input_ids.size(0)

        # 1. Word embeddings
        word_embeds = self.backbone.model.embed_tokens(input_ids)  # [B, N, H]

        # 2. Prompt embeddings
        if prompt_embeds is None:
            prompt_ids = torch.arange(self.prompt_len, device=input_ids.device)
            prompt_embeds = self.prompt_embeddings(prompt_ids).unsqueeze(0).expand(B, -1, -1)  # [B, L, H]
            prompt_embeds = prompt_embeds.to(dtype=self.backbone.dtype)

        # 3. Concatenate: [prompt | text]
        inputs_embeds = torch.cat([prompt_embeds, word_embeds], dim=1)  # [B, L+N, H]

        # 4. Extend attention mask
        prompt_mask = torch.ones(B, self.prompt_len, dtype=attention_mask.dtype, device=attention_mask.device)
        extended_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)  # [B, L+N]

        # 5. Forward through backbone
        outputs = self.backbone(inputs_embeds=inputs_embeds, attention_mask=extended_attention_mask)

        # 6. Hidden state at the LAST text position (causal model: last prompt token can't see text)
        last_pos_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, H]

        # 7. Classification head (fp32 — cast from fp16 hidden state)
        logits = self.classification_head(last_pos_hidden.float())  # [B, num_labels]

        return logits


# ---------------------------------------------------------------------------
# SuperPosPromptWrapper — each prompt vector = linear combination of m real tokens
# ---------------------------------------------------------------------------

class SuperPosPromptWrapper(nn.Module):
    """Soft prompt via superposition of m sampled token embeddings.

    p_i = sum_j(w_ij * e_j)  where e_j are real token embeddings, w_ij are learned weights.

    Key property: prompt vectors can NEVER leave the token manifold. This makes
    cross-model transfer trivial — same weights, different embedding lookup.

    Args:
        model_name: HuggingFace model ID
        prompt_len: number of soft prompt positions (L)
        num_labels: classification head output dim
        m: number of sampled token embeddings per position (default 128)
        sampled_ids: optional pre-existing token IDs for loading/transfer
    """

    def __init__(self, model_name, prompt_len=PROMPT_LEN, num_labels=NUM_LABELS, m=128, sampled_ids=None):
        super().__init__()
        self.prompt_len = prompt_len
        self.m = m
        self.hidden_size = None

        self.backbone = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.backbone.config.output_hidden_states = True
        self.backbone.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.hidden_size = self.backbone.config.hidden_size

        for p in self.backbone.parameters():
            p.requires_grad = False

        # --- Semantic basis injection ---
        if sampled_ids is not None:
            self.register_buffer('sampled_ids', sampled_ids.to(torch.long))
        else:
            vocab_size = self.backbone.model.embed_tokens.weight.size(0)

            # Inject SST-2 sentiment seed words into the basis so prompt vectors
            # are anchored on the sentiment manifold from the start
            try:
                from .discrete_bridge import SST2_SENTIMENT_SEEDS
                seed_words = SST2_SENTIMENT_SEEDS
            except ImportError:
                seed_words = ["positive", "negative", "good", "bad", "great", "terrible"]

            seed_ids_set = set()
            for word in seed_words:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                if len(ids) == 1:  # single-token words only for clean semantics
                    seed_ids_set.add(ids[0])

            seed_ids = list(seed_ids_set)

            if len(seed_ids) < m:
                special = set()
                for name in ["eos_token_id", "bos_token_id", "pad_token_id"]:
                    tid = getattr(self.tokenizer, name, None)
                    if tid is not None:
                        special.add(tid)
                special.add(self.tokenizer.convert_tokens_to_ids(PROMPT_INIT_TOKEN))
                candidates = [i for i in range(vocab_size) if i not in special and i not in seed_ids_set]
                perm = torch.randperm(len(candidates))[:m - len(seed_ids)]
                seed_ids.extend([candidates[i] for i in perm])
            else:
                seed_ids = seed_ids[:m]

            self.register_buffer('sampled_ids', torch.tensor(seed_ids, dtype=torch.long))

        # Trainable logits — softmax + temperature enforces sparse convex combination
        self.prompt_weights = nn.Parameter(torch.randn(prompt_len, m) * 0.02)
        with torch.no_grad():
            for i in range(prompt_len):
                self.prompt_weights.data[i, i % m] += 2.0  # initial high logit breaks symmetry

        # LM Head classification: use pretrained lm_head instead of random nn.Linear.
        # This way the prompt learns to steer the model toward outputting "positive"/"negative"
        # tokens, which transfers naturally to any target model's lm_head.
        # SST-2: 0=negative, 1=positive
        self.neg_ids = (
            self.tokenizer.encode(" negative", add_special_tokens=False)
            + self.tokenizer.encode("negative", add_special_tokens=False)
        )
        self.pos_ids = (
            self.tokenizer.encode(" positive", add_special_tokens=False)
            + self.tokenizer.encode("positive", add_special_tokens=False)
        )

    def get_prompt_embeddings(self):
        """Compute L×H prompt from L×m logits and m×H token embeddings.

        Softmax enforces convex combination on the simplex,
        guaranteeing prompt vectors stay inside the convex hull of basis tokens.
        """
        E = self.backbone.model.embed_tokens.weight[self.sampled_ids].float()  # [m, H] fp32
        norm_weights = F.softmax(self.prompt_weights / SUPERPOS_TEMPERATURE, dim=-1)
        return norm_weights @ E

    def forward(self, input_ids, attention_mask, prompt_embeds=None):
        B = input_ids.size(0)
        if prompt_embeds is None:
            prompt_embeds = self.get_prompt_embeddings().unsqueeze(0).expand(B, -1, -1)
        prompt_embeds = prompt_embeds.to(dtype=self.backbone.dtype)

        word_embeds = self.backbone.model.embed_tokens(input_ids)
        inputs_embeds = torch.cat([prompt_embeds, word_embeds], dim=1)

        prompt_mask = torch.ones(B, self.prompt_len, dtype=attention_mask.dtype, device=attention_mask.device)
        extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.backbone(inputs_embeds=inputs_embeds, attention_mask=extended_mask)
        last_pos_hidden = outputs.hidden_states[-1][:, -1, :]

        # Use pretrained lm_head: map hidden state → vocab space, extract pos/neg scores
        lm_logits = self.backbone.lm_head(last_pos_hidden)  # [B, vocab_size]
        neg_logits = lm_logits[:, self.neg_ids].max(dim=-1, keepdim=True).values  # [B, 1]
        pos_logits = lm_logits[:, self.pos_ids].max(dim=-1, keepdim=True).values  # [B, 1]
        logits = torch.cat([neg_logits, pos_logits], dim=1)  # [B, 2]: neg, pos
        return logits


# ---------------------------------------------------------------------------
# SuperPosReLUWrapper — ReLU-activated superposition (non-negative, no sum-to-1)
# ---------------------------------------------------------------------------

class SuperPosReLUWrapper(nn.Module):
    """Soft prompt via ReLU-activated superposition of m sampled token embeddings.

    p_i = sum_j(relu(w_ij) * e_j)

    Unlike softmax: weights are non-negative but NOT constrained to sum to 1.
    ReLU naturally zeros out useless basis tokens, producing sparse combinations.
    This has three advantages for cross-model transfer:
      1. Pure non-negative addition — semantic meaning preserved across embedding spaces
      2. No zero-sum constraint — each token's contribution magnitude is independent
      3. Native sparsity — ReLU zeros replace manual top-K truncation

    Args:
        model_name: HuggingFace model ID
        prompt_len: number of soft prompt positions (L)
        m: number of sampled token embeddings per position (default 128)
        sampled_ids: optional pre-existing token IDs for loading/transfer
    """

    def __init__(self, model_name, prompt_len=PROMPT_LEN, num_labels=NUM_LABELS, m=128, sampled_ids=None):
        super().__init__()
        self.prompt_len = prompt_len
        self.m = m
        self.hidden_size = None

        self.backbone = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.backbone.config.output_hidden_states = True
        self.backbone.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.hidden_size = self.backbone.config.hidden_size

        for p in self.backbone.parameters():
            p.requires_grad = False

        # --- Semantic basis injection (same as SuperPosPromptWrapper) ---
        if sampled_ids is not None:
            self.register_buffer('sampled_ids', sampled_ids.to(torch.long))
        else:
            vocab_size = self.backbone.model.embed_tokens.weight.size(0)

            try:
                from .discrete_bridge import SST2_SENTIMENT_SEEDS
                seed_words = SST2_SENTIMENT_SEEDS
            except ImportError:
                seed_words = ["positive", "negative", "good", "bad", "great", "terrible"]

            seed_ids_set = set()
            for word in seed_words:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                if len(ids) == 1:
                    seed_ids_set.add(ids[0])

            seed_ids = list(seed_ids_set)

            if len(seed_ids) < m:
                special = set()
                for name in ["eos_token_id", "bos_token_id", "pad_token_id"]:
                    tid = getattr(self.tokenizer, name, None)
                    if tid is not None:
                        special.add(tid)
                special.add(self.tokenizer.convert_tokens_to_ids(PROMPT_INIT_TOKEN))
                candidates = [i for i in range(vocab_size) if i not in special and i not in seed_ids_set]
                perm = torch.randperm(len(candidates))[:m - len(seed_ids)]
                seed_ids.extend([candidates[i] for i in perm])
            else:
                seed_ids = seed_ids[:m]

            self.register_buffer('sampled_ids', torch.tensor(seed_ids, dtype=torch.long))

        # Trainable weights — initialized positive (small uniform) so ReLU is active at start.
        # During training, unused basis tokens get pushed below 0 → ReLU → exactly 0.
        self.prompt_weights = nn.Parameter(torch.rand(prompt_len, m) * 0.1 + 0.05)

        # LM Head classification (same as SuperPosPromptWrapper)
        self.neg_ids = (
            self.tokenizer.encode(" negative", add_special_tokens=False)
            + self.tokenizer.encode("negative", add_special_tokens=False)
        )
        self.pos_ids = (
            self.tokenizer.encode(" positive", add_special_tokens=False)
            + self.tokenizer.encode("positive", add_special_tokens=False)
        )

    def get_prompt_embeddings(self):
        """Compute L×H prompt from L×m weights and m×H token embeddings.

        ReLU zeros out negative weights → pure non-negative superposition.
        No normalization, no sum-to-1 constraint.
        """
        E = self.backbone.model.embed_tokens.weight[self.sampled_ids].float()  # [m, H] fp32
        active_weights = F.relu(self.prompt_weights)  # [L, m], non-negative, naturally sparse
        return active_weights @ E

    def forward(self, input_ids, attention_mask, prompt_embeds=None):
        B = input_ids.size(0)
        if prompt_embeds is None:
            prompt_embeds = self.get_prompt_embeddings().unsqueeze(0).expand(B, -1, -1)
        prompt_embeds = prompt_embeds.to(dtype=self.backbone.dtype)

        word_embeds = self.backbone.model.embed_tokens(input_ids)
        inputs_embeds = torch.cat([prompt_embeds, word_embeds], dim=1)

        prompt_mask = torch.ones(B, self.prompt_len, dtype=attention_mask.dtype, device=attention_mask.device)
        extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.backbone(inputs_embeds=inputs_embeds, attention_mask=extended_mask)
        last_pos_hidden = outputs.hidden_states[-1][:, -1, :]

        lm_logits = self.backbone.lm_head(last_pos_hidden)
        neg_logits = lm_logits[:, self.neg_ids].max(dim=-1, keepdim=True).values
        pos_logits = lm_logits[:, self.pos_ids].max(dim=-1, keepdim=True).values
        logits = torch.cat([neg_logits, pos_logits], dim=1)
        return logits


def save_superpos_relu_prompt(wrapper, path):
    torch.save({
        "type": "superpos_relu",
        "sampled_ids": wrapper.sampled_ids.cpu(),
        "prompt_weights": wrapper.prompt_weights.data.cpu(),
        "neg_ids": list(wrapper.neg_ids),
        "pos_ids": list(wrapper.pos_ids),
        "prompt_len": wrapper.prompt_len,
        "m": wrapper.m,
        "hidden_size": wrapper.hidden_size,
    }, path)


def load_superpos_relu_prompt(model_name, prompt_path, device="cuda"):
    ckpt = torch.load(prompt_path, map_location=device)
    wrapper = SuperPosReLUWrapper(
        model_name,
        prompt_len=ckpt["prompt_len"],
        num_labels=2,
        m=ckpt["m"],
        sampled_ids=ckpt["sampled_ids"].to(device),
    )
    wrapper.prompt_weights.data.copy_(ckpt["prompt_weights"].to(device))
    wrapper.to(device)
    return wrapper


def save_superpos_prompt(wrapper, path):
    torch.save({
        "type": "superpos_lmhead",
        "sampled_ids": wrapper.sampled_ids.cpu(),
        "prompt_weights": wrapper.prompt_weights.data.cpu(),
        "neg_ids": list(wrapper.neg_ids),
        "pos_ids": list(wrapper.pos_ids),
        "prompt_len": wrapper.prompt_len,
        "m": wrapper.m,
        "hidden_size": wrapper.hidden_size,
    }, path)


def load_superpos_prompt(model_name, prompt_path, device="cuda"):
    ckpt = torch.load(prompt_path, map_location=device)
    wrapper = SuperPosPromptWrapper(
        model_name,
        prompt_len=ckpt["prompt_len"],
        num_labels=2,  # fixed for LM Head binary classification
        m=ckpt["m"],
        sampled_ids=ckpt["sampled_ids"].to(device),
    )
    wrapper.prompt_weights.data.copy_(ckpt["prompt_weights"].to(device))
    wrapper.to(device)
    return wrapper


@torch.no_grad()
def transfer_superpos(src_prompt_path, tgt_model_name):
    """Training-free transfer: same weights, target model's token embeddings."""
    ckpt = torch.load(src_prompt_path, map_location="cpu")
    tgt_model = AutoModelForCausalLM.from_pretrained(
        tgt_model_name, torch_dtype=torch.float16, device_map="cpu"
    )
    tgt_emb = tgt_model.model.embed_tokens.weight[ckpt["sampled_ids"]].float()
    weights = ckpt["prompt_weights"].float().detach()
    norm_weights = F.softmax(weights / SUPERPOS_TEMPERATURE, dim=-1)
    projected = (norm_weights @ tgt_emb).detach()
    del tgt_model
    return projected, ckpt


# ---------------------------------------------------------------------------
# Prompt save / load helpers
# ---------------------------------------------------------------------------

def save_prompt(wrapper, path):
    """Save only the prompt embeddings and classification head."""
    torch.save({
        "prompt_embeddings": wrapper.prompt_embeddings.state_dict(),
        "classification_head": wrapper.classification_head.state_dict(),
        "prompt_len": wrapper.prompt_len,
        "hidden_size": wrapper.hidden_size,
    }, path)


def load_prompt(wrapper_or_model_name, prompt_path, device="cuda"):
    """Load prompt embeddings + classification head.

    If a model_name string is passed, create a PromptQwenWrapper first.
    Returns the wrapper with loaded weights.
    """
    checkpoint = torch.load(prompt_path, map_location=device)

    if isinstance(wrapper_or_model_name, str):
        wrapper = PromptQwenWrapper(
            wrapper_or_model_name,
            prompt_len=checkpoint["prompt_len"],
            num_labels=NUM_LABELS,
        )
    else:
        wrapper = wrapper_or_model_name

    wrapper.prompt_embeddings.load_state_dict(checkpoint["prompt_embeddings"])
    wrapper.classification_head.load_state_dict(checkpoint["classification_head"])
    wrapper.to(device)
    return wrapper


def get_prompt_tensor(wrapper):
    """Return the prompt embedding tensor [prompt_len, hidden_size] on CPU."""
    return wrapper.prompt_embeddings.weight.data.clone().cpu()


def set_prompt_tensor(wrapper, prompt_tensor):
    """Set the prompt embedding tensor [prompt_len, hidden_size]."""
    wrapper.prompt_embeddings.weight.data = prompt_tensor.to(wrapper.prompt_embeddings.weight.device)


# ---------------------------------------------------------------------------
# Anchor loss (keep prompt vectors near token embeddings)
# ---------------------------------------------------------------------------

def _get_or_build_anchor_cache(model):
    """Build (or retrieve) a normalized cache of the token embedding matrix."""
    # Unwrap DDP if needed
    if hasattr(model, "module"):
        model = model.module

    if not hasattr(model, "_anchor_cache"):
        emb = model.backbone.model.embed_tokens.weight.detach()
        emb_norm = F.normalize(emb.float(), dim=1)
        model._anchor_cache = emb_norm
    return model._anchor_cache


def compute_anchor_loss(model, k=8):
    """Regularization: pull each prompt vector towards the centroid of its K nearest tokens.

    Using K>1 prevents all vectors from collapsing to the same token — each position
    is pulled towards a REGION of the token manifold, preserving diversity.

    Also adds a weak repulsion term: cosine similarity between prompt vectors is penalized
    to prevent them from clustering on the same set of tokens.
    """
    if hasattr(model, "module"):
        model = model.module

    emb_norm = _get_or_build_anchor_cache(model)           # [V, H]
    prompt_emb = model.prompt_embeddings.weight            # [L, H]
    prompt_norm = F.normalize(prompt_emb, dim=1)           # [L, H]

    cos_sim = prompt_norm @ emb_norm.T                     # [L, V]
    topk_cos, _ = cos_sim.topk(k, dim=1)                  # [L, K]
    pull_loss = (1 - topk_cos.mean()).mean()               # pull towards K-nearest region

    # Repulsion: penalize prompt vectors being too similar to each other
    p2p_sim = prompt_norm @ prompt_norm.T                  # [L, L]
    mask = ~torch.eye(prompt_emb.size(0), dtype=torch.bool, device=p2p_sim.device)
    repel_loss = p2p_sim[mask].clamp_min(0).mean()         # avg pairwise similarity

    return pull_loss + 0.5 * repel_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

_nan_printed = False

def train_one_epoch(model, dataloader, optimizer, scheduler, device, desc="Train", anchor_weight=0.0):
    global _nan_printed
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_anchor_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    pbar = tqdm(dataloader, desc=desc, leave=False)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        task_loss = loss_fn(logits, labels)

        if anchor_weight > 0:
            anchor_loss = compute_anchor_loss(model)
            loss = task_loss + anchor_weight * anchor_loss
        else:
            anchor_loss = torch.tensor(0.0)
            loss = task_loss

        # Debug NaN
        if not _nan_printed and (torch.isnan(loss) or torch.isnan(logits).any()):
            print(f"\n  [DEBUG NaN] logits: min={logits.min().item():.4f} max={logits.max().item():.4f} "
                  f"has_nan={torch.isnan(logits).any().item()}  has_inf={torch.isinf(logits).any().item()}")
            _nan_printed = True

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_anchor_loss += anchor_loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "task": f"{task_loss.item():.4f}"})

    n = len(dataloader)
    if anchor_weight > 0:
        print(f"  avg task_loss={total_task_loss/n:.4f}  anchor_loss={total_anchor_loss/n:.4f}")
    return total_loss / n


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_prompt_checkpoint(path, epoch, model, optimizer, scheduler, best_acc):
    """Save a full training checkpoint (for resume)."""
    torch.save({
        "type": "prompt",
        "epoch": epoch,
        "model_state_dict": {
            "prompt_embeddings": model.prompt_embeddings.state_dict(),
            "classification_head": model.classification_head.state_dict(),
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_acc": best_acc,
    }, path)


def load_prompt_checkpoint(path, model, optimizer=None, scheduler=None, device="cuda"):
    """Load a training checkpoint and restore model/optimizer/scheduler state.

    Returns the epoch to resume from.
    """
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("type") != "prompt":
        raise ValueError(f"Checkpoint type is '{ckpt.get('type')}', expected 'prompt'")

    model.prompt_embeddings.load_state_dict(ckpt["model_state_dict"]["prompt_embeddings"])
    model.classification_head.load_state_dict(ckpt["model_state_dict"]["classification_head"])

    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt["epoch"], ckpt.get("best_acc", 0.0)


def save_projector_checkpoint(path, epoch, projector, classification_head, optimizer, scheduler, best_acc, src_prompt_emb=None):
    """Save a projector training checkpoint."""
    ckpt = {
        "type": "projector",
        "epoch": epoch,
        "projector_state_dict": projector.state_dict(),
        "classification_head_state_dict": classification_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_acc": best_acc,
    }
    if src_prompt_emb is not None:
        ckpt["src_prompt_emb"] = src_prompt_emb.cpu()
    torch.save(ckpt, path)


def load_projector_checkpoint(path, projector, classification_head, optimizer=None, scheduler=None, device="cuda"):
    """Load a projector training checkpoint."""
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("type") != "projector":
        raise ValueError(f"Checkpoint type is '{ckpt.get('type')}', expected 'projector'")

    projector.load_state_dict(ckpt["projector_state_dict"])
    classification_head.load_state_dict(ckpt["classification_head_state_dict"])

    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt["epoch"], ckpt.get("best_acc", 0.0), ckpt.get("src_prompt_emb")


def format_time(seconds):
    """Format seconds into h/m/s string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    elif m > 0:
        return f"{m}m{s:02d}s"
    else:
        return f"{s}s"


@torch.no_grad()
def evaluate_model(model, dataloader, device, prompt_embeds=None, desc="Eval"):
    model.eval()
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        kwargs = {}
        if prompt_embeds is not None:
            B = input_ids.size(0)
            kwargs["prompt_embeds"] = prompt_embeds.unsqueeze(0).expand(B, -1, -1).to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0


def build_dataloader(dataset, batch_size, shuffle=True, sampler=None):
    if sampler is not None:
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
