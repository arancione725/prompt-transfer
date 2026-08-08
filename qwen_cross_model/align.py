"""
Training-free, data-free cross-model prompt transfer via embedding-space alignment.

Qwen 系列模型共享同一个 tokenizer（vocab 完全对齐），因此两个模型的 embedding
矩阵 E_src [V, d_src] 和 E_tgt [V, d_tgt] 在 token 维度上一一对应。利用这 V 对锚点
做岭回归，可以直接解出从源空间到目标空间的线性映射 W，无需任何训练数据。

映射:  P_tgt = (P_src - mean_src) @ W + mean_tgt
其中 W 由  min_W ||E_src_centered @ W - E_tgt_centered||^2 + λ||W||^2  闭式解出。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from .config import PROMPT_LEN, NUM_LABELS


# ---------------------------------------------------------------------------
# Embedding matrix extraction
# ---------------------------------------------------------------------------

def _load_embed_only(model_name):
    """Load only the embedding matrix from a model checkpoint, without loading the full model.

    Attempts safetensors first (most Qwen2.5 models use safetensors), falls back to
    pytorch .bin files. Returns [vocab_size, hidden_size] float32 on CPU.
    """
    import json as _json
    from transformers.utils import cached_file

    try:
        idx_path = cached_file(model_name, "model.safetensors.index.json")
    except Exception:
        idx_path = None

    if idx_path is not None:
        # Sharded safetensors
        with open(idx_path) as f:
            idx = _json.load(f)
        emb_key = "model.embed_tokens.weight"
        if emb_key not in idx["weight_map"]:
            raise KeyError(f"{emb_key} not found in safetensors index")
        shard_file = idx["weight_map"][emb_key]
        shard_path = cached_file(model_name, shard_file)
        from safetensors import safe_open
        with safe_open(shard_path, framework="pt") as f:
            emb = f.get_tensor(emb_key).float()
    else:
        # Try single safetensors file
        try:
            sf_path = cached_file(model_name, "model.safetensors")
            from safetensors import safe_open
            with safe_open(sf_path, framework="pt") as f:
                emb = f.get_tensor("model.embed_tokens.weight").float()
        except Exception:
            # Fall back to full model load
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="cpu"
            )
            emb = model.model.embed_tokens.weight.detach().float()
            del model

    return emb


def get_embedding_matrix(model_or_name, device="cpu", lightweight=True):
    """Extract the input embedding matrix from a Qwen model.

    Args:
        model_or_name: model name string or already-loaded model
        device: unused (embeddings always loaded to CPU for the least-squares solve)
        lightweight: if True, load only embedding weights without full model (~1GB savings)

    Returns:
        emb: [vocab_size, hidden_size] float32 tensor on CPU
        hidden_size: int
    """
    import gc

    if isinstance(model_or_name, str):
        if lightweight:
            emb = _load_embed_only(model_or_name)
            hidden_size = emb.shape[1]
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_or_name,
                torch_dtype=torch.float16,
                device_map="cpu",
            )
            hidden_size = model.config.hidden_size
            emb = model.model.embed_tokens.weight.detach().float().cpu()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        model = model_or_name
        hidden_size = model.config.hidden_size
        emb = model.model.embed_tokens.weight.detach().float().cpu()

    return emb, hidden_size


def _build_token_string_map(tokenizer):
    """Build a dict mapping token string → token ID for the full vocabulary.

    Some tokenizers add prefix spaces. We store both the raw decoded string and
    a stripped version to maximize overlap with the other tokenizer.
    """
    t2i = {}
    for tid in range(tokenizer.vocab_size):
        raw = tokenizer.decode([tid])
        t2i[raw] = tid
        stripped = raw.lstrip()
        if stripped and stripped != raw:
            t2i[stripped] = tid  # may overwrite — rare, harmless for intersection
    return t2i


def align_embeddings_by_token_text(src_model_name, tgt_model_name):
    """Load both models' embeddings and find the token intersection by decoded text.

    Returns:
        src_aligned: [N, d_src] — src embeddings for intersecting tokens (N ≈ V_src)
        tgt_aligned: [N, d_tgt] — corresponding tgt embeddings
        n_total:     total number of intersecting token pairs
        n_skipped:   tokens in src that had no match in tgt
        src_hidden:  int
        tgt_hidden:  int
    """
    from transformers import AutoTokenizer

    # Load tokenizers (lightweight, no model weights)
    src_tok = AutoTokenizer.from_pretrained(src_model_name)
    tgt_tok = AutoTokenizer.from_pretrained(tgt_model_name)

    # Build token-string → ID maps
    tgt_t2i = _build_token_string_map(tgt_tok)

    # Find intersection: for each src token, check if it exists in tgt
    src_ids = []
    tgt_ids = []
    skipped = 0
    for sid in range(src_tok.vocab_size):
        s_text = src_tok.decode([sid])
        tid = tgt_t2i.get(s_text)
        if tid is None:
            stripped = s_text.lstrip()
            tid = tgt_t2i.get(stripped)
        if tid is not None:
            src_ids.append(sid)
            tgt_ids.append(tid)
        else:
            skipped += 1

    # Load embedding matrices
    src_emb_full, src_hidden = get_embedding_matrix(src_model_name)
    tgt_emb_full, tgt_hidden = get_embedding_matrix(tgt_model_name)

    src_aligned = src_emb_full[torch.tensor(src_ids)]  # [N, d_src]
    tgt_aligned = tgt_emb_full[torch.tensor(tgt_ids)]  # [N, d_tgt]

    return src_aligned, tgt_aligned, len(src_ids), skipped, src_hidden, tgt_hidden


# ---------------------------------------------------------------------------
# Ridge regression mapping
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_mapping(src_emb, tgt_emb, reg_lambda=1e-3):
    """Compute linear mapping W: d_src -> d_tgt via ridge regression (uniform weights).

    Args:
        src_emb: [V, d_src] source model embedding matrix
        tgt_emb: [V, d_tgt] target model embedding matrix
        reg_lambda: L2 regularization strength

    Returns:
        W:        [d_src, d_tgt] linear map
        src_mean: [d_src] source embedding mean
        tgt_mean: [d_tgt] target embedding mean
        metrics:  dict with 'mse', 'cos_sim_mean', 'cos_sim_std', 'r2'
    """
    return _solve_ridge(src_emb, tgt_emb, None, reg_lambda)


@torch.no_grad()
def compute_weighted_mapping(src_emb, tgt_emb, prompt_emb, temperature=0.1, topk=5000, reg_lambda=1e-3):
    """Compute linear mapping W weighted by proximity of anchor tokens to the prompt.

    The problem with uniform regression: 152K anchor tokens are dominated by rare tokens
    whose embeddings are poorly trained and geographically far from the prompt vectors.
    The prompt (cos-sim ~0.17) lives in a specific region; we weight anchors that are
    nearby more heavily.

    Args:
        src_emb:   [V, d_src]
        tgt_emb:   [V, d_tgt]
        prompt_emb: [L, d_src] — the trained source prompt
        temperature: softmax temperature for similarity → weight
        topk:       keep only top-K nearest anchor tokens (0 = use all, weighted)
        reg_lambda: L2 regularization

    Returns:
        W, src_mean, tgt_mean, metrics
    """
    # Compute prompt "center of mass"
    prompt_center = prompt_emb.mean(dim=0)                       # [d_src]
    prompt_center = F.normalize(prompt_center, dim=0)            # unit vector

    # Cosine similarity of every anchor token to the prompt center
    src_norm = F.normalize(src_emb, dim=1)                       # [V, d_src]
    sim = src_norm @ prompt_center                               # [V]

    if topk > 0 and topk < len(sim):
        # Keep top-K most similar tokens (local neighborhood around prompt)
        _, top_indices = sim.topk(topk)
        src_sub = src_emb[top_indices]
        tgt_sub = tgt_emb[top_indices]
        weights = torch.softmax(sim[top_indices] / temperature, dim=0)  # [K]
    else:
        src_sub = src_emb
        tgt_sub = tgt_emb
        weights = torch.softmax(sim / temperature, dim=0)        # [V]

    # Safety: remove any NaN/Inf weights
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = weights.clamp_min(1e-12)

    return _solve_ridge(src_sub, tgt_sub, weights, reg_lambda)


def _solve_ridge(src_emb, tgt_emb, weights=None, reg_lambda=1e-3):
    """Core: weighted ridge regression W = (X^T D X + λI)^{-1} X^T D Y."""
    V, d_src = src_emb.shape
    V2, d_tgt = tgt_emb.shape
    assert V == V2, f"Vocab size mismatch: {V} vs {V2}"

    device = src_emb.device

    # Center
    src_mean = src_emb.mean(dim=0)      # [d_src]
    tgt_mean = tgt_emb.mean(dim=0)      # [d_tgt]
    X_unweighted = src_emb - src_mean   # [V, d_src] — saved for eval
    Y_unweighted = tgt_emb - tgt_mean   # [V, d_tgt] — saved for eval

    X = X_unweighted.clone()
    Y = Y_unweighted.clone()

    if weights is not None:
        sqrt_w = weights.sqrt()              # [V]
        X = X * sqrt_w.unsqueeze(1)          # [V, d_src] row-scaled
        Y = Y * sqrt_w.unsqueeze(1)          # [V, d_tgt] row-scaled
        method = "weighted"
    else:
        method = "uniform"

    # Ridge:  W = (X^T X + λI)^{-1} X^T Y
    XtX = X.T @ X                       # [d_src, d_src]
    XtY = X.T @ Y                       # [d_src, d_tgt]
    I = torch.eye(d_src, device=device)
    W = torch.linalg.solve(XtX + reg_lambda * I, XtY)  # [d_src, d_tgt]

    # Reconstruction quality — always on UNWEIGHTED data
    Y_pred = X_unweighted @ W                        # [V, d_tgt]
    mse = F.mse_loss(Y_pred, Y_unweighted).item()
    cos_sim = F.cosine_similarity(Y_pred, Y_unweighted, dim=1)
    cos_mean = cos_sim.mean().item()
    cos_std = cos_sim.std().item()

    ss_res = ((Y_unweighted - Y_pred) ** 2).sum()
    ss_tot = ((Y_unweighted) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()

    metrics = {
        "mse": mse,
        "cos_sim_mean": cos_mean,
        "cos_sim_std": cos_std,
        "r2": r2,
        "reg_lambda": reg_lambda,
        "vocab_size": V,
        "method": method,
    }

    return W, src_mean, tgt_mean, metrics


@torch.no_grad()
def compute_mapping_from_models(src_model_name, tgt_model_name, reg_lambda=1e-3, device="cpu"):
    """High-level: load two Qwen models' embeddings and compute the mapping.

    Uses token-text intersection to handle slight tokenizer differences.

    Returns:
        W, src_mean, tgt_mean, metrics, (src_hidden, tgt_hidden)
    """
    src_aligned, tgt_aligned, n_pairs, n_skipped, src_hidden, tgt_hidden = \
        align_embeddings_by_token_text(src_model_name, tgt_model_name)

    print(f"Source embedding aligned: {src_aligned.shape}  (hidden={src_hidden})")
    print(f"Target embedding aligned: {tgt_aligned.shape}  (hidden={tgt_hidden})")
    if n_skipped > 0:
        print(f"Skipped {n_skipped} tokens with no target match")

    W, src_mean, tgt_mean, metrics = compute_mapping(src_aligned, tgt_aligned, reg_lambda)

    print(f"\nMapping quality on {metrics['vocab_size']} tokens:")
    print(f"  MSE:          {metrics['mse']:.6f}")
    print(f"  Cosine sim:   {metrics['cos_sim_mean']:.4f} ± {metrics['cos_sim_std']:.4f}")
    print(f"  R²:           {metrics['r2']:.4f}")

    return W, src_mean, tgt_mean, metrics, (src_hidden, tgt_hidden)


# ---------------------------------------------------------------------------
# Prompt & head projection
# ---------------------------------------------------------------------------

def project_prompt(src_prompt_emb, W, src_mean, tgt_mean):
    """Project a soft prompt from source to target embedding space.

    Args:
        src_prompt_emb: [L, d_src] or [B, L, d_src]
        W:              [d_src, d_tgt]
        src_mean:       [d_src]
        tgt_mean:       [d_tgt]

    Returns:
        projected: [* , d_tgt]
    """
    return (src_prompt_emb - src_mean.to(src_prompt_emb.device)) @ W.to(src_prompt_emb.device) \
           + tgt_mean.to(src_prompt_emb.device)


def project_head(src_head_weight, src_head_bias, W, src_mean=None, tgt_mean=None, reg=1e-3):
    """Project classification head weights from source to target space via pseudoinverse.

    Derivation: logit = h_tgt @ W_head_tgt^T ≈ h_src @ W_head_src^T = h_tgt @ W^+ @ W_head_src^T
    → W_head_tgt = W_head_src @ (W^+)^T

    Note: src_mean/tgt_mean are input-embedding space statistics. The classification head
    operates on final hidden states (different space), so centering offsets do NOT apply.
    Bias is passed through unchanged.
    """
    W_pinv = torch.linalg.pinv(W)   # [d_tgt, d_src] — maps target space back to source space
    tgt_head_weight = src_head_weight.float() @ W_pinv.T.to(src_head_weight.device)
    tgt_head_bias = src_head_bias.clone()

    return tgt_head_weight, tgt_head_bias


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def diagnose_prompt_transfer(src_prompt_emb, W, src_mean, tgt_mean, src_emb, tgt_emb, topk=5):
    """Check how well the prompt vectors fit the linear map vs regular token embeddings.

    Computes nearest-neighbor cosine similarity of prompt vectors to token embeddings
    in both source and (projected) target space. If the ratios are close to 1.0,
    the linear map generalizes well to the prompt region of the embedding space.

    Returns:
        dict with 'src_cos', 'tgt_cos', 'ratio'
    """
    # Source: nearest token embedding cosine similarity per prompt vector
    prompt_norm = F.normalize(src_prompt_emb, dim=1)           # [L, d_src]
    emb_norm = F.normalize(src_emb, dim=1)                     # [V, d_src]
    src_sim = prompt_norm @ emb_norm.T                         # [L, V]
    src_topk = src_sim.topk(topk, dim=1).values.mean(dim=1)    # [L]

    # Target: project prompt, then same check
    tgt_prompt = project_prompt(src_prompt_emb, W, src_mean, tgt_mean)
    tgt_prompt_norm = F.normalize(tgt_prompt, dim=1)           # [L, d_tgt]
    tgt_emb_norm = F.normalize(tgt_emb, dim=1)                 # [V, d_tgt]
    tgt_sim = tgt_prompt_norm @ tgt_emb_norm.T                 # [L, V]
    tgt_topk = tgt_sim.topk(topk, dim=1).values.mean(dim=1)    # [L]

    ratio = (tgt_topk / src_topk.clamp_min(1e-8)).mean().item()
    src_cos = src_topk.mean().item()
    tgt_cos = tgt_topk.mean().item()

    return {
        "src_nearest_cos": src_cos,
        "tgt_nearest_cos": tgt_cos,
        "ratio": ratio,
    }


# ---------------------------------------------------------------------------
# Full alignment pipeline
# ---------------------------------------------------------------------------

def align_and_save(
    src_model_name,
    tgt_model_name,
    src_prompt_path,
    output_dir,
    reg_lambda=1e-3,
    temperature=0.1,
    topk=5000,
    device="cpu",
):
    """End-to-end training-free alignment: load models → compute W → project & save.

    Uses prompt-weighted regression by default — anchors near the prompt in embedding
    space get higher weight, avoiding noise from 150K+ irrelevant rare-token embeddings.

    Returns:
        dict with paths to saved files and quality metrics.
    """
    import os
    from .utils import save_prompt, PromptQwenWrapper

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load embeddings + token intersection
    print(f"\n{'='*60}")
    print(f"Training-Free Cross-Model Alignment")
    print(f"Source: {src_model_name}")
    print(f"Target: {tgt_model_name}")
    print(f"Ridge λ: {reg_lambda}  |  Temp: {temperature}  |  TopK: {topk}")
    print(f"{'='*60}\n")

    src_aligned, tgt_aligned, n_pairs, n_skipped, src_hidden, tgt_hidden = \
        align_embeddings_by_token_text(src_model_name, tgt_model_name)

    print(f"Source full vocab: {n_pairs + n_skipped}  hidden_size={src_hidden}")
    print(f"Target hidden_size: {tgt_hidden}")
    print(f"Token intersection (by decoded text): {n_pairs} pairs")
    if n_skipped > 0:
        print(f"Skipped (no match in target): {n_skipped} tokens")

    # 2. Load source prompt (needed BEFORE computing mapping — for weighting)
    print(f"\n--- Loading source prompt ---")
    src_ckpt = torch.load(src_prompt_path, map_location="cpu")
    src_prompt_emb = src_ckpt["prompt_embeddings"]["weight"].clone()
    print(f"  Source prompt: {src_prompt_emb.shape}")
    print(f"  Value range:   [{src_prompt_emb.min().item():.4f}, {src_prompt_emb.max().item():.4f}]")

    # 3. Compute weighted mapping
    print(f"\n--- Computing weighted mapping (prompt-local) ---")
    W, src_mean, tgt_mean, metrics = compute_weighted_mapping(
        src_aligned, tgt_aligned, src_prompt_emb,
        temperature=temperature, topk=topk, reg_lambda=reg_lambda,
    )

    print(f"\nMapping quality ({metrics['vocab_size']} tokens, method={metrics['method']}):")
    print(f"  MSE:        {metrics['mse']:.6f}")
    print(f"  Cosine sim: {metrics['cos_sim_mean']:.4f} ± {metrics['cos_sim_std']:.4f}")
    print(f"  R²:         {metrics['r2']:.4f}")
    print(f"  W shape:    {list(W.shape)}")

    # 4. Project prompt
    print(f"\n--- Projecting prompt ---")
    projected_prompt = project_prompt(src_prompt_emb, W, src_mean, tgt_mean)
    print(f"  Projected prompt: {projected_prompt.shape}")
    print(f"  Value range:      [{projected_prompt.min().item():.4f}, {projected_prompt.max().item():.4f}]")

    # 3.5 Diagnostic: how well does linear map preserve prompt "neighborhood"?
    diag = diagnose_prompt_transfer(src_prompt_emb, W, src_mean, tgt_mean, src_aligned, tgt_aligned)
    print(f"\n--- Prompt transfer diagnostic ---")
    print(f"  Source prompt nearest-token cos-sim: {diag['src_nearest_cos']:.4f}")
    print(f"  Target prompt nearest-token cos-sim: {diag['tgt_nearest_cos']:.4f}")
    print(f"  Ratio (tgt/src): {diag['ratio']:.4f}  (≪1 = prompt may be out-of-distribution for linear map)")
    if diag['ratio'] < 0.5:
        print(f"  WARNING: ratio < 0.5 — the prompt region may not be well-aligned by a linear map alone.")
        print(f"  Consider: (1) smaller λ, (2) using more anchor tokens, (3) falling back to trained projector.")

    # Save raw projected prompt
    proj_prompt_path = os.path.join(output_dir, "projected_prompt_aligned.pt")
    torch.save({
        "prompt_embeddings": projected_prompt,
        "src_mean": src_mean,
        "tgt_mean": tgt_mean,
        "W": W,
        "metrics": metrics,
        "src_model": src_model_name,
        "tgt_model": tgt_model_name,
        "method": "ridge_regression_alignment",
    }, proj_prompt_path)
    print(f"  Saved: {proj_prompt_path}")

    # 4. Build target wrapper with projected prompt
    print(f"\n--- Building target model wrapper ---")
    tgt_wrapper = PromptQwenWrapper(tgt_model_name, prompt_len=src_prompt_emb.size(0), num_labels=NUM_LABELS)
    tgt_wrapper.to(device)
    with torch.no_grad():
        tgt_wrapper.prompt_embeddings.weight.copy_(projected_prompt.to(tgt_wrapper.prompt_embeddings.weight.device))

    # 5. Project classification head (if available in source checkpoint)
    if "classification_head" in src_ckpt:
        print(f"\n--- Projecting classification head ---")
        src_head_w = src_ckpt["classification_head"]["weight"].clone()  # [num_labels, d_src]
        src_head_b = src_ckpt["classification_head"]["bias"].clone()    # [num_labels]
        tgt_head_w, tgt_head_b = project_head(src_head_w, src_head_b, W, src_mean=src_mean, tgt_mean=tgt_mean)
        with torch.no_grad():
            tgt_wrapper.classification_head.weight.copy_(tgt_head_w.to(tgt_wrapper.classification_head.weight.device))
            tgt_wrapper.classification_head.bias.copy_(tgt_head_b.to(tgt_wrapper.classification_head.bias.device))
        print(f"  Source head: {src_head_w.shape} -> Target head: {tgt_head_w.shape}")
    else:
        print(f"\n--- No classification head in source checkpoint, target head stays random ---")

    # 6. Save full checkpoint (compatible format)
    aligned_path = os.path.join(output_dir, "prompt_aligned.pt")
    save_prompt(tgt_wrapper, aligned_path)
    print(f"  Saved: {aligned_path}")

    # Also save W and means for reproducibility
    mapping_path = os.path.join(output_dir, "alignment_mapping.pt")
    torch.save({
        "W": W,
        "src_mean": src_mean,
        "tgt_mean": tgt_mean,
        "src_hidden": src_hidden,
        "tgt_hidden": tgt_hidden,
        "metrics": metrics,
    }, mapping_path)
    print(f"  Saved: {mapping_path}")

    print(f"\n{'='*60}")
    print(f"Alignment complete.")
    print(f"  Projected prompt:  {proj_prompt_path}")
    print(f"  Aligned checkpoint: {aligned_path}")
    print(f"  Mapping:           {mapping_path}")
    print(f"{'='*60}\n")

    return {
        "proj_prompt_path": proj_prompt_path,
        "aligned_path": aligned_path,
        "mapping_path": mapping_path,
        "metrics": metrics,
        "W": W,
        "tgt_wrapper": tgt_wrapper,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training-free Qwen cross-model prompt alignment")
    parser.add_argument("--src", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--tgt", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--src-prompt", type=str, required=True, help="Path to source soft prompt .pt")
    parser.add_argument("--output-dir", type=str, default="outputs_qwen")
    parser.add_argument("--reg", type=float, default=1e-3, help="Ridge lambda")
    parser.add_argument("--device", type=str, default="cpu", help="Device for model loading")
    args = parser.parse_args()

    align_and_save(
        src_model_name=args.src,
        tgt_model_name=args.tgt,
        src_prompt_path=args.src_prompt,
        output_dir=args.output_dir,
        reg_lambda=args.reg,
        device=args.device,
    )
