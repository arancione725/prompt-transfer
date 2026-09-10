"""
Discrete Anchor Bridge: training-free cross-model prompt transfer via natural language.

Key insight: embedding spaces are model-specific, but natural language text is universal.

Algorithm:
  1. For each soft prompt vector, find its K nearest neighbors among source token embeddings
  2. Decode those tokens to TEXT (the universal bridge)
  3. Re-encode the text with the TARGET tokenizer
  4. Weighted average of the resulting target embeddings (by cosine similarity)

Variant (hard): decode top-1 token per position, join as text, re-encode.
Variant (soft): decode top-K tokens per position, weighted sum of their target embeddings.

This avoids the "extrapolation in embedding space" problem entirely — the prompt
vectors are quantized back to the discrete vocabulary, pushed through the text
bottleneck, and reconstructed in the target space.

Optimizations (2026-06-23):
  - Direct ID mapping: skip decode→encode round-trip for shared-vocab Qwen models
  - Local norm calibration: per-position weighted norm instead of global average
  - Dynamic temperature: adapt softmax sharpness per position based on confidence
  - Subset ridge: fit classification head projection on sentiment-relevant tokens only
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Sentiment seed vocabulary — used for subset ridge regression
# ---------------------------------------------------------------------------

SST2_SENTIMENT_SEEDS = [
    "positive", "negative", "good", "bad", "great", "terrible",
    "excellent", "awful", "wonderful", "horrible", "love", "hate",
    "liked", "disliked", "amazing", "boring", "fantastic", "poor",
    "best", "worst", "happy", "sad", "enjoy", "disappointed",
    "beautiful", "ugly", "fun", "dull", "brilliant",
    "nice", "nasty", "superb", "dreadful", "delightful", "miserable",
    "impressive", "pathetic", "lovely", "disgusting", "incredible",
    "lousy", "outstanding", "disappointing", "magnificent",
    "enjoyable", "tedious", "fabulous", "rotten", "splendid", "appalling",
    "charming", "vile", "pleasing", "atrocious", "awesome", "lame",
    "cool", "decent", "fine", "warm", "cold",
    "masterpiece", "garbage", "trash",
    "recommend", "avoid", "worth", "waste",
    # SST-2 label words
    "Positive", "Negative",
    "sentiment", "review", "movie", "film",
    # Common review modifiers
    "not", "very", "really", "so", "too", "quite", "pretty",
    "one", "of", "the", "most", "least", "ever", "never",
    "always", "sometimes", "rarely",
    # Punctuation / emphasis carriers
    "!", "?", "...",
]


# ---------------------------------------------------------------------------
# Fast token ID resolution (avoids decode→encode fragmentation)
# ---------------------------------------------------------------------------

def _get_min_shared_vocab(src_model_name, tgt_model_name):
    """Return the smaller vocab size — tokens below this are 1:1 across Qwen2.5 series."""
    src_tok = AutoTokenizer.from_pretrained(src_model_name)
    tgt_tok = AutoTokenizer.from_pretrained(tgt_model_name)
    return min(src_tok.vocab_size, tgt_tok.vocab_size)


def _resolve_target_ids(src_tid, src_tok, tgt_tok, min_shared_vocab, use_direct_id=True):
    """Resolve a source token ID to target token ID(s).

    When use_direct_id=True and src_tid < min_shared_vocab, returns [src_tid]
    directly — the Qwen2.5 series share the same tokenizer, so IDs are 1:1 for
    the shared vocabulary. This avoids the decode→encode round-trip that can
    fragment subword tokens (e.g. 'Ġing' → ' ing' → [' ', 'ing']).

    For tokens outside the shared range, falls back to text-based encode.
    """
    if use_direct_id and src_tid < min_shared_vocab:
        return [src_tid]
    text = src_tok.decode([src_tid])
    return tgt_tok.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------------------
# Attention-sink token masking (prevents EOS collapse in soft bridge)
# ---------------------------------------------------------------------------

def _get_sink_token_ids(tokenizer):
    """Return token IDs that act as attention sinks / structural anchors.

    In LLMs, certain tokens (<|endoftext|>, chat format markers, whitespace, etc.)
    serve as "attention sinks" — their hidden states absorb attention mass without
    carrying semantic content. When soft prompt vectors drift near these tokens
    in embedding space, the bridge picks them as nearest neighbors, producing
    semantically empty prompt text.

    Masking these tokens before top-K forces the bridge to choose tokens that carry
    actual linguistic meaning, even if they are geometrically slightly farther away.
    """
    sink_ids = set()

    # 1. All registered special tokens (eos, bos, pad, etc.)
    if tokenizer.all_special_ids:
        sink_ids.update(tokenizer.all_special_ids)

    # 2. Explicit structural / control tokens
    structural_texts = [
        "<|im_start|>", "<|im_end|>",
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
        "<|video_pad|>", "<|audio_pad|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
        "<|grounding|>",
        "\n", "\t",
    ]
    for text in structural_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        sink_ids.update(ids)

    return torch.tensor(sorted(sink_ids))


# ---------------------------------------------------------------------------
# Sentiment-aligned embedding subset
# ---------------------------------------------------------------------------

@torch.no_grad()
def _get_sentiment_aligned_subset(
    src_model_name, tgt_model_name, bridge_topk_ids=None, expand_k=5
):
    """Build aligned src/tgt embedding matrices filtered to sentiment-relevant tokens.

    1. Find single-token IDs in source vocab matching sentiment seed words
    2. Include bridge-activated tokens (the top-K tokens from the soft bridge)
    3. Expand by embedding similarity (expand_k nearest neighbors in src space)
    4. Align with target tokenizer by text matching
    5. Return filtered [N_subset, d_src] and [N_subset, d_tgt]

    Args:
        src_model_name:   source model name
        tgt_model_name:   target model name
        bridge_topk_ids:  [L, K] tensor of top-K source token IDs from the bridge
        expand_k:         if > 0, expand each seed token to its K nearest neighbors

    Returns:
        src_aligned: [N_subset, d_src]
        tgt_aligned: [N_subset, d_tgt]
    """
    from .align import _build_token_string_map, get_embedding_matrix

    src_tok = AutoTokenizer.from_pretrained(src_model_name)
    tgt_tok = AutoTokenizer.from_pretrained(tgt_model_name)

    # 1. Collect source token IDs for sentiment seeds (single-token encodings)
    src_sentiment_ids = set()
    for word in SST2_SENTIMENT_SEEDS:
        ids = src_tok.encode(word, add_special_tokens=False)
        for tid in ids:
            src_sentiment_ids.add(tid)

    # 2. Add bridge-activated tokens
    if bridge_topk_ids is not None:
        src_sentiment_ids.update(bridge_topk_ids.flatten().tolist())

    # 3. Optional: expand by embedding similarity
    if expand_k > 0:
        src_emb_full, _ = get_embedding_matrix(src_model_name)
        src_emb_norm = F.normalize(src_emb_full, dim=1)
        seed_ids = torch.tensor(list(src_sentiment_ids))
        seed_emb = src_emb_norm[seed_ids]                            # [|seeds|, d_src]
        sim = seed_emb @ src_emb_norm.T                              # [|seeds|, V]
        _, top_ids = sim.topk(expand_k + 1, dim=1)                   # +1: self included
        src_sentiment_ids.update(top_ids.flatten().tolist())

    # 4. Align with target tokenizer by text
    tgt_t2i = _build_token_string_map(tgt_tok)
    src_ids = []
    tgt_ids = []
    for sid in src_sentiment_ids:
        s_text = src_tok.decode([sid])
        tid = tgt_t2i.get(s_text)
        if tid is None:
            stripped = s_text.lstrip()
            tid = tgt_t2i.get(stripped)
        if tid is not None:
            src_ids.append(sid)
            tgt_ids.append(tid)

    # 5. Load embeddings for matched pairs
    src_emb_full, _ = get_embedding_matrix(src_model_name)
    tgt_emb_full, _ = get_embedding_matrix(tgt_model_name)
    src_aligned = src_emb_full[torch.tensor(src_ids)]
    tgt_aligned = tgt_emb_full[torch.tensor(tgt_ids)]

    print(f"  Sentiment subset: {len(src_ids)} aligned tokens "
          f"(from {len(src_sentiment_ids)} seeds, {len(src_sentiment_ids) - len(src_ids)} unmatched)")

    return src_aligned, tgt_aligned


# ---------------------------------------------------------------------------
# Core bridge functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_prompt_to_text(src_prompt_emb, src_model_name, topk=1):
    """Convert a soft prompt to a natural language string.

    Each prompt vector is matched to its nearest source token(s), decoded to text.

    Returns:
        text: the decoded string
        tokens: list of (token_id, token_text, cosine_similarity) per position
    """
    from .align import get_embedding_matrix

    src_tok = AutoTokenizer.from_pretrained(src_model_name)
    src_emb_full, _ = get_embedding_matrix(src_model_name)
    src_emb_norm = F.normalize(src_emb_full, dim=1)
    prompt_norm = F.normalize(src_prompt_emb.float(), dim=1)

    sim = prompt_norm @ src_emb_norm.T            # [L, V]
    topk_sim, topk_ids = sim.topk(topk, dim=1)    # [L, K], [L, K]

    tokens = []
    for i in range(src_prompt_emb.size(0)):
        for k in range(topk):
            tid = topk_ids[i, k].item()
            ttext = src_tok.decode([tid])
            tcos = topk_sim[i, k].item()
            tokens.append((tid, ttext, tcos))

    # Build text: for top-1, just join decoded tokens
    top1_texts = [src_tok.decode([topk_ids[i, 0].item()]) for i in range(src_prompt_emb.size(0))]
    text = "".join(top1_texts)

    return text, tokens


@torch.no_grad()
def soft_bridge_prompt(
    src_prompt_emb, src_model_name, tgt_model_name,
    topk=5, temperature=0.05,
    dynamic_temp=True,
    local_norm=True,
    use_direct_id=True,
    mask_sink_tokens=True,
    sampled_ids=None,
    prompt_weights=None,
    weight_mode="softmax",
    weight_temperature=0.5,
):
    """Soft discrete bridge: transfer prompt from source to target embedding space.

    Two modes:
    A) WITH sampled_ids + prompt_weights (SuperPos path — the correct approach):
       1. Constrain NN search to the m basis tokens only (not full 150K vocab)
       2. Use original prompt_weights as primary weights — no re-computed softmax
       3. Cosine similarity acts only as a per-position quality gate
       4. Decompose: top-K basis tokens by weight → bridge individually → recombine

    B) WITHOUT sampled_ids (legacy path — for standard soft prompts):
       Full-vocab NN search → softmax re-weighting → bridge.
       Only used when the prompt is a free vector (not a basis combination).

    Args:
        src_prompt_emb:   [L, d_src] mixed prompt embeddings
        src_model_name:    source model name
        tgt_model_name:    target model name
        topk:              number of candidate tokens per position
        temperature:       base softmax temperature (legacy path only)
        dynamic_temp:      adapt temperature per position (legacy path only)
        local_norm:        rescale each position to its weighted token norm
        use_direct_id:     use direct ID mapping for shared-vocab tokens
        mask_sink_tokens:  mask attention-sink tokens (legacy path only)
        sampled_ids:       [m] basis token IDs — when provided, enables SuperPos path
        prompt_weights:    [L, m] ACTIVATED weights — caller must apply softmax/relu first.
                           Only used when sampled_ids is provided.
        weight_mode:       "softmax" or "relu" — for diagnostics only
        weight_temperature: T value used to activate weights — for diagnostics only

    Returns:
        projected:   [prompt_len, d_tgt] target prompt embeddings
        diagnostics: dict
    """
    from .align import get_embedding_matrix

    src_tok = AutoTokenizer.from_pretrained(src_model_name)
    tgt_tok = AutoTokenizer.from_pretrained(tgt_model_name)
    src_emb_full, _ = get_embedding_matrix(src_model_name)
    tgt_emb_full, tgt_hidden = get_embedding_matrix(tgt_model_name)

    prompt_len = src_prompt_emb.size(0)
    prompt_norm = F.normalize(src_prompt_emb.float(), dim=1)

    # =========================================================================
    # SuperPos path: constrained search within basis tokens, original weights
    # =========================================================================
    if sampled_ids is not None and prompt_weights is not None:
        m = sampled_ids.size(0)

        # --- 1. Constrain search space to the m basis tokens only ---
        src_emb_sampled = src_emb_full[sampled_ids].float()                 # [m, d_src]
        src_emb_sampled_norm = F.normalize(src_emb_sampled, dim=1)           # [m, d_src]
        sim = prompt_norm @ src_emb_sampled_norm.T                          # [L, m]

        # --- 2. Select top-K basis tokens by ORIGINAL weight (not cosine sim) ---
        k = min(topk, m)
        if k < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        topk_vals, topk_idx = prompt_weights.topk(k, dim=1)                 # [L, K]
        # Top-K is a sparse approximation of the original convex mixture.
        # Renormalize the retained mass so changing K does not also change
        # the prompt scale.
        topk_weights = topk_vals / topk_vals.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # --- 3. Per-position cosine-similarity gate ---
        # Gate = max cosine sim between mixed vector and its selected basis tokens.
        # High → geometry is clean, weights transfer faithfully.
        # Low  → mixing drifted, dampen this position to prevent noise injection.
        gate_per_pos = torch.zeros(prompt_len)
        for i in range(prompt_len):
            gate_per_pos[i] = sim[i, topk_idx[i]].max()

        # --- 4. Per-position: bridge top-K basis tokens + recombine with original weights ---
        tgt_emb_sampled = tgt_emb_full[sampled_ids].float()                 # [m, d_tgt]
        min_shared_vocab = _get_min_shared_vocab(src_model_name, tgt_model_name) if use_direct_id else 0

        projected = torch.zeros(prompt_len, tgt_hidden)
        diag_texts = []

        for i in range(prompt_len):
            gate = gate_per_pos[i].item()
            pos_vec = torch.zeros(tgt_hidden)
            expected_norm = 0.0
            texts_for_pos = []

            for k_idx in range(k):
                basis_idx = topk_idx[i, k_idx].item()       # index into [0, m-1]
                w = topk_weights[i, k_idx].item()           # renormalized weight
                if w <= 0:
                    continue

                src_tid = sampled_ids[basis_idx].item()
                text = src_tok.decode([src_tid])

                # Bridge this basis token to target space
                tgt_ids = _resolve_target_ids(src_tid, src_tok, tgt_tok, min_shared_vocab, use_direct_id)
                if len(tgt_ids) == 0:
                    continue

                tok_emb = tgt_emb_full[torch.tensor(tgt_ids)].mean(dim=0)  # [d_tgt]
                pos_vec += w * tok_emb

                if local_norm:
                    expected_norm += w * tok_emb.norm(p=2, dim=-1)

                texts_for_pos.append(f"{text}(w={w:.3f})")

            # --- Apply cosine-similarity gate ---
            pos_vec = pos_vec * gate

            # --- Local norm calibration (on gated vector) ---
            if local_norm and expected_norm > 0:
                actual_norm = pos_vec.norm(p=2, dim=-1).clamp_min(1e-8)
                pos_vec = pos_vec / actual_norm * expected_norm * gate

            projected[i] = pos_vec
            diag_texts.append(" | ".join(texts_for_pos[:3]))

        diagnostics = {
            "mode": "superpos",
            "topk": min(topk, m),
            "weight_mode": weight_mode,
            "weight_temperature": weight_temperature,
            "local_norm": local_norm,
            "use_direct_id": use_direct_id,
            "gate_mean": gate_per_pos.mean().item(),
            "gate_min": gate_per_pos.min().item(),
            "gate_max": gate_per_pos.max().item(),
            "mean_top1_sim": sim.max(dim=1).values.mean().item(),
            "projected_norm": projected.norm(p=2, dim=-1).mean().item(),
            "sample_texts": diag_texts[:5],
        }

        return projected, diagnostics

    # =========================================================================
    # Legacy path: full-vocab NN search → softmax re-weighting
    # (for standard soft prompts that are free vectors, not basis combinations)
    # =========================================================================

    src_emb_norm = F.normalize(src_emb_full, dim=1)

    sim = prompt_norm @ src_emb_norm.T            # [L, V_src]

    if mask_sink_tokens:
        sink_ids = _get_sink_token_ids(src_tok)
        sim[:, sink_ids] = -float('inf')

    topk_sim, topk_ids = sim.topk(topk, dim=1)    # [L, K], [L, K]

    if dynamic_temp:
        std_sim = topk_sim.std(dim=1, keepdim=True) + 1e-6
        effective_temp = temperature / (std_sim * 10)
    else:
        effective_temp = temperature

    weights = torch.softmax(topk_sim / effective_temp, dim=1)  # [L, K]

    min_shared_vocab = _get_min_shared_vocab(src_model_name, tgt_model_name) if use_direct_id else 0

    projected = torch.zeros(prompt_len, tgt_hidden)
    diag_texts = []

    for i in range(prompt_len):
        pos_vec = torch.zeros(tgt_hidden)
        expected_norm = 0.0
        texts_for_pos = []
        for k in range(topk):
            tid = topk_ids[i, k].item()
            w = weights[i, k].item()
            text = src_tok.decode([tid])

            tgt_ids = _resolve_target_ids(tid, src_tok, tgt_tok, min_shared_vocab, use_direct_id)
            if len(tgt_ids) == 0:
                continue

            tgt_emb = tgt_emb_full[torch.tensor(tgt_ids)].mean(dim=0)
            pos_vec += w * tgt_emb

            if local_norm:
                expected_norm += w * tgt_emb.norm(p=2, dim=-1)

            texts_for_pos.append(f"{text}({w:.3f})")

        if local_norm and expected_norm > 0:
            actual_norm = pos_vec.norm(p=2, dim=-1).clamp_min(1e-8)
            pos_vec = pos_vec / actual_norm * expected_norm

        projected[i] = pos_vec
        diag_texts.append(" | ".join(texts_for_pos[:3]))

    if not local_norm:
        target_norm = tgt_emb_full.norm(p=2, dim=-1).mean()
        prompt_norm_val = projected.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        projected = projected / prompt_norm_val * target_norm

    diagnostics = {
        "mode": "legacy",
        "topk": topk,
        "temperature": temperature,
        "dynamic_temp": dynamic_temp,
        "local_norm": local_norm,
        "use_direct_id": use_direct_id,
        "mean_top1_sim": topk_sim[:, 0].mean().item(),
        "mean_topk_sim": topk_sim.mean().item(),
        "projected_norm": projected.norm(p=2, dim=-1).mean().item(),
        "sample_texts": diag_texts[:5],
    }

    return projected, diagnostics


@torch.no_grad()
def hard_bridge_prompt(src_prompt_emb, src_model_name, tgt_model_name, use_direct_id=True):
    """Hard discrete bridge: top-1 token per position, join as text, re-encode.

    When use_direct_id=True, skips decode→re-encode for shared-vocab tokens and
    directly concatenates their target embeddings. This avoids the subword
    fragmentation problem.

    Returns:
        projected: [L', d_tgt] — L' may differ from original prompt_len
        text: the decoded prompt text
        diagnostics: dict
    """
    from .align import get_embedding_matrix

    src_tok = AutoTokenizer.from_pretrained(src_model_name)
    tgt_tok = AutoTokenizer.from_pretrained(tgt_model_name)
    src_emb_full, _ = get_embedding_matrix(src_model_name)
    tgt_emb_full, tgt_hidden = get_embedding_matrix(tgt_model_name)

    src_emb_norm = F.normalize(src_emb_full, dim=1)
    prompt_norm = F.normalize(src_prompt_emb.float(), dim=1)

    sim = prompt_norm @ src_emb_norm.T            # [L, V_src]
    _, top1_ids = sim.topk(1, dim=1)              # [L, 1]

    # Decode prompt to text
    top1_tokens = [src_tok.decode([top1_ids[i, 0].item()]) for i in range(src_prompt_emb.size(0))]
    text = "".join(top1_tokens)

    # Direct ID path: each src token → its target embedding (no text round-trip)
    if use_direct_id:
        min_shared_vocab = _get_min_shared_vocab(src_model_name, tgt_model_name)
        tgt_ids_all = []
        fragment_count = 0
        for i in range(src_prompt_emb.size(0)):
            tid = top1_ids[i, 0].item()
            resolved = _resolve_target_ids(tid, src_tok, tgt_tok, min_shared_vocab, use_direct_id=True)
            if len(resolved) == 1:
                tgt_ids_all.append(resolved[0])
            elif len(resolved) > 1:
                tgt_ids_all.extend(resolved)
                fragment_count += 1
        if fragment_count:
            print(f"  Hard bridge (direct): {fragment_count}/{src_prompt_emb.size(0)} positions fragmented → {len(tgt_ids_all)} target tokens")
        else:
            print(f"  Hard bridge (direct): {src_prompt_emb.size(0)} tokens → {len(tgt_ids_all)} target tokens (0 fragmented)")

        # Truncate/pad
        prompt_len = src_prompt_emb.size(0)
        if len(tgt_ids_all) >= prompt_len:
            tgt_ids_all = tgt_ids_all[:prompt_len]
        else:
            pad_id = tgt_tok.eos_token_id or 0
            tgt_ids_all = tgt_ids_all + [pad_id] * (prompt_len - len(tgt_ids_all))
        projected = tgt_emb_full[torch.tensor(tgt_ids_all)]
    else:
        # Original decode→join→re-encode path
        tgt_ids = tgt_tok.encode(text, add_special_tokens=False)
        print(f"  Hard bridge: {src_prompt_emb.size(0)} prompt tokens -> \"{text[:80]}...\" -> {len(tgt_ids)} target tokens")

        prompt_len = src_prompt_emb.size(0)
        if len(tgt_ids) >= prompt_len:
            tgt_ids = tgt_ids[:prompt_len]
        else:
            pad_id = tgt_tok.eos_token_id or 0
            tgt_ids = tgt_ids + [pad_id] * (prompt_len - len(tgt_ids))
        projected = tgt_emb_full[torch.tensor(tgt_ids)]
        print(f"  Hard bridge text: \"{text[:200]}\"")

    diagnostics = {
        "decoded_text": text,
        "target_tokens": projected.size(0),
        "mean_top1_sim": sim.topk(1, dim=1).values.mean().item(),
    }

    return projected, diagnostics


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def bridge_align_and_save(
    src_model_name,
    tgt_model_name,
    src_prompt_path,
    output_dir,
    mode="soft",
    topk=5,
    temperature=0.05,
    dynamic_temp=True,
    local_norm=True,
    use_direct_id=True,
    mask_sink_tokens=True,
    head_mode="full",
    device="cpu",
):
    """Full pipeline: discrete bridge → project → save.

    Args:
        mode:             "soft" = weighted average of top-K, "hard" = direct text encoding
        dynamic_temp:     adapt softmax temperature per position
        local_norm:       per-position weighted norm calibration
        use_direct_id:    skip decode→encode round-trip for shared-vocab tokens
        mask_sink_tokens: mask attention-sink tokens before top-K search
        head_mode:        "full" = ridge on all 151K tokens; "subset" = ridge on sentiment
                          tokens + bridge-activated tokens only
    """
    import os
    from .utils import save_prompt, PromptQwenWrapper
    from .config import NUM_LABELS

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Discrete Anchor Bridge (Training-Free)")
    print(f"Source: {src_model_name}")
    print(f"Target: {tgt_model_name}")
    print(f"Mode: {mode}  |  TopK: {topk}  |  Temp: {temperature}")
    print(f"Dynamic temp: {dynamic_temp}  |  Local norm: {local_norm}  |  Direct ID: {use_direct_id}  |  Sink mask: {mask_sink_tokens}")
    print(f"Head mode: {head_mode}")
    print(f"{'='*60}\n")

    # 1. Load source prompt
    print("--- Loading source prompt ---")
    src_ckpt = torch.load(src_prompt_path, map_location="cpu")
    src_prompt_emb = src_ckpt["prompt_embeddings"]["weight"].clone()
    print(f"  Shape: {src_prompt_emb.shape}")

    # 2. Bridge
    print(f"\n--- {mode.capitalize()} bridging ---")
    topk_ids_for_head = None
    if mode == "soft":
        projected, diag = soft_bridge_prompt(
            src_prompt_emb, src_model_name, tgt_model_name,
            topk=topk, temperature=temperature,
            dynamic_temp=dynamic_temp,
            local_norm=local_norm,
            use_direct_id=use_direct_id,
            mask_sink_tokens=mask_sink_tokens,
        )
        # Precompute top-K activated token IDs for subset ridge
        # Must apply the same sink mask to keep head training consistent with bridge output
        if head_mode == "subset":
            from .align import get_embedding_matrix
            src_emb_full, _ = get_embedding_matrix(src_model_name)
            src_emb_norm = F.normalize(src_emb_full, dim=1)
            prompt_norm = F.normalize(src_prompt_emb.float(), dim=1)
            sim = prompt_norm @ src_emb_norm.T
            if mask_sink_tokens:
                src_tok = AutoTokenizer.from_pretrained(src_model_name)
                sink_ids = _get_sink_token_ids(src_tok)
                sim[:, sink_ids] = -float('inf')
            _, topk_ids_for_head = sim.topk(topk, dim=1)  # [L, K]
    elif mode == "seq":
        raise NotImplementedError("seq_bridge_prompt is not yet implemented")
    else:
        projected, diag = hard_bridge_prompt(
            src_prompt_emb, src_model_name, tgt_model_name,
            use_direct_id=use_direct_id,
        )

    print(f"  Projected shape: {projected.shape}")
    print(f"  Top-1 cos-sim:   {diag.get('mean_top1_sim', diag.get('mean_topk_sim', 0)):.4f}")
    if "sample_texts" in diag:
        print(f"  Sample (first 5 positions):")
        for t in diag["sample_texts"]:
            print(f"    {t}")

    # 3. Build target wrapper
    print(f"\n--- Building target model wrapper ---")
    tgt_wrapper = PromptQwenWrapper(
        tgt_model_name, prompt_len=src_prompt_emb.size(0), num_labels=NUM_LABELS
    )
    tgt_wrapper.to(device)
    with torch.no_grad():
        tgt_wrapper.prompt_embeddings.weight.copy_(
            projected.to(tgt_wrapper.prompt_embeddings.weight.device)
        )

    # 4. Project classification head
    if "classification_head" in src_ckpt:
        print(f"\n--- Projecting classification head (head_mode={head_mode}) ---")
        from .align import compute_mapping, project_head

        if head_mode == "subset":
            src_aligned, tgt_aligned = _get_sentiment_aligned_subset(
                src_model_name, tgt_model_name,
                bridge_topk_ids=topk_ids_for_head,
                expand_k=5,
            )
        else:
            from .align import align_embeddings_by_token_text
            src_aligned, tgt_aligned, _, _, _, _ = align_embeddings_by_token_text(
                src_model_name, tgt_model_name
            )

        W, src_mean, tgt_mean, metrics = compute_mapping(src_aligned, tgt_aligned, reg_lambda=1e-3)
        print(f"  Ridge on {metrics['vocab_size']} tokens: "
              f"cos_sim={metrics['cos_sim_mean']:.4f}  R²={metrics['r2']:.4f}")

        src_head_w = src_ckpt["classification_head"]["weight"].clone()
        src_head_b = src_ckpt["classification_head"]["bias"].clone()
        tgt_head_w, tgt_head_b = project_head(src_head_w, src_head_b, W, src_mean=src_mean, tgt_mean=tgt_mean)
        with torch.no_grad():
            tgt_wrapper.classification_head.weight.copy_(tgt_head_w.to(device))
            tgt_wrapper.classification_head.bias.copy_(tgt_head_b.to(device))

    # 5. Save
    fname = f"prompt_bridge_{mode}.pt"
    aligned_path = os.path.join(output_dir, fname)
    save_prompt(tgt_wrapper, aligned_path)
    print(f"\n  Saved: {aligned_path}")

    diag_path = os.path.join(output_dir, f"bridge_{mode}_diag.pt")
    torch.save(diag, diag_path)
    print(f"  Saved: {diag_path}")
    print(f"\n{'='*60}\n")

    return {"aligned_path": aligned_path, "diag_path": diag_path, "diagnostics": diag}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Discrete Anchor Bridge — training-free cross-model prompt transfer"
    )
    parser.add_argument("--src", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--tgt", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--src-prompt", required=True)
    parser.add_argument("--output-dir", default="outputs_qwen")
    parser.add_argument("--mode", default="soft", choices=["soft", "hard"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--no-dynamic-temp", action="store_true",
                        help="Disable dynamic temperature (use fixed --temp)")
    parser.add_argument("--no-local-norm", action="store_true",
                        help="Disable local norm calibration (use global average)")
    parser.add_argument("--no-direct-id", action="store_true",
                        help="Disable direct ID mapping (use decode→encode)")
    parser.add_argument("--no-sink-mask", action="store_true",
                        help="Disable sink token masking (allow EOS as bridge token)")
    parser.add_argument("--head-mode", default="full", choices=["full", "subset"],
                        help="Token set for classification head ridge projection")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    bridge_align_and_save(
        src_model_name=args.src,
        tgt_model_name=args.tgt,
        src_prompt_path=args.src_prompt,
        output_dir=args.output_dir,
        mode=args.mode,
        topk=args.topk,
        temperature=args.temp,
        dynamic_temp=not args.no_dynamic_temp,
        local_norm=not args.no_local_norm,
        use_direct_id=not args.no_direct_id,
        mask_sink_tokens=not args.no_sink_mask,
        head_mode=args.head_mode,
        device=args.device,
    )
