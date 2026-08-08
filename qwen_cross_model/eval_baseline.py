#!/usr/bin/env python3
"""Baseline: evaluate raw Qwen models on SST-2 via LM Head fill-in-the-blank.

No soft prompts, no training — just a natural language instruction prefix.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="LM Head baseline for SST-2")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pos-word", default="positive")
    parser.add_argument("--neg-word", default="negative")
    parser.add_argument("--prompt-prefix", default="Classify the sentiment of this movie review as positive or negative. The sentiment is",
                        help="Natural language instruction prepended before the review text")
    args = parser.parse_args()

    print(f"LM Head Baseline")
    print(f"Model:     {args.model}")
    print(f"Prefix:    \"{args.prompt_prefix}\"")
    print(f"Pos/Neg:   '{args.pos_word}' / '{args.neg_word}'")

    # --- 1. Load model ---
    print("\n[1/3] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device
    )
    model.eval()

    # --- 2. Setup tokenizer + pos/neg IDs ---
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    pos_ids = set()
    neg_ids = set()
    for prefix in ["", " "]:
        pos_ids.update(tokenizer.encode(prefix + args.pos_word, add_special_tokens=False))
        neg_ids.update(tokenizer.encode(prefix + args.neg_word, add_special_tokens=False))
    pos_ids = sorted(pos_ids)
    neg_ids = sorted(neg_ids)
    print(f"  Positive tokens: {pos_ids}  → {[tokenizer.decode([i]) for i in pos_ids]}")
    print(f"  Negative tokens: {neg_ids}  → {[tokenizer.decode([i]) for i in neg_ids]}")

    # Encode the fixed prefix once
    prefix_ids = tokenizer.encode(args.prompt_prefix, add_special_tokens=False)
    prefix_len = len(prefix_ids)
    print(f"  Prefix length: {prefix_len} tokens")

    # --- 3. Load SST-2 ---
    print("[2/3] Loading SST-2 dataset...")
    from datasets import load_dataset
    raw = load_dataset("glue", "sst2")

    def tokenize(examples):
        texts = [f"{t}" for t in examples["sentence"]]
        # Concatenate prefix + " " + review
        full_texts = [f"{tokenizer.decode(prefix_ids)} {t}" for t in texts]
        enc = tokenizer(full_texts, truncation=True, max_length=256, padding=False, return_tensors=None)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": examples["label"],
        }

    eval_ds = raw["validation"].map(tokenize, batched=True, remove_columns=raw["validation"].column_names)

    from torch.utils.data import DataLoader

    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append([tokenizer.pad_token_id] * pad_len + x["input_ids"])
            attention_mask.append([0] * pad_len + x["attention_mask"])
            labels.append(x["label"])
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "label": torch.tensor(labels),
        }

    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # --- 4. Evaluate ---
    print("[3/3] Evaluating...")
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Eval"):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["label"].to(args.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_logits = outputs.logits[:, -1, :].float()

        pos_score = next_logits[:, pos_ids].max(dim=1).values
        neg_score = next_logits[:, neg_ids].max(dim=1).values
        preds = (pos_score > neg_score).long()
        correct += (preds == labels).sum().item()
        total += input_ids.size(0)

    acc = correct / total
    print(f"\n{args.model} direct (no prompt): {acc:.4f}")
    return acc


if __name__ == "__main__":
    main()
