"""Labeled source/evaluation data. Refinement probe loading stays in rfa.py."""

import torch
from torch.utils.data import DataLoader, Dataset

from .config import MAX_LENGTH
from .rfa import load_sst2_cached_first


class ClassificationDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=MAX_LENGTH):
        self.rows = []
        for row in rows:
            text = row.get("sentence") or row.get("text") or ""
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
                    "label": torch.tensor(int(row["label"]), dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def load_classification_data(name, tokenizer, max_length=MAX_LENGTH):
    if name != "sst2":
        raise ValueError(f"Unsupported labeled dataset: {name}")
    raw = load_sst2_cached_first()
    return (
        ClassificationDataset(raw["train"], tokenizer, max_length),
        ClassificationDataset(raw["validation"], tokenizer, max_length),
    )


def build_loader(dataset, batch_size, shuffle=False, seed=None, sampler=None):
    if sampler is not None:
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
