"""Static/toy tests for fixed-support K sweep; no downloads required."""

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

import torch
import torch.nn as nn

from qwen_cross_model.refine_target import (
    build_topk_support,
    prompt_from_topk_coordinates,
    refine_topk_from_cache,
)
from qwen_cross_model.superpos import SuperPosCheckpoint, prompt_from_coordinates
from qwen_cross_model.sweep_k import (
    assign_k_values,
    collect_and_select,
    _subprocess_module_name,
    select_one_se,
    split_probe_texts,
)


class ToyBase(nn.Module):
    def forward(self, inputs_embeds, attention_mask, **unused):
        masked = inputs_embeds * attention_mask.unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=masked.cumsum(dim=1))


class ToyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(
            torch.tensor([
                [0.0, 0.0], [1.0, 0.2], [0.1, 1.0], [-0.5, 0.4]
            ]),
            freeze=True,
        )
        self.model = ToyBase()
        self.lm_head = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.tensor([
                    [1.0, 0.0], [0.0, 1.0], [-0.8, 0.3], [0.2, -0.7]
                ])
            )
        self.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embeddings

    def get_output_embeddings(self):
        return self.lm_head


def make_checkpoint(path):
    logits = torch.tensor([
        [3.0, 1.0, 2.0],
        [-1.0, 4.0, 0.5],
    ])
    sampled_ids = torch.tensor([1, 2, 3])
    path.write_bytes(b"toy checkpoint hash input")
    return SuperPosCheckpoint(
        path=path,
        payload={"source_model": "toy"},
        prompt_logits=logits,
        sampled_ids=sampled_ids,
        temperature=0.5,
        seed=42,
        source_model="toy",
        position_id_mode="physical_sequence",
    )


class TopKSweepTests(unittest.TestCase):
    def test_subprocess_inherits_parent_package_prefix(self):
        self.assertEqual(
            _subprocess_module_name("qwen_transfer.qwen_cross_model"),
            "qwen_transfer.qwen_cross_model.sweep_k",
        )

    def test_support_is_per_position_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = make_checkpoint(Path(directory) / "source.pt")
            support = build_topk_support(checkpoint, 2)
        torch.testing.assert_close(
            support.basis_indices, torch.tensor([[0, 2], [1, 2]])
        )
        torch.testing.assert_close(
            support.token_ids, torch.tensor([[1, 3], [2, 3]])
        )
        torch.testing.assert_close(support.source_coordinates.sum(-1), torch.ones(2))

    def test_full_k_reproduces_existing_direct_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = make_checkpoint(Path(directory) / "source.pt")
            support = build_topk_support(checkpoint, 3)
        embedding = ToyCausalLM().get_input_embeddings().weight
        expected = prompt_from_coordinates(
            checkpoint.coordinates, checkpoint.sampled_ids, embedding
        )
        actual = prompt_from_topk_coordinates(
            support.source_coordinates, support.token_ids, embedding
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_k_one_is_measured_without_optimization(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = make_checkpoint(Path(directory) / "source.pt")
            model = ToyCausalLM()
            split = {
                "input_ids": torch.tensor([[0, 1, 2], [0, 2, 1]]),
                "attention_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
                "source_effect": torch.tensor([
                    [0.7, -0.4, 0.2, -0.1],
                    [0.2, 0.1, -0.3, 0.5],
                ]),
                "target_reference": torch.zeros(2, 4),
                "anchor_plan": None,
            }
            cache = {
                "token_ids": torch.arange(4),
                "rfa_temperature": 2.0,
                "target_model": "toy-target",
                "target_model_revision": None,
                "train": split,
                "validation": split,
            }
            payload, result, history = refine_topk_from_cache(
                model, checkpoint, cache, basis_k=1, epochs=5, batch_size=1,
                learning_rate=0.1, stay_weight=1.0, cache_sha256="abc",
            )
        self.assertFalse(result["optimized"])
        self.assertEqual(len(history), 1)
        torch.testing.assert_close(
            payload["source_topk_coordinates"], payload["target_topk_coordinates"]
        )
        torch.testing.assert_close(
            payload["direct_target_prompt"], payload["refined_target_prompt"]
        )

    def test_k_two_refines_only_the_recorded_fixed_support(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = make_checkpoint(Path(directory) / "source.pt")
            model = ToyCausalLM()
            split = {
                "input_ids": torch.tensor([[0, 1, 2], [0, 2, 1]]),
                "attention_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
                "source_effect": torch.tensor([
                    [0.7, -0.4, 0.2, -0.1],
                    [0.2, 0.1, -0.3, 0.5],
                ]),
                "target_reference": torch.zeros(2, 4),
                "anchor_plan": None,
            }
            cache = {
                "token_ids": torch.arange(4),
                "rfa_temperature": 2.0,
                "target_model": "toy-target",
                "target_model_revision": None,
                "train": split,
                "validation": split,
            }
            expected_support = build_topk_support(checkpoint, 2).token_ids
            payload, result, history = refine_topk_from_cache(
                model, checkpoint, cache, basis_k=2, epochs=1, batch_size=1,
                learning_rate=0.1, stay_weight=1.0, cache_sha256="abc",
            )
        self.assertTrue(result["optimized"])
        self.assertEqual(len(history), 2)
        self.assertEqual(tuple(payload["target_active_logits"].shape), (2, 2))
        torch.testing.assert_close(payload["support_token_ids"], expected_support)

    def test_probe_split_is_deterministic_and_disjoint(self):
        texts = [f"probe-{index}" for index in range(30)]
        first = split_probe_texts(texts, 7, 11, 42, "toy")
        second = split_probe_texts(texts, 7, 11, 42, "toy")
        self.assertEqual(first[0].indices, second[0].indices)
        self.assertEqual(first[1].indices, second[1].indices)
        self.assertTrue(set(first[0].indices).isdisjoint(first[1].indices))

    def test_probe_split_cannot_leak_duplicate_text_across_partitions(self):
        texts = ["duplicate", "duplicate"] + [f"unique-{index}" for index in range(20)]
        train, validation = split_probe_texts(texts, 7, 10, 42, "toy")
        self.assertTrue(set(train.texts).isdisjoint(validation.texts))
        self.assertEqual(len(train.texts), len(set(train.texts)))
        self.assertEqual(len(validation.texts), len(set(validation.texts)))

    def test_round_robin_assignments_have_no_duplicates_or_omissions(self):
        values = [1, 2, 3, 4, 5, 8, 10]
        assignments = assign_k_values(values, 3)
        flattened = [value for group in assignments for value in group]
        self.assertEqual(sorted(flattened), sorted(values))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_one_se_selects_smallest_eligible_k(self):
        rows = [
            {"basis_k": 4, "refined_validation_rfa": {"mean": 0.104, "se": 0.001}},
            {"basis_k": 8, "refined_validation_rfa": {"mean": 0.100, "se": 0.006}},
            {"basis_k": 16, "refined_validation_rfa": {"mean": 0.101, "se": 0.002}},
        ]
        selection = select_one_se(rows)
        self.assertEqual(selection["minimum_mean_k"], 8)
        self.assertEqual(selection["selected_k"], 4)

    def test_selection_rejects_posthoc_metrics(self):
        rows = [{
            "basis_k": 4,
            "refined_validation_rfa": {"mean": 0.1, "se": 0.01},
            "accuracy": 0.9,
        }]
        with self.assertRaisesRegex(ValueError, "forbidden key"):
            select_one_se(rows)

    def test_collect_and_select_writes_label_free_reports(self):
        def summary(mean, se=0.01):
            return {"mean": mean, "se": se}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for basis_k, value in ((4, 0.104), (8, 0.100)):
                k_dir = root / f"k_{basis_k:03d}"
                k_dir.mkdir()
                result = {
                    "basis_k": basis_k,
                    "direct_train_rfa": summary(value + 0.02),
                    "direct_validation_rfa": summary(value + 0.02),
                    "refined_train_rfa": summary(value),
                    "refined_validation_rfa": summary(value, 0.006),
                    "optimized": True,
                    "cache_sha256": "cache-hash",
                    "refinement_config": {"epochs": 1},
                }
                (k_dir / "rfa_result.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )
                torch.save(
                    {
                        "type": "rfa_refined_target_prompt_topk",
                        "basis_k": basis_k,
                        "cache_sha256": "cache-hash",
                    },
                    k_dir / f"refined_target_k{basis_k:03d}.pt",
                )
            selection, rows = collect_and_select(root, [4, 8], "cache-hash")
            report = json.loads((root / "k_sweep_rfa.json").read_text())
        self.assertEqual(selection["selected_k"], 4)
        self.assertEqual(len(rows), 2)
        self.assertNotIn("accuracy", json.dumps(report).lower())


if __name__ == "__main__":
    unittest.main()
