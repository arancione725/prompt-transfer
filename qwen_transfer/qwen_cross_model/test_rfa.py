"""Offline tests for the standalone RFA pipeline."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from qwen_cross_model import analyze_rfa
from qwen_cross_model import rfa


class FakeTokenizer:
    def __init__(self, tokens=None, special_ids=None):
        self.tokens = tokens or ["<pad>", "a", "b", "c"]
        self.all_special_ids = list(special_ids or [])
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "left"

    def __len__(self):
        return len(self.tokens)

    def convert_ids_to_tokens(self, token_ids):
        if isinstance(token_ids, list):
            return [self.tokens[index] for index in token_ids]
        return self.tokens[token_ids]

    def __call__(self, text, **kwargs):
        length = kwargs["max_length"]
        values = [min(ord(character), len(self.tokens) - 1) for character in text[:length]]
        values = [0] * (length - len(values)) + values
        return {
            "input_ids": torch.tensor([values]),
            "attention_mask": torch.tensor([[int(value != 0) for value in values]]),
        }


class FakeEmbeddings:
    def __init__(self):
        self.weight = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        )


class FakeModel:
    def __init__(self, name):
        self.name = name
        self.embeddings = FakeEmbeddings()

    def get_input_embeddings(self):
        return self.embeddings

    def eval(self):
        return self

    def requires_grad_(self, unused):
        return self


class RFATests(unittest.TestCase):
    def test_sst2_loader_uses_local_cache_before_network(self):
        calls = []

        def load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return "cached-sst2"

        fake_datasets = SimpleNamespace(
            DownloadConfig=lambda **kwargs: kwargs,
            load_dataset=load_dataset,
        )
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            result = rfa.load_sst2_cached_first("train")
        self.assertEqual(result, "cached-sst2")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][1]["download_config"]["local_files_only"])
        self.assertEqual(calls[0][1]["download_config"]["max_retries"], 0)

    def test_coordinate_kl_is_zero_at_initialization_and_has_gradient(self):
        from qwen_cross_model.rfa import coordinate_kl

        source_logits = torch.tensor([[1.0, 0.0, -1.0], [0.2, 0.4, 0.1]])
        source = torch.softmax(source_logits / 0.5, dim=-1)
        target_logits = source_logits.clone().requires_grad_(True)
        target = torch.softmax(target_logits / 0.5, dim=-1)
        initial = coordinate_kl(target, source)
        self.assertAlmostEqual(float(initial.detach()), 0.0, places=6)
        changed = torch.softmax((target_logits + torch.tensor([[0.4, 0.0, 0.0], [0.0, 0.0, 0.3]])) / 0.5, dim=-1)
        loss = coordinate_kl(changed, source)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertIsNotNone(target_logits.grad)
        self.assertTrue(torch.isfinite(target_logits.grad).all())

    def test_rfa_loss_backpropagates_to_target_effect_only(self):
        source = torch.tensor([[0.2, -0.1, 0.7, 0.0]])
        target = torch.tensor([[0.1, 0.3, -0.2, 0.0]], requires_grad=True)
        loss = rfa.rowwise_rfa(source, target, temperature=2.0).mean()
        loss.backward()
        self.assertIsNotNone(target.grad)
        self.assertGreater(float(target.grad.abs().sum()), 0.0)

    def test_activation_modes_and_temperature_metadata(self):
        base = {
            "prompt_weights": torch.tensor([[2.0, -1.0]]),
            "sampled_ids": torch.tensor([1, 2]),
        }
        with self.assertRaisesRegex(ValueError, "no temperature"):
            rfa.activate_weights(base)
        weights, ids, mode, temperature = rfa.activate_weights(
            dict(base, temperature=0.5)
        )
        self.assertEqual(mode, "softmax")
        self.assertEqual(temperature, 0.5)
        self.assertEqual(ids.tolist(), [1, 2])
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(1))

        raw, _, mode, temperature = rfa.activate_weights(
            dict(base, weight_mode="raw")
        )
        torch.testing.assert_close(raw, base["prompt_weights"])
        self.assertEqual(mode, "raw")
        self.assertIsNone(temperature)

        normalized, _, mode, _ = rfa.activate_weights(
            dict(base, weight_mode="l1_normalized_raw")
        )
        torch.testing.assert_close(normalized.abs().sum(dim=1), torch.ones(1))
        self.assertEqual(mode, "l1_normalized_raw")

    def test_project_prompt_topk_rules(self):
        basis = torch.eye(3)
        softmax_weights = torch.tensor([[0.6, 0.3, 0.1]])
        projected = rfa.project_prompt(softmax_weights, basis, 2, "softmax")
        torch.testing.assert_close(projected, torch.tensor([[2 / 3, 1 / 3, 0.0]]))

        signed = torch.tensor([[-2.0, 1.0, 0.5]])
        raw = rfa.project_prompt(signed, basis, 1, "raw")
        torch.testing.assert_close(raw, torch.tensor([[-2.0, 0.0, 0.0]]))

    def test_discovery_and_candidate_expansion_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = {
                "prompt_weights": torch.tensor([[1.0, 0.0]]),
                "sampled_ids": torch.tensor([0, 1]),
                "weight_mode": "raw",
            }
            torch.save(payload, root / "b.pt")
            torch.save(payload, root / "a.pt")
            checkpoints = rfa.discover_source_checkpoints(root, "*.pt", False)
            self.assertEqual([item.checkpoint_id for item in checkpoints], ["a.pt", "b.pt"])
            candidates = rfa.expand_candidates(checkpoints, [0, 1])
            self.assertEqual(
                [item.candidate_id for item in candidates],
                ["a.pt::k=1", "a.pt::k=all", "b.pt::k=1", "b.pt::k=all"],
            )

    def test_tokenizer_alignment_and_special_exclusion(self):
        source = FakeTokenizer(special_ids=[0, 3])
        target = FakeTokenizer(special_ids=[0, 3])
        vocabulary = rfa.validate_tokenizer_alignment(source, target)
        self.assertEqual(vocabulary.token_ids.tolist(), [1, 2])
        self.assertEqual(vocabulary.excluded_special_ids, [0, 3])
        mismatched = FakeTokenizer(tokens=["<pad>", "a", "WRONG", "c"])
        with self.assertRaisesRegex(ValueError, "identical token semantics"):
            rfa.validate_tokenizer_alignment(source, mismatched)

    def test_probe_bank_is_deterministic_and_label_free(self):
        texts = [f"probe {index}" for index in range(10)]
        first = rfa._select_probe_rows(texts, 4, 17, "unit")
        second = rfa._select_probe_rows(texts, 4, 17, "unit")
        self.assertEqual(first.indices, second.indices)
        self.assertEqual(first.fingerprint, second.fingerprint)
        dataset = rfa.ProbeDataset(first.texts, FakeTokenizer(), 8)
        self.assertEqual(set(dataset[0]), {"input_ids", "attention_mask"})

    def test_collect_logits_rejects_labels_and_selects_aligned_ids(self):
        clean = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        model = FakeModel("source")
        with patch.object(
            rfa,
            "forward_next_token_logits",
            return_value=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        ):
            result = rfa.collect_logits(
                model, [clean], None, torch.tensor([1, 3]), "unit"
            )
        torch.testing.assert_close(result, torch.tensor([[2.0, 4.0]]))
        with self.assertRaisesRegex(ValueError, "may not contain labels"):
            rfa.collect_logits(
                model,
                [dict(clean, label=torch.tensor([0]))],
                None,
                torch.tensor([1, 3]),
                "unit",
            )

    def test_log_probability_and_logit_shift_effects_are_equivalent(self):
        prompted = torch.tensor([1.2, -0.3, 2.1, 0.7], dtype=torch.float64)
        reference = torch.tensor([-0.1, 0.4, 0.8, 1.5], dtype=torch.float64)
        temperature = 1.7
        log_probability_delta = (
            torch.log_softmax(prompted, -1) - torch.log_softmax(reference, -1)
        )
        expected = torch.softmax(log_probability_delta / temperature, -1)
        actual = torch.softmax((prompted - reference) / temperature, -1)
        torch.testing.assert_close(actual, expected)

    def test_fixed_source_anchor_plan_and_other_bucket(self):
        contributions = torch.tensor(
            [[0.7, 0.2, 0.08, 0.02], [0.1, 0.1, 0.4, 0.4]]
        )
        plan = rfa.build_anchor_plan(contributions, 0.8, 1, 2)
        self.assertEqual(plan.indices[0].tolist(), [0, 1])
        self.assertEqual(plan.selected_counts, [2, 2])

        source = torch.tensor([[3.0, 2.0, 0.0, -1.0], [0.0, 1.0, 2.0, 3.0]])
        target = torch.tensor([[3.0, 2.0, -5.0, -5.0], [3.0, 2.0, 1.0, 0.0]])
        identical = rfa.rowwise_rfa(source, source, 2.0, plan)
        torch.testing.assert_close(identical, torch.zeros_like(identical), atol=1e-12, rtol=0)
        changed = rfa.rowwise_rfa(source, target, 2.0, plan)
        self.assertTrue(torch.isfinite(changed).all())
        self.assertTrue((changed > 0).all())

    def test_offline_analysis(self):
        selection = {
            "ranking": [
                {"candidate": "a::k=all", "rfa_score": 0.1, "selected": True},
                {"candidate": "b::k=all", "rfa_score": 0.2, "selected": False},
                {"candidate": "c::k=all", "rfa_score": 0.3, "selected": False},
            ]
        }
        records = [
            {"candidate": "a::k=all", "accuracy": 0.8},
            {"candidate": "b::k=all", "accuracy": 0.9},
            {"candidate": "c::k=all", "accuracy": 0.7},
        ]
        joined, summary = analyze_rfa.analyze(selection, records)
        self.assertEqual(len(joined), 3)
        self.assertAlmostEqual(summary["spearman_negative_rfa_accuracy"], 0.5)
        self.assertAlmostEqual(summary["top1_selection_regret"], 0.1)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                rfa.parse_args(
                    [
                        "--checkpoints-dir", "checkpoints",
                        "--accuracy-file", "forbidden.csv",
                    ]
                )

    def test_end_to_end_with_test_doubles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            torch.save(
                {
                    "prompt_weights": torch.tensor([[1.0]]),
                    "sampled_ids": torch.tensor([0]),
                    "weight_mode": "raw",
                    "seed": 1,
                },
                checkpoint_dir / "a.pt",
            )
            torch.save(
                {
                    "prompt_weights": torch.tensor([[1.0]]),
                    "sampled_ids": torch.tensor([1]),
                    "weight_mode": "raw",
                    "seed": 2,
                },
                checkpoint_dir / "b.pt",
            )
            args = rfa.parse_args(
                [
                    "--checkpoints-dir", str(checkpoint_dir),
                    "--source-model", "source",
                    "--target-model", "target",
                    "--num-probes", "2",
                    "--max-length", "4",
                    "--anchor-mode", "source-js",
                    "--min-anchor-tokens", "1",
                    "--max-anchor-tokens", "2",
                    "--device", "cpu",
                    "--dtype", "float32",
                    "--output-dir", str(root / "reports"),
                ]
            )
            tokenizer = FakeTokenizer()
            probes = rfa.ProbeBank(["one", "two"], [0, 1], "unit", "abc")

            def fake_load_model(name, *unused):
                return FakeModel(name)

            def fake_collect(model, unused_loader, prompt, unused_ids, unused_description):
                if prompt is None:
                    row = torch.zeros(4)
                elif model.name == "source":
                    row = (
                        torch.tensor([3.0, 1.0, 0.0, -2.0])
                        if prompt[0, 0] > 0
                        else torch.tensor([-2.0, 0.0, 1.0, 3.0])
                    )
                else:
                    row = torch.tensor([3.0, 1.0, 0.0, -2.0])
                return row.unsqueeze(0).repeat(2, 1)

            with patch.object(rfa, "load_tokenizer", return_value=tokenizer), \
                    patch.object(rfa, "load_probe_bank", return_value=probes), \
                    patch.object(rfa, "load_model", side_effect=fake_load_model), \
                    patch.object(rfa, "collect_logits", side_effect=fake_collect), \
                    patch.object(rfa, "release_model"), \
                    contextlib.redirect_stdout(io.StringIO()):
                rows, json_path, csv_path = rfa.run(args)
            self.assertEqual(rows[0]["candidate"], "a.pt::k=all")
            self.assertTrue(rows[0]["selected"])
            self.assertLess(rows[0]["rfa_score"], rows[1]["rfa_score"])
            self.assertTrue(csv_path.is_file())
            report = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertFalse(report["method"]["labels_read"])
            self.assertEqual(report["method"]["anchors"]["mode"], "source-js")
            self.assertTrue(report["method"]["anchors"]["shared_by_all_candidates"])
            self.assertNotIn("accuracy", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
