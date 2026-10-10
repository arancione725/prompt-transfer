"""Focused tests for the gradient-refinement path; no downloads required."""

from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from qwen_cross_model import rfa
from qwen_cross_model.refine_target import _validate_gradient
from qwen_cross_model.superpos import prompt_from_coordinates


class ToyBase(nn.Module):
    def forward(self, inputs_embeds, attention_mask, **unused):
        masked = inputs_embeds * attention_mask.unsqueeze(-1)
        # Causal cumulative state: the final real token depends on every prompt token.
        return SimpleNamespace(last_hidden_state=masked.cumsum(dim=1))


class ToyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(
            torch.tensor([[0.0, 0.0], [1.0, 0.2], [0.1, 1.0], [-0.5, 0.4]]),
            freeze=True,
        )
        self.model = ToyBase()
        self.lm_head = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [-0.8, 0.3], [0.2, -0.7]])
            )
        self.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embeddings

    def get_output_embeddings(self):
        return self.lm_head


class PaddingTokenizer:
    pad_token = None
    eos_token = "<eos>"
    padding_side = "right"


class RefinementTests(unittest.TestCase):
    def test_tokenizer_contract_forces_left_padding(self):
        tokenizer = rfa.configure_tokenizer(PaddingTokenizer())
        self.assertEqual(tokenizer.pad_token, "<eos>")
        self.assertEqual(tokenizer.padding_side, "left")

    def test_last_valid_gather_ignores_right_padding_in_toy_model(self):
        model = ToyCausalLM()
        right_ids = torch.tensor([[1, 2, 0, 0]])
        right_mask = torch.tensor([[1, 1, 0, 0]])
        left_ids = torch.tensor([[0, 0, 1, 2]])
        left_mask = torch.tensor([[0, 0, 1, 1]])
        right = rfa.forward_next_token_logits(model, right_ids, right_mask, None)
        left = rfa.forward_next_token_logits(model, left_ids, left_mask, None)
        torch.testing.assert_close(right, left)

    def test_full_rfa_step_updates_only_target_logits(self):
        model = ToyCausalLM()
        ids = torch.tensor([[0, 1, 2]])
        mask = torch.tensor([[0, 1, 1]])
        sampled_ids = torch.tensor([1, 2, 3])
        source_logits = torch.tensor([[1.0, 0.0, -0.5], [0.1, 0.8, -0.2]])
        target_logits = nn.Parameter(source_logits.clone())
        source_coordinates = torch.softmax(source_logits / 0.5, -1)
        reference = rfa.forward_next_token_logits(model, ids, mask, None).detach()
        before = target_logits.detach().clone()
        optimizer = torch.optim.Adam([target_logits], lr=0.05)
        optimizer.zero_grad(set_to_none=True)
        coordinates = torch.softmax(target_logits / 0.5, -1)
        prompt = prompt_from_coordinates(coordinates, sampled_ids, model.embeddings.weight)
        prompted = rfa.forward_next_token_logits(model, ids, mask, prompt)
        target_effect = prompted - reference
        source_effect = torch.tensor([[0.7, -0.4, 0.2, -0.1]])
        loss = rfa.rowwise_rfa(source_effect, target_effect, 2.0).mean()
        loss = loss + rfa.coordinate_kl(coordinates, source_coordinates)
        loss.backward()
        _validate_gradient(target_logits, loss)
        optimizer.step()
        self.assertFalse(torch.equal(before, target_logits.detach()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
