import unittest

import torch

from src.model import MiniLLM, TransformerBlock


class TestCausalSelfAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        self.model = MiniLLM(
        vocab_size=10,
        block_size=8,
        n_embd=16,
        num_heads=4,
        num_layers=3,
        )
        self.model.eval()

    def test_output_shape(self):
        tokens = torch.randint(0, 10, (2, 5))

        logits, loss = self.model(tokens)

        self.assertEqual(logits.shape, (2, 5, 10))
        self.assertIsNone(loss)

    def test_future_tokens_do_not_affect_earlier_positions(self):
        sequence_a = torch.tensor([[1, 2, 3, 4]])
        sequence_b = torch.tensor([[1, 2, 8, 9]])

        logits_a, _ = self.model(sequence_a)
        logits_b, _ = self.model(sequence_b)

        self.assertTrue(
            torch.allclose(
                logits_a[:, :2, :],
                logits_b[:, :2, :],
                atol=1e-6,
            )
        )

    def test_earlier_context_affects_current_prediction(self):
        sequence_a = torch.tensor([[1, 2, 3]])
        sequence_b = torch.tensor([[4, 5, 3]])

        logits_a, _ = self.model(sequence_a)
        logits_b, _ = self.model(sequence_b)

        self.assertFalse(
            torch.allclose(
                logits_a[:, -1, :],
                logits_b[:, -1, :],
                atol=1e-6,
            )
        )

    def test_model_uses_requested_number_of_heads(self):
        first_block = self.model.blocks[0]

        self.assertIsInstance(first_block, TransformerBlock)

        if isinstance(first_block, TransformerBlock):
            self.assertEqual(len(first_block.self_attention.heads), 4)

    def test_model_uses_requested_number_of_layers(self):
        self.assertEqual(len(self.model.blocks), 3)

    def test_model_requires_at_least_one_layer(self):
        with self.assertRaises(ValueError):
            MiniLLM(
                vocab_size=10,
                block_size=8,
                n_embd=16,
                num_heads=4,
                num_layers=0,
            )

    def test_embedding_size_must_be_divisible_by_head_count(self):
        with self.assertRaises(ValueError):
            MiniLLM(
                vocab_size=10,
                block_size=8,
                n_embd=10,
                num_heads=4,
                num_layers=1
            )
    def test_model_requires_at_least_one_head(self):
        with self.assertRaises(ValueError):
            MiniLLM(
                vocab_size=10,
                block_size=8,
                n_embd=16,
                num_heads=0,
                num_layers=1,
            )

    def test_backward_pass_through_transformer_block(self):
        inputs = torch.randint(0, 10, (2, 5))
        targets = torch.randint(0, 10, (2, 5))

        _, loss = self.model(inputs, targets)

        if loss is None:
            self.fail("Expected a loss when targets are provided")

        loss.backward()

        for parameter in self.model.parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad)

                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()