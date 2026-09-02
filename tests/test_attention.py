import unittest

import torch

from src.model import SimpleLanguageModel


class TestCausalSelfAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        self.model = SimpleLanguageModel(
            vocab_size=10,
            block_size=8,
            n_embd=16,
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


if __name__ == "__main__":
    unittest.main()