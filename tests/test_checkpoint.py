import tempfile
import unittest
from pathlib import Path

import torch

from src.model import MiniLLM
from src.tokenizer import CharacterTokenizer


class TestCheckpoint(unittest.TestCase):
    def test_checkpoint_round_trip(self):
        torch.manual_seed(42)

        tokenizer = CharacterTokenizer("abc ")

        model_config = {
            "vocab_size": tokenizer.vocab_size,
            "block_size": 4,
            "n_embd": 8,
            "num_heads": 2,
            "num_layers": 2,
        }

        original_model = MiniLLM(**model_config)
        original_model.eval()

        optimizer = torch.optim.AdamW(original_model.parameters())

        tokens = torch.tensor([[0, 1, 2]])

        original_logits, _ = original_model(tokens)

        checkpoint = {
            "model_state_dict": original_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "tokenizer_chars": tokenizer.chars,
            "step": 10,
            "best_val_loss": 2.5,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

            loaded_checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
            )

        restored_model = MiniLLM(**loaded_checkpoint["model_config"])
        restored_model.load_state_dict(loaded_checkpoint["model_state_dict"])
        restored_model.eval()

        restored_logits, _ = restored_model(tokens)

        restored_tokenizer = CharacterTokenizer(
            "".join(loaded_checkpoint["tokenizer_chars"])
        )

        self.assertTrue(torch.allclose(original_logits, restored_logits))
        self.assertEqual(restored_tokenizer.chars, tokenizer.chars)
        self.assertEqual(loaded_checkpoint["step"], 10)
        self.assertEqual(loaded_checkpoint["best_val_loss"], 2.5)


if __name__ == "__main__":
    unittest.main()
