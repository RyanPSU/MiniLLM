import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        # Each token directly reads off logits for the next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx: input token IDs, shape (batch_size, block_size)
        targets: target token IDs, shape (batch_size, block_size)
        """

        # logits shape: (batch_size, block_size, vocab_size)
        logits = self.token_embedding_table(idx)

        loss = None

        if targets is not None:
            batch_size, block_size, vocab_size = logits.shape

            # PyTorch cross_entropy expects shape (N, C)
            # N = number of examples
            # C = number of classes
            logits = logits.view(batch_size * block_size, vocab_size)
            targets = targets.view(batch_size * block_size)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        """
        Generate new tokens from the model.

        idx: starting token IDs, shape (batch_size, current_context_length)
        """

        for _ in range(max_new_tokens):
            # Get predictions
            logits, loss = self(idx)

            # Focus only on the last time step
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled token to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx