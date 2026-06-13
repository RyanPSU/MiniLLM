import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        super().__init__()

        self.block_size = block_size

        # Converts token IDs into learned embedding vectors
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Gives the model information about positions in the sequence
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Converts embeddings back into vocabulary logits
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx: input token IDs, shape (batch_size, block_size)
        targets: target token IDs, shape (batch_size, block_size)
        """

        batch_size, sequence_length = idx.shape

        # Token embeddings: shape (batch_size, sequence_length, n_embd)
        token_embeddings = self.token_embedding_table(idx)

        # Position embeddings: shape (sequence_length, n_embd)
        position_indices = torch.arange(sequence_length, device=idx.device)
        position_embeddings = self.position_embedding_table(position_indices)

        # Combine token identity and position information
        x = token_embeddings + position_embeddings

        # Convert embeddings into logits
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            batch_size, sequence_length, vocab_size = logits.shape

            logits = logits.view(batch_size * sequence_length, vocab_size)
            targets = targets.view(batch_size * sequence_length)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        """
        Generate new tokens from the model.

        idx: starting token IDs, shape (batch_size, current_context_length)
        """

        for _ in range(max_new_tokens):
            # Crop context so it never exceeds block_size
            idx_cond = idx[:, -self.block_size:]

            # Get predictions
            logits, loss = self(idx_cond)

            # Focus only on the last time step
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled token
            idx = torch.cat((idx, idx_next), dim=1)

        return idx