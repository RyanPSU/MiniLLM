import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttentionHead(nn.Module):
    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()

        # TODO: Create three bias-free linear projections:
        # n_embd -> head_size
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)

        # TODO: Create a block_size × block_size lower-triangular matrix.
        # Register it as a buffer named "tril" so it moves with the model
        # between CPU, CUDA, and MPS without becoming a trainable parameter.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.head_size = head_size

    def forward(self, x):
        batch_size, sequence_length, channels = x.shape

        # TODO: Produce Q, K, and V.
        # Each should have shape (B, T, H).
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # TODO: Compute scaled attention scores.
        # Resulting shape: (B, T, T)
        attention_scores = q @ k.transpose(-2, -1)
        attention_scores = attention_scores * (self.head_size ** -0.5)

        # Apply only the relevant T × T portion of the causal mask.
        attention_scores = attention_scores.masked_fill(
            self.tril[:sequence_length, :sequence_length] == 0,
            float("-inf"))

        # TODO: Normalize each row into probabilities.
        attention_weights = ...

        # TODO: Calculate the weighted sum of values.
        # Resulting shape: (B, T, H)
        output = ...

        return output

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