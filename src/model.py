import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttentionHead(nn.Module):
    tril: torch.Tensor

    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()

        # Bias free linear projections 
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Lower-triangular matrix that goes through the model but does not get trained
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.head_size = head_size

    def forward(self, x):
        batch_size, sequence_length, channels = x.shape

        # Vectors with shape (B, T, H)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Scaled attention score in shape (B, T, T)
        attention_scores = q @ k.transpose(-2, -1)
        attention_scores = attention_scores * (self.head_size ** -0.5)

        # Apply only the relevant T × T portion of the causal mask.
        attention_scores = attention_scores.masked_fill(
            self.tril[:sequence_length, :sequence_length] == 0,
            float("-inf")
        )

        # Normalize into probabilities
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Weighted sum of value in shape (B, T, H)
        output = attention_weights @ v

        return output


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        num_heads: int,
        block_size: int,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be at least 1")

        if n_embd % num_heads != 0:
            raise ValueError("n_embd must be divisible by num_heads")

        head_size = n_embd // num_heads

        self.heads = nn.ModuleList(
            [
                CausalSelfAttentionHead(
                    n_embd=n_embd,
                    head_size=head_size,
                    block_size=block_size,
                )
                for _ in range(num_heads)
            ]
        )

        self.output_projection = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        concatenated_heads = torch.cat(
            [head(x) for head in self.heads],
            dim=-1,
        )

        return self.output_projection(concatenated_heads)


class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        num_heads: int,
        block_size: int,
    ):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.self_attention = MultiHeadCausalSelfAttention(
            n_embd=n_embd,
            num_heads=num_heads,
            block_size=block_size,
        )

        self.layer_norm_2 = nn.LayerNorm(n_embd)
        self.feed_forward = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

    
class MiniLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_embd=n_embd,
                    num_heads=num_heads,
                    block_size=block_size,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(n_embd)
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

        x = self.blocks(x)
        x = self.final_layer_norm(x)
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