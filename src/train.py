import torch
from tokenizer import CharacterTokenizer, load_text
from model import SimpleLanguageModel

# Hyperparameters
batch_size = 32
block_size = 8
train_split = 0.9
num_heads = 4

learning_rate = 1e-3
max_iters = 10000
eval_interval = 300

n_embd = 32

# Load and tokenize data
text = load_text("data/input.txt")
tokenizer = CharacterTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# Split into train and validation sets
n = int(train_split * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str):
    """
    Create a batch of input sequences x and target sequences y.

    x is the model input.
    y is what the model should predict.
    """
    source_data = train_data if split == "train" else val_data

    # Random starting positions
    ix = torch.randint(len(source_data) - block_size, (batch_size,))

    # Stack input sequences
    x = torch.stack([source_data[i:i + block_size] for i in ix])

    # Stack target sequences, shifted one character forward
    y = torch.stack([source_data[i + 1:i + block_size + 1] for i in ix])

    return x, y

@torch.no_grad()
def estimate_loss(model):
    """
    Estimate average training and validation loss.
    torch.no_grad() tells PyTorch not to track gradients here,
    because we are only evaluating, not training.
    """
    out = {}

    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(10)

        for k in range(10):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out

if __name__ == "__main__":
    print(f"Dataset length: {len(text):,} characters")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")

    model = SimpleLanguageModel(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    num_heads=num_heads,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iters):
        if step % eval_interval == 0:
            losses = estimate_loss(model)
            print(
                f"Step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        x, y = get_batch("train")

        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\nTraining complete.")

    context = torch.zeros((1, 1), dtype=torch.long)
    generated_tokens = model.generate(context, max_new_tokens=300)[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)

    print("\nGenerated text:")
    print(generated_text)