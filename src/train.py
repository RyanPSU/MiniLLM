import torch
from tokenizer import CharacterTokenizer, load_text
from model import MiniLLM
from pathlib import Path

seed = 42
eval_iters = 50
checkpoint_dir = Path("checkpoints")

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


torch.manual_seed(seed)
device = select_device()

# Hyperparameters
batch_size = 32
block_size = 8
train_split = 0.9
num_heads = 4
num_layers = 4

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

    return x.to(device), y.to(device)

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
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()

    return out

if __name__ == "__main__":
    print(f"Dataset length: {len(text):,} characters")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")

    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(device)

    print(f"Device: {device}")

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Trainable parameters: {parameter_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for step in range(max_iters):
        if step % eval_interval == 0:
            losses = estimate_loss(model)
            print(
                f"Step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )
            current_val_loss = losses["val"]

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": {
                        "vocab_size": tokenizer.vocab_size,
                        "block_size": block_size,
                        "n_embd": n_embd,
                        "num_heads": num_heads,
                        "num_layers": num_layers,
                    },
                    "training_config": {
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "seed": seed,
                    },
                    "tokenizer_chars": tokenizer.chars,
                    "step": step,
                    "best_val_loss": best_val_loss,
                }

                checkpoint_path = checkpoint_dir / "best.pt"
                torch.save(checkpoint, checkpoint_path)

                print(f"Saved checkpoint to {checkpoint_path}")

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