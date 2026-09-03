import argparse
import json
import time
from pathlib import Path

import torch

from model import MiniLLM
from tokenizer import CharacterTokenizer, load_text

checkpoint_dir = Path("checkpoints")
results_dir = Path("results")


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MiniLLM on Tiny Shakespeare."
    )

    parser.add_argument("--run-name", type=str, default="baseline")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


arguments = parse_arguments()

run_name = arguments.run_name
batch_size = arguments.batch_size
block_size = arguments.block_size
n_embd = arguments.n_embd
num_heads = arguments.num_heads
num_layers = arguments.num_layers
dropout = arguments.dropout

learning_rate = arguments.learning_rate
max_iters = arguments.max_iters
eval_interval = arguments.eval_interval
eval_iters = arguments.eval_iters
train_split = arguments.train_split
seed = arguments.seed

torch.manual_seed(seed)
device = select_device()

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
    x = torch.stack([source_data[i : i + block_size] for i in ix])

    # Stack target sequences, shifted one character forward
    y = torch.stack([source_data[i + 1 : i + block_size + 1] for i in ix])

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

    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "block_size": block_size,
        "n_embd": n_embd,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
    }

    training_config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "eval_iters": eval_iters,
        "train_split": train_split,
        "seed": seed,
    }

    model = MiniLLM(**model_config).to(device)

    print(f"Device: {device}")

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Trainable parameters: {parameter_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_step = -1
    training_history = []
    start_time = time.perf_counter()

    for step in range(max_iters):
        if step % eval_interval == 0:
            losses = estimate_loss(model)
            print(
                f"Step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

            current_val_loss = losses["val"]

            training_history.append(
                {
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                }
            )

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_step = step

                checkpoint = {
                    "run_name": run_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": model_config,
                    "training_config": training_config,
                    "tokenizer_chars": tokenizer.chars,
                    "step": step,
                    "best_val_loss": best_val_loss,
                }

                checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"
                torch.save(checkpoint, checkpoint_path)

                print(f"Saved checkpoint to {checkpoint_path}")

        x, y = get_batch("train")
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    elapsed_seconds = time.perf_counter() - start_time

    experiment_results = {
        "run_name": run_name,
        "model_config": model_config,
        "training_config": training_config,
        "parameter_count": parameter_count,
        "device": str(device),
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "elapsed_seconds": elapsed_seconds,
        "history": training_history,
    }

    results_path = results_dir / f"{run_name}.json"

    with results_path.open("w", encoding="utf-8") as results_file:
        json.dump(experiment_results, results_file, indent=2)

    print(f"Saved experiment results to {results_path}")
    print("\nTraining complete.")

    context = torch.zeros(
        (1, 1),
        dtype=torch.long,
        device=device,
    )
    generated_tokens = model.generate(context, max_new_tokens=300)[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)

    print("\nGenerated text:")
    print(generated_text)
