import torch
from tokenizer import CharacterTokenizer, load_text

# Hyperparameters
batch_size = 4       # How many sequences we process at once
block_size = 8       # How many characters are in each sequence
train_split = 0.9    # 90% train, 10% validation


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


if __name__ == "__main__":
    print(f"Dataset length: {len(text):,} characters")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")

    x, y = get_batch("train")

    print("\nInput batch shape:", x.shape)
    print("Target batch shape:", y.shape)

    print("\nFirst input example:")
    print(x[0])
    print(tokenizer.decode(x[0].tolist()))

    print("\nFirst target example:")
    print(y[0])
    print(tokenizer.decode(y[0].tolist()))