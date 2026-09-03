from pathlib import Path


class CharacterTokenizer:
    def __init__(self, text: str):
        # Get every unique character in the dataset
        chars = sorted(list(set(text)))

        self.chars = chars
        self.vocab_size = len(chars)

        # Character to integer
        self.stoi = {ch: i for i, ch in enumerate(chars)}

        # Integer to character
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        """
        Convert a string into a list of integers.
        Example: "hello" -> [12, 5, 18, 18, 21]
        """
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        """
        Convert a list of integers back into a string.
        Example: [12, 5, 18, 18, 21] -> "hello"
        """
        return "".join([self.itos[token] for token in tokens])


def load_text(path: str) -> str:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find file: {file_path}")

    return file_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    text = load_text("data/input.txt")
    tokenizer = CharacterTokenizer(text)

    print(f"Dataset length: {len(text):,} characters")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Vocabulary: {tokenizer.chars}")

    sample = "To be"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    print(f"\nSample text: {sample}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
