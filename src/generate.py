import argparse
from pathlib import Path

import torch

from model import MiniLLM
from tokenizer import CharacterTokenizer


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text using a trained MiniLLM checkpoint."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best.pt"),
    )
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if not arguments.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {arguments.checkpoint}"
        )

    if not arguments.prompt:
        raise ValueError("prompt cannot be empty")

    device = select_device()
    torch.manual_seed(arguments.seed)

    checkpoint = torch.load(
        arguments.checkpoint,
        map_location=device,
    )

    model_config = checkpoint["model_config"]
    tokenizer_chars = checkpoint["tokenizer_chars"]

    tokenizer = CharacterTokenizer("".join(tokenizer_chars))

    unknown_characters = sorted(
        set(arguments.prompt) - set(tokenizer.chars)
    )

    if unknown_characters:
        raise ValueError(
            f"Prompt contains characters outside the vocabulary: "
            f"{unknown_characters}"
        )

    model = MiniLLM(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_tokens = tokenizer.encode(arguments.prompt)
    context = torch.tensor(
        [prompt_tokens],
        dtype=torch.long,
        device=device,
    )

    generated_tokens = model.generate(
        context,
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        top_k=arguments.top_k,
    )

    generated_text = tokenizer.decode(
        generated_tokens[0].tolist()
    )

    print(generated_text)


if __name__ == "__main__":
    main()