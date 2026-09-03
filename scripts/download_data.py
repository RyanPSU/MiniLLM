import argparse
from pathlib import Path
from urllib.request import urlopen

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "input.txt"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Tiny Shakespeare dataset."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite data/input.txt if it already exists.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if OUTPUT_PATH.exists() and not arguments.force:
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists. Use --force to replace it."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(DATA_URL) as response:
        text = response.read().decode("utf-8")

    OUTPUT_PATH.write_text(text, encoding="utf-8")

    print(f"Saved dataset to {OUTPUT_PATH}")
    print(f"Characters: {len(text):,}")
    print(f"Vocabulary size: {len(set(text))}")


if __name__ == "__main__":
    main()
