import json
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

RUNS = {
    "baseline": {
        "label": "Baseline",
        "change": "Reference configuration",
    },
    "deeper_4_layers": {
        "label": "4 layers",
        "change": "Transformer layers: 2 → 4",
    },
    "context_128": {
        "label": "Context 128",
        "change": "Context length: 64 → 128",
    },
    "lr_1e3": {
        "label": "Learning rate 0.001",
        "change": "Learning rate: 0.0003 → 0.001",
    },
}


def load_results():
    results = {}

    for run_name in RUNS:
        results_path = RESULTS_DIR / f"{run_name}.json"

        if not results_path.exists():
            raise FileNotFoundError(f"Missing experiment file: {results_path}")

        with results_path.open("r", encoding="utf-8") as results_file:
            results[run_name] = json.load(results_file)

    return results


def create_loss_plot(results):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    for run_name, run_information in RUNS.items():
        history = results[run_name]["history"]
        steps = [measurement["step"] for measurement in history]
        train_losses = [measurement["train_loss"] for measurement in history]
        validation_losses = [measurement["val_loss"] for measurement in history]

        axes[0].plot(
            steps,
            train_losses,
            marker="o",
            markersize=3,
            label=run_information["label"],
        )
        axes[1].plot(
            steps,
            validation_losses,
            marker="o",
            markersize=3,
            label=run_information["label"],
        )

    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation Loss")

    for axis in axes:
        axis.set_xlabel("Training Step")
        axis.set_ylabel("Cross-Entropy Loss")
        axis.grid(alpha=0.3)
        axis.legend()

    figure.suptitle("MiniLLM Controlled Experiment Comparison")
    figure.tight_layout()

    output_path = RESULTS_DIR / "loss_comparison.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved loss plot to {output_path}")


def create_summary(results):
    baseline_loss = results["baseline"]["best_val_loss"]

    summary_lines = [
        "# Controlled Training Experiments",
        "",
        (
            "All experiments used Tiny Shakespeare, seed 42, and 3,000 "
            "training iterations. Each experiment changed one setting "
            "from the baseline."
        ),
        "",
        (
            "| Run | Controlled change | Parameters | Best validation loss "
            "| Improvement vs. baseline | Best step | Runtime (s) |"
        ),
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for run_name, run_information in RUNS.items():
        result = results[run_name]
        validation_loss = result["best_val_loss"]

        if run_name == "baseline":
            improvement = "Reference"
        else:
            improvement_percent = (
                (baseline_loss - validation_loss) / baseline_loss
            ) * 100
            improvement = f"{improvement_percent:+.1f}%"

        summary_lines.append(
            f"| {run_information['label']} "
            f"| {run_information['change']} "
            f"| {result['parameter_count']:,} "
            f"| {validation_loss:.4f} "
            f"| {improvement} "
            f"| {result['best_step']} "
            f"| {result['elapsed_seconds']:.2f} |"
        )

    summary_path = RESULTS_DIR / "experiment_summary.md"
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Saved experiment summary to {summary_path}")


def main():
    results = load_results()
    create_loss_plot(results)
    create_summary(results)


if __name__ == "__main__":
    main()
