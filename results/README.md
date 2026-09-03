# Experiment Results

This directory contains the reproducible evidence from MiniLLM's controlled
training and generation experiments.

## Training experiments

Four configurations were trained on Tiny Shakespeare for 3,000 iterations
using random seed 42. Each experiment changed one setting from the baseline.

The best tested configuration used a learning rate of `0.001` and achieved a
validation loss of `1.8502`, a 10.0% improvement over the baseline without
increasing parameter count or runtime.

See:

- [`experiment_summary.md`](experiment_summary.md) for the comparison table
- [`loss_comparison.png`](loss_comparison.png) for training and validation curves
- Individual JSON files for complete configurations and loss histories

## Generation experiments

The [`generation`](generation/) directory compares temperature and top-k
settings using the best tested checkpoint.

All samples use the same prompt, output length, and random seed so differences
can be attributed to the generation settings.

## Reproducing the analysis

Regenerate the comparison table and graph by running:

```bash
python scripts/plot_results.py
```