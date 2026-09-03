# Generation Setting Comparison

These samples were generated from the `lr_1e3_best.pt` checkpoint, which
achieved the lowest validation loss in the controlled training experiments.

Every sample uses:

- Prompt: `ROMEO:`
- Maximum new tokens: 500
- Random seed: 42

| Sample | Temperature | Top-k | Observed behavior |
|---|---:|---:|---|
| [`temperature_05_top_k_20.txt`](temperature_05_top_k_20.txt) | 0.5 | 20 | Repetitive, favors common words and structures |
| [`temperature_08_top_k_20.txt`](temperature_08_top_k_20.txt) | 0.8 | 20 | Best balance of structure and variety |
| [`temperature_12_top_k_20.txt`](temperature_12_top_k_20.txt) | 1.2 | 20 | More diverse but produces more malformed words |
| [`temperature_08_no_top_k.txt`](temperature_08_no_top_k.txt) | 0.8 | None | Allows unlikely choices and more abrupt phrases |

## Interpretation

At temperature `0.5`, the model repeatedly selects high-probability characters,
creating safer but repetitive text.

Temperature `0.8` preserves recognizable speaker labels, line breaks, and
Shakespeare-like formatting while producing more varied output.

Temperature `1.2` increases diversity but causes less consistent names,
punctuation, and word structure.

Removing top-k filtering allows the model to sample from the entire vocabulary.
This occasionally introduces unlikely characters and malformed phrases that
top-k sampling suppresses.

These comparisons are qualitative. The model is small and character-level, so
the generated samples demonstrate learned structural patterns rather than
consistent semantic coherence.