# Generation Setting Comparison

These samples were generated from the `lr_1e3_best.pt` checkpoint, which
achieved the lowest validation loss in the controlled training experiments.

Every sample uses:

- Prompt: `ROMEO:`
- Maximum new tokens: 500
- Random seed: 42

| Sample | Temperature | Top-k | Comparison purpose |
|---|---:|---:|---|
| `temperature_05_top_k_20.txt` | 0.5 | 20 | Lower randomness |
| `temperature_08_top_k_20.txt` | 0.8 | 20 | Balanced sampling |
| `temperature_12_top_k_20.txt` | 1.2 | 20 | Higher randomness |
| `temperature_08_no_top_k.txt` | 0.8 | None | Effect of removing top-k filtering |