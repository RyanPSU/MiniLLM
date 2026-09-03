# Controlled Training Experiments

All experiments used Tiny Shakespeare, seed 42, and 3,000 training iterations. Each experiment changed one setting from the baseline.

| Run | Controlled change | Parameters | Best validation loss | Improvement vs. baseline | Best step | Runtime (s) |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Reference configuration | 112,193 | 2.0567 | Reference | 2750 | 33.24 |
| 4 layers | Transformer layers: 2 → 4 | 211,777 | 1.9787 | +3.8% | 2750 | 58.03 |
| Context 128 | Context length: 64 → 128 | 116,289 | 2.1307 | -3.6% | 2750 | 82.42 |
| Learning rate 0.001 | Learning rate: 0.0003 → 0.001 | 112,193 | 1.8502 | +10.0% | 2750 | 32.71 |
