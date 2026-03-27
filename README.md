# CHART

Continuous Hierarchical Autoregressive Rhythm Tracker (CHART).

## Status

This repository currently contains a boilerplate project structure for a PyTorch-based
Sequential Variational Transformer (SVT) approach to beat/downbeat tracking with
continuous phase variables.

Core modules are scaffolded only (model, loss, training, and evaluation), with
placeholder interfaces and TODO markers for custom implementation.

## Project Layout

- `models/` – SVT model scaffold and loss function stubs
- `training/` – dataset and training entrypoint stubs
- `evaluation/` – inference, phase conversion, and scoring utilities

## Inference

Run CHART phase inference from precomputed acoustic activations (`.npy`):

```bash
python -m evaluation.inference \
	--checkpoint checkpoints/chart_end2end_smoke.pt \
	--input_npy /path/to/activations.npy \
	--output_npy /path/to/predicted_phase.npy
```

- Input shape: `[T, 2]` or `[B, T, 2]`
- Output shape: `[T, 3]` or `[B, T, 3]`
- Default mode is frame-wise autoregressive rollout; add `--non_autoregressive` for one-pass inference.

## Next Steps

- Implement model forward logic in `models/svt_core.py`
- Implement objective terms in `models/loss.py`
- Add data loading and training logic in `training/`
- Add inference and metric integration (`mir_eval`) in `evaluation/`
