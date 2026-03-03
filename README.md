# Thesis NEAT Reinforcement Learning Multi-Task Solver

## Setup

This project uses `uv`.

```bash
uv sync
```

## Main Training Script (neat-python)

Trains a single shared NEAT network on Brax Hopper and Walker2D using input/output padding:

```bash
uv run python scripts/main.py
```

## TensorNEAT + Brax

TensorNEAT integration is implemented via:

- `scripts/tensorneat_brax.py`
- `src/training/tensorneat_runner.py`

Install optional dependencies:

```bash
uv sync --extra tensorneat
```

### macOS / local development

```bash
uv run python scripts/tensorneat_brax.py --profile cpu
```

Uses the `positional` Brax backend with conservative defaults (pop=24, 8 generations).

> **Note:** Brax's MJCF loader calls MJX internally, which does not support Apple Metal GPUs. The `positional` backend avoids this and runs on CPU. JAX Metal acceleration is not compatible with Brax at this time.

Detailed setup notes: `docs/tensorneat-setup.md`

### CUDA GPU server

```bash
uv run python scripts/tensorneat_brax.py --profile gpu
```

Uses the `mjx` backend for full GPU-accelerated physics (pop=96, 40 generations).

## Example custom run

```bash
uv run python scripts/tensorneat_brax.py \
  --profile custom \
  --envs hopper walker2d \
  --population-size 64 \
  --generations 20 \
  --episodes-per-env 2 \
  --max-steps 1000
```
