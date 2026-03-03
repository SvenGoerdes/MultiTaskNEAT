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

### macOS with Metal GPU (Apple Silicon)

Install the JAX Metal plugin for GPU acceleration on Mac:

```bash
uv sync --extra tensorneat --extra metal
```

Run with the CPU profile (conservative defaults suitable for local development):

```bash
uv run python scripts/tensorneat_brax.py --profile cpu
```

JAX will automatically detect and use the Metal GPU backend when `jax-metal` is installed. You can verify with:

```bash
uv run python -c "import jax; print(jax.devices())"
```

Detailed setup notes: `docs/tensorneat-setup.md`

### CUDA GPU server

Use larger defaults:

```bash
uv run python scripts/tensorneat_brax.py --profile gpu
```

With CUDA-enabled JAX, both TensorNEAT and Brax leverage full GPU acceleration.

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
