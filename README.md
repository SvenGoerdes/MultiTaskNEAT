# Thesis NEAT Reinforcement Learning Multi-Task Solver

## Setup

This project uses `uv`.

```bash
uv sync
```

## TensorNEAT + MetaWorld

TensorNEAT integration is implemented via:

- `scripts/tensorneat_metaworld.py`
- `src/training/tensorneat_runner.py`

Install optional dependencies:

```bash
uv sync --extra metaworld --extra tensorneat
```

### macOS M1 Pro (local)

Use the conservative profile first:

```bash
uv run python scripts/tensorneat_metaworld.py --profile m1
```

This uses smaller population/generation defaults to keep runtime manageable on CPU.

Detailed setup notes: `docs/tensorneat-setup.md`

### University GPU server (later)

Use larger defaults:

```bash
uv run python scripts/tensorneat_metaworld.py --profile gpu
```

If your server has CUDA-enabled JAX configured, TensorNEAT can offload model-side JAX ops there. MetaWorld/MuJoCo stepping itself remains environment-simulation bound.

## Example custom run

```bash
uv run python scripts/tensorneat_metaworld.py \
  --profile custom \
  --tasks reach-v3 drawer-open-v3 button-press-v3 \
  --population-size 64 \
  --generations 20 \
  --episodes-per-task 2 \
  --max-steps 500
```
