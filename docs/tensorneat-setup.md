# TensorNEAT Setup Notes

## Local development (MacBook M1 Pro, 32 GB RAM)

Recommended first run:

```bash
uv sync --extra metaworld --extra tensorneat
uv run python scripts/tensorneat_metaworld.py --profile m1
```

The `m1` profile is intentionally conservative for MuJoCo + MetaWorld CPU simulation.

## GPU server setup (later)

1. Install CUDA-enabled JAX in that server environment (follow official JAX install matrix for the server CUDA version).
2. Install project extras:

```bash
uv sync --extra metaworld --extra tensorneat
```

3. Run a larger profile:

```bash
uv run python scripts/tensorneat_metaworld.py --profile gpu
```

## Notes

- TensorNEAT JAX ops can benefit from GPU acceleration.
- MetaWorld simulation still spends substantial time in environment stepping, so expect mixed CPU/GPU bottlenecks.
- Artifacts are written to `output/tensorneat/` by default (`history.json`, `best_genome.npz`).
