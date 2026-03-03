from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable
import json
import random

import numpy as np


@dataclass
class TensorNEATRunConfig:
    env_names: tuple[str, ...] = (
        "hopper",
        "walker2d",
    )
    population_size: int = 4
    species_size: int = 2
    generations: int = 1
    episodes_per_env: int = 1
    max_steps_per_episode: int = 1000
    seed: int = 42
    survival_threshold: float = 0.2
    compatibility_threshold: float = 2.0
    max_nodes: int = 64
    max_conns: int = 256
    activation_options: tuple[str, ...] = (
        "relu",
        "sigmoid",
        "tanh",
        "sin",
        "gaussian",
    )
    activation_replace_rate: float = 0.15
    action_decoder: str = "clip"
    brax_backend: str = "positional"
    output_dir: str = "output/tensorneat"
    verbose: bool = True
    render: bool = False


@dataclass
class TensorNEATRunResult:
    best_fitness: float
    best_generation: int
    env_names: list[str]
    observation_dim: int
    action_dim: int
    history: list[dict[str, float]]
    history_path: str
    best_genome_path: str


def run_tensorneat_brax(config: TensorNEATRunConfig) -> TensorNEATRunResult:
    modules = _import_runtime_dependencies()
    jax = modules["jax"]
    jnp = modules["jnp"]
    tensorneat = modules["tensorneat"]
    brax_envs = modules["brax_envs"]

    random.seed(config.seed)
    np.random.seed(config.seed)

    env_layouts, total_obs_dim, total_act_dim = _build_env_layout(
        brax_envs, config.env_names, config.brax_backend
    )
    # Brax continuous actions are typically in [-1, 1]
    action_low = np.full(total_act_dim, -1.0, dtype=np.float32)
    action_high = np.full(total_act_dim, 1.0, dtype=np.float32)

    activation_functions = _build_activation_functions(
        jnp=jnp,
        act_registry=tensorneat["ACT"],
        activation_names=config.activation_options,
    )

    node_gene = tensorneat["DefaultNode"](
        activation_options=activation_functions,
        activation_replace_rate=config.activation_replace_rate,
        aggregation_options=tensorneat["AGG"].sum,
    )
    genome = tensorneat["DefaultGenome"](
        num_inputs=total_obs_dim,
        num_outputs=total_act_dim,
        max_nodes=config.max_nodes,
        max_conns=config.max_conns,
        node_gene=node_gene,
    )
    algorithm = tensorneat["NEAT"](
        genome=genome,
        pop_size=config.population_size,
        species_size=config.species_size,
        survival_threshold=config.survival_threshold,
        compatibility_threshold=config.compatibility_threshold,
    )

    state = tensorneat["State"](randkey=jax.random.PRNGKey(config.seed))
    state = algorithm.setup(state)

    population_transform = jax.vmap(algorithm.transform, in_axes=(None, 0))
    tell_fn = jax.jit(algorithm.tell)

    history: list[dict[str, float]] = []
    best_fitness = float("-inf")
    best_generation = -1
    best_genome = None

    for generation in range(config.generations):
        generation_start = perf_counter()
        population = algorithm.ask(state)
        transformed_population = population_transform(state, population)

        fitnesses = []
        for idx in range(config.population_size):
            transformed = jax.tree_util.tree_map(lambda arr: arr[idx], transformed_population)
            individual_seed = config.seed + generation * 100_000 + idx
            fitness = _evaluate_individual(
                brax_envs=brax_envs,
                jax=jax,
                jnp=jnp,
                forward_fn=algorithm.forward,
                state=state,
                transformed=transformed,
                env_layouts=env_layouts,
                total_obs_dim=total_obs_dim,
                episodes_per_env=config.episodes_per_env,
                max_steps=config.max_steps_per_episode,
                base_seed=individual_seed,
                action_low=action_low,
                action_high=action_high,
                action_decoder=config.action_decoder,
                brax_backend=config.brax_backend,
            )
            fitnesses.append(fitness)

        fitness_array = jnp.asarray(fitnesses, dtype=jnp.float32)
        fitness_array = jnp.where(jnp.isnan(fitness_array), -jnp.inf, fitness_array)
        cpu_fitness = np.asarray(jax.device_get(fitness_array), dtype=np.float32)

        best_idx = int(np.argmax(cpu_fitness))
        generation_best = float(cpu_fitness[best_idx])
        generation_mean = float(np.mean(cpu_fitness))
        generation_min = float(np.min(cpu_fitness))
        duration = perf_counter() - generation_start

        if generation_best > best_fitness:
            best_fitness = generation_best
            best_generation = generation
            best_genome = (
                jax.device_get(population[0][best_idx]),
                jax.device_get(population[1][best_idx]),
            )

        history.append(
            {
                "generation": float(generation),
                "best_fitness": generation_best,
                "mean_fitness": generation_mean,
                "min_fitness": generation_min,
                "duration_seconds": float(duration),
            }
        )

        if config.verbose:
            print(
                f"[tensorneat] generation={generation} "
                f"best={generation_best:.3f} mean={generation_mean:.3f} "
                f"min={generation_min:.3f} duration={duration:.2f}s"
            )

        state = tell_fn(state, fitness_array)

    history_path, best_path = _persist_outputs(
        config=config,
        history=history,
        best_genome=best_genome,
        best_fitness=best_fitness,
        env_names=list(config.env_names),
        observation_dim=total_obs_dim,
        action_dim=total_act_dim,
    )

    return TensorNEATRunResult(
        best_fitness=best_fitness,
        best_generation=best_generation,
        env_names=list(config.env_names),
        observation_dim=total_obs_dim,
        action_dim=total_act_dim,
        history=history,
        history_path=history_path,
        best_genome_path=best_path,
    )


def brax_cpu_profile() -> TensorNEATRunConfig:
    return TensorNEATRunConfig(
        population_size=24,
        species_size=6,
        generations=8,
        episodes_per_env=1,
        max_steps_per_episode=500,
        env_names=("hopper", "walker2d"),
        brax_backend="positional",
    )


def brax_gpu_profile() -> TensorNEATRunConfig:
    """Profile for CUDA GPU servers. Uses mjx backend for full
    GPU-accelerated physics."""
    return TensorNEATRunConfig(
        population_size=96,
        species_size=16,
        generations=40,
        episodes_per_env=2,
        max_steps_per_episode=1000,
        env_names=("hopper", "walker2d"),
        brax_backend="mjx",
    )


def _import_runtime_dependencies() -> dict[str, Any]:
    try:
        import jax
        import jax.numpy as jnp
        from brax import envs as brax_envs
        from tensorneat.algorithm.neat import NEAT
        from tensorneat.common import ACT, AGG, State
        from tensorneat.genome import DefaultGenome
        from tensorneat.genome.gene.node.default import DefaultNode
    except ImportError as exc:
        raise RuntimeError(
            "TensorNEAT runtime dependencies are missing. "
            "Install them first: uv sync --extra tensorneat"
        ) from exc

    return {
        "jax": jax,
        "jnp": jnp,
        "brax_envs": brax_envs,
        "tensorneat": {
            "NEAT": NEAT,
            "ACT": ACT,
            "AGG": AGG,
            "State": State,
            "DefaultGenome": DefaultGenome,
            "DefaultNode": DefaultNode,
        },
    }


@dataclass
class _EnvLayout:
    """Per-environment dimension info and offsets within the padded vector."""
    name: str
    obs_dim: int
    act_dim: int
    input_offset: int
    output_offset: int


def _build_env_layout(brax_envs, env_names: Iterable[str], backend: str) -> tuple[list[_EnvLayout], int, int]:
    """Compute padded input/output layout across all environments.

    Returns (layouts, total_input_dims, total_output_dims).
    """
    layouts: list[_EnvLayout] = []
    input_offset = 0
    output_offset = 0
    for name in env_names:
        env = brax_envs.get_environment(name, backend=backend)
        obs_dim = int(env.observation_size)
        act_dim = int(env.action_size)
        layouts.append(_EnvLayout(
            name=name,
            obs_dim=obs_dim,
            act_dim=act_dim,
            input_offset=input_offset,
            output_offset=output_offset,
        ))
        input_offset += obs_dim
        output_offset += act_dim
    return layouts, input_offset, output_offset


def _build_activation_functions(jnp, act_registry, activation_names: Iterable[str]) -> list[Any]:
    names = list(activation_names)
    if not names:
        raise ValueError("activation_options must contain at least one activation name.")

    if "gaussian" in names and not hasattr(act_registry, "gaussian"):
        act_registry.add_func("gaussian", lambda z: jnp.exp(-(z ** 2)))

    functions = []
    unknown = []
    for name in names:
        if hasattr(act_registry, name):
            functions.append(getattr(act_registry, name))
        else:
            unknown.append(name)

    if unknown:
        raise ValueError(f"Unknown TensorNEAT activation options: {unknown}")
    return functions


def _evaluate_individual(
    brax_envs,
    jax,
    jnp,
    forward_fn,
    state,
    transformed,
    env_layouts: list[_EnvLayout],
    total_obs_dim: int,
    episodes_per_env: int,
    max_steps: int,
    base_seed: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    action_decoder: str,
    brax_backend: str,
) -> float:
    env_scores: list[float] = []

    for env_idx, layout in enumerate(env_layouts):
        episode_returns: list[float] = []

        env = brax_envs.get_environment(layout.name, backend=brax_backend)
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)

        for episode_idx in range(episodes_per_env):
            episode_seed = base_seed + env_idx * 10_000 + episode_idx
            rng = jax.random.PRNGKey(episode_seed)
            brax_state = jit_reset(rng)
            raw_obs = np.asarray(brax_state.obs, dtype=np.float32)
            total_reward = 0.0

            for _ in range(max_steps):
                # Pad observation into full input vector
                padded_input = np.zeros(total_obs_dim, dtype=np.float32)
                padded_input[layout.input_offset:layout.input_offset + layout.obs_dim] = raw_obs

                obs_arr = jnp.asarray(padded_input, dtype=jnp.float32)
                raw_output = forward_fn(state, transformed, obs_arr)
                raw_output = np.asarray(raw_output, dtype=np.float32)

                # Slice this environment's action outputs
                env_action = raw_output[layout.output_offset:layout.output_offset + layout.act_dim]
                action = _decode_action(
                    raw_action=env_action,
                    action_low=action_low[layout.output_offset:layout.output_offset + layout.act_dim],
                    action_high=action_high[layout.output_offset:layout.output_offset + layout.act_dim],
                    mode=action_decoder,
                )
                action_jnp = jnp.asarray(action, dtype=jnp.float32)
                brax_state = jit_step(brax_state, action_jnp)
                raw_obs = np.asarray(brax_state.obs, dtype=np.float32)
                total_reward += float(brax_state.reward)
                if float(brax_state.done):
                    break

            episode_returns.append(total_reward)

        env_scores.append(float(np.mean(episode_returns)))

    return float(np.mean(env_scores))


def _decode_action(raw_action, action_low: np.ndarray, action_high: np.ndarray, mode: str) -> np.ndarray:
    action = np.asarray(raw_action, dtype=np.float32)
    if action.ndim == 0:
        action = np.asarray([float(action)], dtype=np.float32)
    if mode == "tanh":
        action = np.tanh(action)
    elif mode != "clip":
        raise ValueError(f"Unsupported action decoder '{mode}'. Use 'clip' or 'tanh'.")
    return np.clip(action, action_low, action_high)


def _persist_outputs(
    config: TensorNEATRunConfig,
    history: list[dict[str, float]],
    best_genome,
    best_fitness: float,
    env_names: list[str],
    observation_dim: int,
    action_dim: int,
) -> tuple[str, str]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "history.json"
    best_path = output_dir / "best_genome.npz"

    payload = {
        "config": asdict(config),
        "env_names": env_names,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "best_fitness": best_fitness,
        "history": history,
    }
    history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if best_genome is not None:
        np.savez(
            best_path,
            nodes=np.asarray(best_genome[0]),
            conns=np.asarray(best_genome[1]),
            fitness=np.asarray([best_fitness], dtype=np.float32),
        )
    else:
        np.savez(
            best_path,
            nodes=np.asarray([]),
            conns=np.asarray([]),
            fitness=np.asarray([best_fitness], dtype=np.float32),
        )

    return str(history_path), str(best_path)
