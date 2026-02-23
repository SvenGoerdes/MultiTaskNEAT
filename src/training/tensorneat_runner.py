from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable
import json
import random

import numpy as np
    # "population_size": 4,
    # "species_size": 2,
    # "generations": 1,
    # "episodes_per_task": 1,
    # "max_steps_per_episode": 10,
    # "seed": 42,
    # "survival_threshold": 0.2,
    # "compatibility_threshold": 2.0,
    # "max_nodes": 64,
    # "max_conns": 256,
    # "activation_options": [

@dataclass
class TensorNEATRunConfig:
    task_names: tuple[str, ...] = (
        "reach-v3",
        "window-open-v3",
        "button-press-v3",
    )
    population_size: int = 4
    species_size: int = 2
    generations: int = 1
    episodes_per_task: int = 1
    max_steps_per_episode: int = 10
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
    success_bonus: float = 100.0
    output_dir: str = "output/tensorneat"
    verbose: bool = True
    render: bool = False


@dataclass
class TensorNEATRunResult:
    best_fitness: float
    best_generation: int
    resolved_tasks: list[str]
    observation_dim: int
    action_dim: int
    history: list[dict[str, float]]
    history_path: str
    best_genome_path: str


def run_tensorneat_metaworld(config: TensorNEATRunConfig) -> TensorNEATRunResult:
    modules = _import_runtime_dependencies()
    gym = modules["gym"]
    jax = modules["jax"]
    jnp = modules["jnp"]
    tensorneat = modules["tensorneat"]

    random.seed(config.seed)
    np.random.seed(config.seed)

    resolved_tasks = _resolve_tasks(gym, config.task_names)
    obs_dim, act_dim, action_low, action_high = _inspect_spaces(gym, resolved_tasks[0])

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
        num_inputs=obs_dim,
        num_outputs=act_dim,
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
                gym=gym,
                jnp=jnp,
                forward_fn=algorithm.forward,
                state=state,
                transformed=transformed,
                task_names=resolved_tasks,
                episodes_per_task=config.episodes_per_task,
                max_steps=config.max_steps_per_episode,
                base_seed=individual_seed,
                action_low=action_low,
                action_high=action_high,
                action_decoder=config.action_decoder,
                success_bonus=config.success_bonus,
                render=config.render,
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
        resolved_tasks=resolved_tasks,
        observation_dim=obs_dim,
        action_dim=act_dim,
    )

    return TensorNEATRunResult(
        best_fitness=best_fitness,
        best_generation=best_generation,
        resolved_tasks=resolved_tasks,
        observation_dim=obs_dim,
        action_dim=act_dim,
        history=history,
        history_path=history_path,
        best_genome_path=best_path,
    )


def m1_profile() -> TensorNEATRunConfig:
    return TensorNEATRunConfig(
        population_size=24,
        species_size=6,
        generations=8,
        episodes_per_task=1,
        max_steps_per_episode=200,
        task_names=("reach-v3", "window-open-v3", "button-press-v3"),
    )


def gpu_profile() -> TensorNEATRunConfig:
    return TensorNEATRunConfig(
        population_size=96,
        species_size=16,
        generations=40,
        episodes_per_task=2,
        max_steps_per_episode=500,
        task_names=(
            "reach-v3",
            "window-open-v3",
            "button-press-v3",
            "drawer-open-v3",
            "door-open-v3",
        ),
    )


def _import_runtime_dependencies() -> dict[str, Any]:
    try:
        import gymnasium as gym
        import metaworld  # noqa: F401
        import jax
        import jax.numpy as jnp
        from tensorneat.algorithm.neat import NEAT
        from tensorneat.common import ACT, AGG, State
        from tensorneat.genome import DefaultGenome
        from tensorneat.genome.gene.node.default import DefaultNode
    except ImportError as exc:
        raise RuntimeError(
            "TensorNEAT runtime dependencies are missing. "
            "Install them first (see README TensorNEAT section)."
        ) from exc

    return {
        "gym": gym,
        "jax": jax,
        "jnp": jnp,
        "tensorneat": {
            "NEAT": NEAT,
            "ACT": ACT,
            "AGG": AGG,
            "State": State,
            "DefaultGenome": DefaultGenome,
            "DefaultNode": DefaultNode,
        },
    }


def _resolve_tasks(gym, task_names: Iterable[str]) -> list[str]:
    resolved: list[str] = []
    for task_name in task_names:
        resolved.append(_resolve_single_task(gym, task_name))
    return resolved


def _resolve_single_task(gym, task_name: str) -> str:
    candidates = [task_name]
    if task_name.endswith("-v3"):
        candidates.append(task_name.replace("-v3", "-v2"))
    elif task_name.endswith("-v2"):
        candidates.append(task_name.replace("-v2", "-v3"))

    for candidate in candidates:
        env = None
        try:
            env = gym.make("Meta-World/MT1", env_name=candidate, render_mode=None)
            env.reset(seed=0)
            return candidate
        except Exception:
            continue
        finally:
            if env is not None:
                env.close()

    tried = ", ".join(candidates)
    raise ValueError(f"Could not resolve MetaWorld task '{task_name}'. Tried: {tried}")


def _inspect_spaces(gym, task_name: str) -> tuple[int, int, np.ndarray, np.ndarray]:
    env = gym.make("Meta-World/MT1", env_name=task_name, render_mode=None)
    try:
        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        if len(obs_shape) != 1:
            raise ValueError(f"Expected 1D observation shape, got {obs_shape}")
        if len(action_shape) != 1:
            raise ValueError(f"Expected 1D action shape, got {action_shape}")

        return (
            int(obs_shape[0]),
            int(action_shape[0]),
            np.asarray(env.action_space.low, dtype=np.float32),
            np.asarray(env.action_space.high, dtype=np.float32),
        )
    finally:
        env.close()


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
    gym,
    jnp,
    forward_fn,
    state,
    transformed,
    task_names: Iterable[str],
    episodes_per_task: int,
    max_steps: int,
    base_seed: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    action_decoder: str,
    success_bonus: float,
    render: bool,
) -> float:
    task_scores: list[float] = []

    for task_idx, task_name in enumerate(task_names):
        episode_returns: list[float] = []
        episode_success: list[float] = []

        env = gym.make(
            "Meta-World/MT1",
            env_name=task_name,
            render_mode="human" if render else None,
            max_episode_steps=max_steps,
        )

        try:
            for episode_idx in range(episodes_per_task):
                episode_seed = base_seed + task_idx * 10_000 + episode_idx
                obs, _ = env.reset(seed=episode_seed)
                total_reward = 0.0
                success = 0.0

                for _ in range(max_steps):
                    obs_arr = jnp.asarray(obs, dtype=jnp.float32)
                    raw_action = forward_fn(state, transformed, obs_arr)
                    action = _decode_action(
                        raw_action=raw_action,
                        action_low=action_low,
                        action_high=action_high,
                        mode=action_decoder,
                    )
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    success = max(success, float(info.get("success", 0.0)))
                    if terminated or truncated:
                        break

                episode_returns.append(total_reward)
                episode_success.append(success)
        finally:
            env.close()

        mean_return = float(np.mean(episode_returns))
        mean_success = float(np.mean(episode_success))
        task_scores.append(mean_return + success_bonus * mean_success)

    return float(np.mean(task_scores))


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
    resolved_tasks: list[str],
    observation_dim: int,
    action_dim: int,
) -> tuple[str, str]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "history.json"
    best_path = output_dir / "best_genome.npz"

    payload = {
        "config": asdict(config),
        "resolved_tasks": resolved_tasks,
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
