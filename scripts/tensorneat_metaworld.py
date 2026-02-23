import argparse

from training.tensorneat_runner import (
    TensorNEATRunConfig,
    gpu_profile,
    m1_profile,
    run_tensorneat_metaworld,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TensorNEAT on MetaWorld MT1 tasks."
    )
    parser.add_argument(
        "--profile",
        choices=("m1", "gpu", "custom"),
        default="m1",
        help="Hardware profile defaults. Use custom to fully control flags.",
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--species-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--episodes-per-task", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--success-bonus", type=float, default=None)
    parser.add_argument(
        "--action-decoder",
        choices=("clip", "tanh"),
        default=None,
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TensorNEATRunConfig:
    if args.profile == "m1":
        config = m1_profile()
    elif args.profile == "gpu":
        config = gpu_profile()
    else:
        config = TensorNEATRunConfig()

    if args.tasks is not None:
        config.task_names = tuple(args.tasks)
    if args.population_size is not None:
        config.population_size = args.population_size
    if args.species_size is not None:
        config.species_size = args.species_size
    if args.generations is not None:
        config.generations = args.generations
    if args.episodes_per_task is not None:
        config.episodes_per_task = args.episodes_per_task
    if args.max_steps is not None:
        config.max_steps_per_episode = args.max_steps
    if args.seed is not None:
        config.seed = args.seed
    if args.success_bonus is not None:
        config.success_bonus = args.success_bonus
    if args.action_decoder is not None:
        config.action_decoder = args.action_decoder
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    config.verbose = not args.quiet
    config.render = args.render
    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)

    try:
        result = run_tensorneat_metaworld(config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        "TensorNEAT run finished. "
        f"best_fitness={result.best_fitness:.3f}, "
        f"best_generation={result.best_generation}, "
        f"tasks={result.resolved_tasks}"
    )
    print(f"History: {result.history_path}")
    print(f"Best genome: {result.best_genome_path}")


if __name__ == "__main__":
    main()
