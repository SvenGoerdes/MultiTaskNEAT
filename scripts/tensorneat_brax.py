import argparse

from training.tensorneat_runner import (
    TensorNEATRunConfig,
    brax_cpu_profile,
    brax_gpu_profile,
    run_tensorneat_brax,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TensorNEAT on Brax environments (Hopper, Walker2D)."
    )
    parser.add_argument(
        "--profile",
        choices=("cpu", "gpu", "custom"),
        default="cpu",
        help="Hardware profile: cpu (macOS/local), gpu (CUDA server), custom.",
    )
    parser.add_argument("--envs", nargs="+", default=None,
                        help="Brax environment names (e.g. hopper walker2d)")
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--species-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--episodes-per-env", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--action-decoder",
        choices=("clip", "tanh"),
        default=None,
    )
    parser.add_argument("--brax-backend", type=str, default=None,
                        help="Brax physics backend (e.g. positional, spring, generalized, mjx)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TensorNEATRunConfig:
    if args.profile == "cpu":
        config = brax_cpu_profile()
    elif args.profile == "gpu":
        config = brax_gpu_profile()
    else:
        config = TensorNEATRunConfig()

    if args.envs is not None:
        config.env_names = tuple(args.envs)
    if args.population_size is not None:
        config.population_size = args.population_size
    if args.species_size is not None:
        config.species_size = args.species_size
    if args.generations is not None:
        config.generations = args.generations
    if args.episodes_per_env is not None:
        config.episodes_per_env = args.episodes_per_env
    if args.max_steps is not None:
        config.max_steps_per_episode = args.max_steps
    if args.seed is not None:
        config.seed = args.seed
    if args.action_decoder is not None:
        config.action_decoder = args.action_decoder
    if args.brax_backend is not None:
        config.brax_backend = args.brax_backend
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    config.verbose = not args.quiet
    config.render = args.render
    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)

    try:
        result = run_tensorneat_brax(config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        "TensorNEAT run finished. "
        f"best_fitness={result.best_fitness:.3f}, "
        f"best_generation={result.best_generation}, "
        f"envs={result.env_names}"
    )
    print(f"History: {result.history_path}")
    print(f"Best genome: {result.best_genome_path}")


if __name__ == "__main__":
    main()
