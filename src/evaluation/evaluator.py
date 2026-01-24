import gymnasium as gym
import neat
from typing import List
from .environment import Environment


class MultiEnvironmentEvaluator:
    """
    Evaluates NEAT genomes across multiple environments using a shared neural network.

    The evaluator uses input and output padding to allow a single network to handle
    multiple environments with different observation and action space sizes.

    Input layout: [env1_obs | env2_obs | ... | envN_obs]
    Output layout: [env1_actions | env2_actions | ... | envN_actions]

    When evaluating on a specific environment, other environment's inputs are padded
    with zeros, and only that environment's output neurons are used for actions.
    """

    def __init__(self, environments: List[Environment]):
        """
        Initialize the multi-environment evaluator.

        Args:
            environments: List of Environment instances to evaluate on
        """
        self.environments = environments

        # Calculate total dimensions
        self.total_input_dims = sum(env.observation_space_size
                                    for env in environments)
        self.total_output_dims = sum(env.action_space_size
                                     for env in environments)

        # Assign input and output offsets to each environment, and set total dims
        input_offset = 0
        output_offset = 0
        for env in environments:
            env.input_offset = input_offset
            env.output_offset = output_offset
            env.total_input_dims = self.total_input_dims
            env.total_output_dims = self.total_output_dims
            input_offset += env.observation_space_size
            output_offset += env.action_space_size

    def eval_genomes(self, genomes, config):
        """
        Evaluate all genomes across all environments.

        This function can be passed directly to NEAT's Population.run() method.

        For each genome:
        1. Create a neural network from the genome
        2. Evaluate on each environment
        3. Normalize each fitness score to [0, 1]
        4. Combine normalized scores using arithmetic mean

        Args:
            genomes: List of (genome_id, genome) tuples from NEAT
            config: NEAT configuration object
        """
        # Create all Gymnasium environments once
        gym_envs = {env: gym.make(env.gym_env_id) for env in self.environments}

        try:
            for genome_id, genome in genomes:
                # Create neural network from genome
                net = neat.nn.FeedForwardNetwork.create(genome, config)

                # Evaluate on each environment
                normalized_fitnesses = []
                for env in self.environments:
                    gym_env = gym_envs[env]

                    # Run evaluation and get shaped fitness
                    fitness = env.evaluate(net, gym_env)

                    # Normalize to [0, 1]
                    normalized_fitness = env.normalize_fitness(fitness)
                    normalized_fitnesses.append(normalized_fitness)

                # Combine normalized fitnesses using arithmetic mean
                genome.fitness = sum(normalized_fitnesses) / len(normalized_fitnesses)

        finally:
            # Clean up environments
            for gym_env in gym_envs.values():
                gym_env.close()

    def get_total_input_dims(self) -> int:
        """Get the total number of input dimensions for the shared network."""
        return self.total_input_dims

    def get_total_output_dims(self) -> int:
        """Get the total number of output dimensions for the shared network."""
        return self.total_output_dims

    def get_environment_summary(self) -> str:
        """
        Get a human-readable summary of the environment configuration.

        Returns:
            Multi-line string describing the environment setup
        """
        lines = [
            "Multi-Environment Configuration:",
            f"  Total Inputs: {self.total_input_dims}",
            f"  Total Outputs: {self.total_output_dims}",
            ""
        ]

        for i, env in enumerate(self.environments, 1):
            lines.extend([
                f"  Environment {i}: {env.name} ({env.gym_env_id})",
                f"    Observations: {env.observation_space_size} "
                f"(inputs {env.input_offset}-{env.input_offset + env.observation_space_size - 1})",
                f"    Actions: {env.action_space_size} "
                f"(outputs {env.output_offset}-{env.output_offset + env.action_space_size - 1})",
                f"    Max Steps: {env.max_steps}",
                f"    Fitness Range: {env.get_normalization_range()}",
                ""
            ])

        return "\n".join(lines)
