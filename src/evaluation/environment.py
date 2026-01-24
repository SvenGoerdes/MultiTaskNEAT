from abc import ABC, abstractmethod
from typing import Tuple
import gymnasium as gym


class Environment(ABC):
    """
    Abstract base class for NEAT training environments.

    Each environment encapsulates the logic for evaluating a neural network
    on a specific Gymnasium environment, including fitness shaping and
    normalization parameters.
    """

    def __init__(self, name: str, gym_env_id: str, observation_space_size: int,
                 action_space_size: int, max_steps: int):
        """
        Initialize the environment.

        Args:
            name: Human-readable environment identifier
            gym_env_id: Gymnasium environment string (e.g., "CartPole-v1")
            observation_space_size: Number of inputs from this environment
            action_space_size: Number of outputs for this environment
            max_steps: Maximum steps per episode
        """
        self.name = name
        self.gym_env_id = gym_env_id
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.max_steps = max_steps
        self.input_offset = 0  # Will be set by MultiEnvironmentEvaluator
        self.output_offset = 0  # Will be set by MultiEnvironmentEvaluator
        self.total_input_dims = 0  # Will be set by MultiEnvironmentEvaluator
        self.total_output_dims = 0  # Will be set by MultiEnvironmentEvaluator

    @abstractmethod
    def evaluate(self, net, gym_env: gym.Env) -> float:
        """
        Run one episode and return the shaped fitness score.

        Args:
            net: NEAT neural network
            gym_env: Initialized Gymnasium environment

        Returns:
            Shaped fitness score (before normalization)
        """
        pass

    @abstractmethod
    def get_normalization_range(self) -> Tuple[float, float]:
        """
        Return the (min, max) range for fitness normalization.

        Returns:
            Tuple of (min_fitness, max_fitness)
        """
        pass

    def normalize_fitness(self, fitness: float) -> float:
        """
        Normalize fitness to [0, 1] range using min-max scaling.

        Args:
            fitness: Raw fitness score

        Returns:
            Normalized fitness clamped to [0, 1]
        """
        r_min, r_max = self.get_normalization_range()
        normalized = (fitness - r_min) / (r_max - r_min)
        return max(0.0, min(1.0, normalized))
