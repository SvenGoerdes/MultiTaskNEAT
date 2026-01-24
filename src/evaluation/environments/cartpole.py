import numpy as np
import gymnasium as gym
from typing import Tuple
from ..environment import Environment


class CartPoleEnvironment(Environment):
    """
    CartPole-v1 environment for NEAT training.

    The agent must balance a pole on a cart by moving left or right.
    Fitness is the sum of rewards (number of steps the pole stays balanced).
    """

    def __init__(self):
        super().__init__(
            name="CartPole",
            gym_env_id="CartPole-v1",
            observation_space_size=4,
            action_space_size=2,
            max_steps=500
        )

    def evaluate(self, net, gym_env: gym.Env) -> float:
        """
        Run one CartPole episode and return the fitness.

        The neural network receives the 4 CartPole observations at the start,
        followed by zeros padding for other environments.

        Args:
            net: NEAT neural network
            gym_env: Initialized CartPole-v1 environment

        Returns:
            Fitness score (sum of rewards, typically number of steps)
        """
        obs, _ = gym_env.reset()
        fitness = 0.0

        for _ in range(self.max_steps):
            # Create input vector with zeros for all inputs
            inputs = [0.0] * self.total_input_dims

            # Place CartPole observations at the correct offset
            for i, val in enumerate(obs):
                inputs[self.input_offset + i] = val

            # Get network outputs
            outputs = net.activate(inputs)

            # Use only the outputs for this environment
            action_outputs = outputs[self.output_offset:
                                     self.output_offset + self.action_space_size]
            action = int(np.argmax(action_outputs))

            # Step the environment
            obs, reward, terminated, truncated, _ = gym_env.step(action)
            fitness += reward

            if terminated or truncated:
                break

        return fitness

    def get_normalization_range(self) -> Tuple[float, float]:
        """
        Return normalization range for CartPole fitness.

        CartPole-v1 has a maximum of 500 steps, so fitness ranges from 0 to 500.

        Returns:
            (0.0, 500.0)
        """
        return (0.0, 500.0)
