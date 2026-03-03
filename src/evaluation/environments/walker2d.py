import numpy as np
import gymnasium as gym
from typing import Tuple
from ..environment import Environment


class Walker2DEnvironment(Environment):
    """
    Brax Walker2D environment for NEAT training.

    The agent controls a bipedal walker to move forward by applying
    continuous torques to its joints. Uses tanh + clip for action decoding.
    """

    def __init__(self):
        super().__init__(
            name="Walker2D",
            gym_env_id="Walker2d-v4",
            observation_space_size=17,
            action_space_size=6,
            max_steps=1000
        )

    def evaluate(self, net, gym_env: gym.Env) -> float:
        """
        Run one Walker2D episode and return the fitness.

        Uses continuous action decoding: tanh activation followed by
        clipping to the environment's action bounds.

        Args:
            net: NEAT neural network
            gym_env: Initialized Walker2D environment

        Returns:
            Fitness score (sum of rewards)
        """
        obs, _ = gym_env.reset()
        fitness = 0.0

        action_low = gym_env.action_space.low
        action_high = gym_env.action_space.high

        for _ in range(self.max_steps):
            inputs = [0.0] * self.total_input_dims

            for i, val in enumerate(obs):
                inputs[self.input_offset + i] = float(val)

            outputs = net.activate(inputs)

            action_outputs = outputs[self.output_offset:
                                     self.output_offset + self.action_space_size]
            action = np.tanh(action_outputs)
            action = np.clip(action, action_low, action_high)

            obs, reward, terminated, truncated, _ = gym_env.step(action)
            fitness += reward

            if terminated or truncated:
                break

        return fitness

    def get_normalization_range(self) -> Tuple[float, float]:
        """
        Return normalization range for Walker2D fitness.

        Walker2d-v4 typically yields rewards in the range of 0 to ~5000+.
        Using a conservative upper bound.

        Returns:
            (0.0, 5000.0)
        """
        return (0.0, 5000.0)
