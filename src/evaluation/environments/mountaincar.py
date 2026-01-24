import numpy as np
import gymnasium as gym
from typing import Tuple
from ..environment import Environment


class MountainCarEnvironment(Environment):
    """
    MountainCar-v0 environment for NEAT training.

    The agent must drive a car up a steep hill by building momentum.
    Fitness uses custom shaping based on position, velocity, and goal achievement.
    """

    def __init__(self):
        super().__init__(
            name="MountainCar",
            gym_env_id="MountainCar-v0",
            observation_space_size=2,
            action_space_size=3,
            max_steps=200
        )

    def evaluate(self, net, gym_env: gym.Env) -> float:
        """
        Run one MountainCar episode with custom fitness shaping.

        Fitness components:
        - Position score: Rewards progress towards the goal (0-100)
        - Velocity score: Rewards higher velocities (~0-35)
        - Goal bonus: Large bonus if goal is reached (200-400 based on steps)

        Args:
            net: NEAT neural network
            gym_env: Initialized MountainCar-v0 environment

        Returns:
            Shaped fitness score
        """
        obs, _ = gym_env.reset()
        max_pos = -1.2  # Starting position
        max_vel = 0.0
        steps = 0
        reached_goal = False

        for _ in range(self.max_steps):
            # Create input vector with zeros for all inputs
            inputs = [0.0] * self.total_input_dims

            # Place MountainCar observations at the correct offset
            for i, val in enumerate(obs):
                inputs[self.input_offset + i] = val

            # Get network outputs
            outputs = net.activate(inputs)

            # Use only the outputs for this environment
            action_outputs = outputs[self.output_offset:
                                     self.output_offset + self.action_space_size]
            action = int(np.argmax(action_outputs))

            # Step the environment
            obs, _, terminated, truncated, _ = gym_env.step(action)

            # Track statistics for fitness shaping
            pos, vel = obs
            if pos > max_pos:
                max_pos = pos
            if vel > max_vel:
                max_vel = vel
            steps += 1

            if terminated:
                reached_goal = True
                break
            if truncated:
                break

        # Custom fitness shaping
        # Position score: normalized progress from -1.2 to 0.5 (goal), scaled to 0-100
        pos_score = (max_pos + 1.2) / 1.7 * 100

        # Velocity score: reward higher velocities (scaled by 500)
        vel_score = max_vel * 500

        # Goal bonus: large reward if goal reached, with bonus for speed
        goal_bonus = (200 + (200 - steps)) if reached_goal else 0

        fitness = pos_score + vel_score + goal_bonus
        return fitness

    def get_normalization_range(self) -> Tuple[float, float]:
        """
        Return normalization range for MountainCar fitness.

        Based on fitness shaping:
        - Position score: max 100
        - Velocity score: max ~35 (velocity range is ~0.07)
        - Goal bonus: max 400 (reached in 0 steps)
        Total theoretical max: ~535

        Returns:
            (0.0, 535.0)
        """
        return (0.0, 535.0)
