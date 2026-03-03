import neat
import gymnasium as gym
import numpy as np
import importlib.resources
from evaluation.environments import HopperEnvironment, Walker2DEnvironment
from evaluation.evaluator import MultiEnvironmentEvaluator

# Load configuration
with importlib.resources.path("config", "config") as config_path:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path)
    )

# Create environment instances
environments = [
    HopperEnvironment(),
    Walker2DEnvironment()
]

# Initialize multi-environment evaluator
evaluator = MultiEnvironmentEvaluator(environments)

# Print environment configuration
print(evaluator.get_environment_summary())

# Verify configuration matches NEAT config
assert config.genome_config.num_inputs == evaluator.get_total_input_dims(), \
    f"Config num_inputs ({config.genome_config.num_inputs}) != " \
    f"total input dims ({evaluator.get_total_input_dims()})"
assert config.genome_config.num_outputs == evaluator.get_total_output_dims(), \
    f"Config num_outputs ({config.genome_config.num_outputs}) != " \
    f"total output dims ({evaluator.get_total_output_dims()})"

# Create population
p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Run evolution for up to 100 generations
winner = p.run(evaluator.eval_genomes, 100)

# Test the winner visually
print('\n--- Best Genome ---')
print(winner)

# Create network from winner
net = neat.nn.FeedForwardNetwork.create(winner, config)

# Test on Hopper
print('\n--- Testing Winner (Hopper-v4) ---')
env_hopper = gym.make("Hopper-v4", render_mode="human")
hopper_env = environments[0]  # HopperEnvironment instance

observation, _ = env_hopper.reset()
total_reward = 0.0
steps = 0

while True:
    inputs = [0.0] * evaluator.get_total_input_dims()
    for i, val in enumerate(observation):
        inputs[hopper_env.input_offset + i] = float(val)
    outputs = net.activate(inputs)

    action_outputs = outputs[hopper_env.output_offset:
                             hopper_env.output_offset + hopper_env.action_space_size]
    action = np.tanh(action_outputs)
    action = np.clip(action, env_hopper.action_space.low, env_hopper.action_space.high)

    observation, reward, terminated, truncated, _ = env_hopper.step(action)
    total_reward += reward
    steps += 1

    if terminated or truncated:
        print(f'Episode ended after {steps} steps, total reward: {total_reward:.2f}')
        break

env_hopper.close()

# Test on Walker2D
print('\n--- Testing Winner (Walker2d-v4) ---')
env_walker = gym.make("Walker2d-v4", render_mode="human")
walker_env = environments[1]  # Walker2DEnvironment instance

observation, _ = env_walker.reset()
total_reward = 0.0
steps = 0

while True:
    inputs = [0.0] * evaluator.get_total_input_dims()
    for i, val in enumerate(observation):
        inputs[walker_env.input_offset + i] = float(val)
    outputs = net.activate(inputs)

    action_outputs = outputs[walker_env.output_offset:
                             walker_env.output_offset + walker_env.action_space_size]
    action = np.tanh(action_outputs)
    action = np.clip(action, env_walker.action_space.low, env_walker.action_space.high)

    observation, reward, terminated, truncated, _ = env_walker.step(action)
    total_reward += reward
    steps += 1

    if terminated or truncated:
        print(f'Episode ended after {steps} steps, total reward: {total_reward:.2f}')
        break

env_walker.close()
