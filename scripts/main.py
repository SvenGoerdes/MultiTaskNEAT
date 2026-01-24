import neat
import gymnasium as gym
import numpy as np
import importlib.resources
from evaluation.environments import CartPoleEnvironment, MountainCarEnvironment
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
    CartPoleEnvironment(),
    MountainCarEnvironment()
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

# Test on CartPole
print('\n--- Testing Winner (CartPole-v1) ---')
env_cartpole = gym.make("CartPole-v1", render_mode="human")
cartpole_env = environments[0]  # CartPoleEnvironment instance

observation, _ = env_cartpole.reset()
steps = 0

while True:
    # Use proper input padding for CartPole
    inputs = list(observation) + [0.0] * (evaluator.get_total_input_dims() - len(observation))
    outputs = net.activate(inputs)

    # Use CartPole's output slice
    action_outputs = outputs[cartpole_env.output_offset:
                             cartpole_env.output_offset + cartpole_env.action_space_size]
    action = int(np.argmax(action_outputs))

    observation, reward, terminated, truncated, _ = env_cartpole.step(action)
    steps += 1

    if terminated or truncated:
        print(f'Episode ended after {steps} steps')
        break

env_cartpole.close()

# Test on MountainCar
print('\n--- Testing Winner (MountainCar-v0) ---')
env_mountaincar = gym.make("MountainCar-v0", render_mode="human")
mountaincar_env = environments[1]  # MountainCarEnvironment instance

observation, _ = env_mountaincar.reset()
steps = 0

while True:
    # Use proper input padding for MountainCar
    inputs = [0.0] * mountaincar_env.input_offset + list(observation)
    outputs = net.activate(inputs)

    # Use MountainCar's output slice
    action_outputs = outputs[mountaincar_env.output_offset:
                             mountaincar_env.output_offset + mountaincar_env.action_space_size]
    action = int(np.argmax(action_outputs))

    observation, reward, terminated, truncated, _ = env_mountaincar.step(action)
    steps += 1

    if terminated:
        print(f'🎉 Goal reached in {steps} steps!')
        break
    if truncated:
        print(f'Episode ended after {steps} steps (did not reach goal)')
        break

env_mountaincar.close()