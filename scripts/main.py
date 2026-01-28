import neat
import gymnasium as gym
import numpy as np
import importlib.resources
from evaluation.evaluation import eval_genomes


# "config" ist der Ordnername in src, "config-feedforward" der Dateiname
with importlib.resources.path("config", "config") as config_path:
    config = neat.Config(
        neat.DefaultGenome, 
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet, 
        neat.DefaultStagnation,
        str(config_path) 
    )


# # Load configuration
# config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                      'src/config')

# Create population
p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Run evolution for up to 100 generations
# Using arithmetic mean fitness from both CartPole and MountainCar
winner = p.run(lambda genomes, config: eval_genomes(genomes, config,
                                                     env_1='CartPole-v1',
                                                     env_2='MountainCar-v0',
                                                     pareto=False), 100)

# Test the winner visually
print('\n--- Best Genome ---')
print(winner)

# Show the winner in action on both environments
net = neat.nn.FeedForwardNetwork.create(winner, config)

# Test on CartPole
print('\n--- Testing Winner (CartPole-v1) ---')
env_cartpole = gym.make("CartPole-v1", render_mode="human")
observation, _ = env_cartpole.reset()
steps_cartpole = 0

while True:
    # Input: CartPole observations + zero padding for MountainCar
    inputs = list(observation) + [0.0, 0.0]
    output = net.activate(inputs)
    # Use first 2 output neurons for CartPole
    action = int(np.argmax(output[0:2]))
    observation, reward, terminated, truncated, _ = env_cartpole.step(action)
    steps_cartpole += 1

    if terminated or truncated:
        print(f'CartPole episode ended after {steps_cartpole} steps')
        break

env_cartpole.close()

# Test on MountainCar
print('\n--- Testing Winner (MountainCar-v0) ---')
env_mountaincar = gym.make("MountainCar-v0", render_mode="human")
observation, _ = env_mountaincar.reset()
steps_mountaincar = 0

while True:
    # Input: zero padding for CartPole + MountainCar observations
    inputs = [0.0, 0.0, 0.0, 0.0] + list(observation)
    output = net.activate(inputs)
    # Use output neurons 2, 3, 4 for MountainCar
    action = int(np.argmax(output[2:5]))
    observation, reward, terminated, truncated, _ = env_mountaincar.step(action)
    steps_mountaincar += 1

    if terminated:
        print(f'Goal reached in {steps_mountaincar} steps!')
        break
    if truncated:
        print(f'Episode ended after {steps_mountaincar} steps (did not reach goal)')
        break

env_mountaincar.close()