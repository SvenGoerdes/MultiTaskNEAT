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
winner = p.run(eval_genomes, 100)

# Test the winner visually
print('\n--- Best Genome ---')
print(winner)

# Show the winner in action
print('\n--- Testing Winner (MountainCar) ---')
env = gym.make("MountainCar-v0", render_mode="human")
net = neat.nn.FeedForwardNetwork.create(winner, config)
observation, _ = env.reset()
steps = 0

while True:
    output = net.activate(observation)
    action = int(np.argmax(output))
    observation, reward, terminated, truncated, _ = env.step(action)
    steps += 1
    
    if terminated:
        print(f'🎉 Goal reached in {steps} steps!')
        break
    if truncated:
        print(f'Episode ended after {steps} steps (did not reach goal)')
        break

env.close()