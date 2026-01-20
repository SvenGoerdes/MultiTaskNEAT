import neat
import gymnasium as gym
import numpy as np


def eval_genomes(genomes, config):
    """Fitness function: evaluates how well each genome solves MountainCar."""
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        total_fitness = 0.0
        num_episodes = 3
        
        for _ in range(num_episodes):
            env = gym.make("MountainCar-v0")
            observation, _ = env.reset()
            
            max_position = -1.2  # Track highest position reached
            max_velocity_right = 0.0  # Track max velocity going right
            steps = 0
            reached_goal = False
            
            for step in range(200):  # Max 200 steps
                output = net.activate(observation)
                # 3 outputs for 3 discrete actions: left, nothing, right
                action = int(np.argmax(output))
                
                observation, reward, terminated, truncated, _ = env.step(action)
                steps += 1
                
                position, velocity = observation
                
                # Track maximum position (how high did we get?)
                if position > max_position:
                    max_position = position
                
                # Track velocity toward goal (positive = going right)
                if velocity > max_velocity_right:
                    max_velocity_right = velocity
                
                if terminated:  # Reached goal!
                    reached_goal = True
                    break
                
                if truncated:
                    break
            
            env.close()
            
            # === FITNESS SHAPING ===
            # Base fitness: how far right did we get? (range: -1.2 to 0.5)
            # Normalize to 0-1 range: (pos + 1.2) / 1.7
            position_score = (max_position + 1.2) / 1.7 * 100  # 0 to 100
            
            # Bonus for velocity (shows the car learned to swing)
            velocity_score = max_velocity_right * 500  # 0 to ~35
            
            # Big bonus for reaching the goal
            if reached_goal:
                # Faster = better (200 - steps gives more points for fewer steps)
                goal_bonus = 200 + (200 - steps)
            else:
                goal_bonus = 0
            
            episode_fitness = position_score + velocity_score + goal_bonus
            total_fitness += episode_fitness
        
        genome.fitness = total_fitness / num_episodes


# Load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'src/config')

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