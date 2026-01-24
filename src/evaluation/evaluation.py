import gymnasium as gym
import numpy as np
import neat

def normalize_reward(shaped_fitness, env_type="CartPole-v1"):
    """
    Normalisiert die Fitness auf den Bereich [0, 1].
    Speziell angepasst an das MountainCar Fitness-Shaping.
    """
    if env_type == "CartPole-v1":
        # CartPole-v1: Standard-Belohnung ist die Anzahl der Schritte (max 500)
        r_min, r_max = 0.0, 500.0
    
    elif env_type == "MountainCar-v0":
        # Basierend auf deinem Code:
        # Position (100) + Velocity (~35) + Goal Bonus (max 400)
        # Wir setzen das Maximum auf 535, um den Bereich voll auszuschöpfen.
        r_min, r_max = 0.0, 535.0
    
    else:
        raise ValueError("Unbekannter Environment-Typ")

    # Min-Max Skalierung
    normalized = (shaped_fitness - r_min) / (r_max - r_min)
    
    # Clamping, falls durch Rundungen oder extreme Geschwindigkeiten Werte 
    # außerhalb von [0, 1] entstehen
    return max(0.0, min(1.0, normalized))


def _pareto_dominates(obj_a, obj_b):
    """
    Returns True if obj_a Pareto-dominates obj_b.
    obj_a dominates obj_b if obj_a is >= in all objectives and > in at least one.
    """
    at_least_as_good = all(a >= b for a, b in zip(obj_a, obj_b))
    strictly_better = any(a > b for a, b in zip(obj_a, obj_b))
    return at_least_as_good and strictly_better


def _compute_pareto_ranks(objectives):
    """
    Compute Pareto ranks using non-dominated sorting.
    Returns a list of ranks (0 = non-dominated front, higher = dominated).
    
    Args:
        objectives: List of tuples, each containing objective values for a genome.
        
    Returns:
        List of integer ranks corresponding to each genome.
    """
    n = len(objectives)
    ranks = [-1] * n
    remaining = set(range(n))
    current_rank = 0
    
    while remaining:
        # Find non-dominated individuals in the remaining set
        non_dominated = []
        for i in remaining:
            is_dominated = False
            for j in remaining:
                if i != j and _pareto_dominates(objectives[j], objectives[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(i)
        
        # Assign current rank to non-dominated individuals
        for i in non_dominated:
            ranks[i] = current_rank
            remaining.remove(i)
        
        current_rank += 1
    
    return ranks


def eval_genomes(genomes, config, env_1='CartPole-v1', env_2='MountainCar-v0', pareto=False):
    """
    Evaluates the genomes on multiple environments.

    Args:
        genomes (list): List of genomes to be evaluated.
        config (neat.Config): Configuration for the NEAT algorithm.
        env_1 (str): Name of the first environment.
        env_2 (str): Name of the second environment.
    
    """
    # Store environment names before creating gym objects
    env_1_name = env_1
    env_2_name = env_2

    # Wir erstellen die Environments einmal außerhalb der Genome-Schleife
    env_1 = gym.make(env_1_name)
    env_2 = gym.make(env_2_name)

    # Store objectives for each genome (needed for Pareto evaluation)
    genome_objectives = []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # --- TASK 1: CARTPOLE ---
        obs_1, _ = env_1.reset()
        fitness_1 = 0.0
        for _ in range(500): # Max Steps für CartPole
            # Input Padding: 4 Obs + 2 Nullen für MountainCar
            inputs = list(obs_1) + [0.0, 0.0]
            outputs = net.activate(inputs)
            
            # Nutze nur die ersten 2 Output-Neuronen
            action = np.argmax(outputs[0:2])
            obs_1, reward, terminated, truncated, _ = env_1.step(action)
            fitness_1 += reward
            if terminated or truncated:
                break
        
        # --- TASK 2: MOUNTAIN CAR (mit deinem Shaping) ---
        obs_2, _ = env_2.reset()
        max_pos = -1.2
        max_vel = 0.0
        steps_2 = 0
        reached_goal = False
        
        for _ in range(200):
            # Input Padding: 4 Nullen für CartPole + 2 Obs
            inputs = [0.0, 0.0, 0.0, 0.0] + list(obs_2)
            outputs = net.activate(inputs)
            
            # Nutze die Output-Neuronen Index 2, 3 und 4
            action = np.argmax(outputs[2:5])
            obs_2, _, terminated, truncated, _ = env_2.step(action)
            
            pos, vel = obs_2
            if pos > max_pos: max_pos = pos
            if vel > max_vel: max_vel = vel
            steps_2 += 1
            
            if terminated:
                reached_goal = True
                break
            if truncated:
                break
        
        # Dein Fitness-Shaping für MountainCar
        pos_score = (max_pos + 1.2) / 1.7 * 100
        vel_score = max_vel * 500
        goal_bonus = (200 + (200 - steps_2)) if reached_goal else 0
        fitness_2 = pos_score + vel_score + goal_bonus

        # --- NORMALISIERUNG ---
        norm_1 = normalize_reward(fitness_1, env_1_name)
        norm_2 = normalize_reward(fitness_2, env_2_name)

        # Store normalized objectives for this genome
        genome_objectives.append((norm_1, norm_2))

    env_1.close()
    env_2.close()

    # --- FITNESS ASSIGNMENT ---
    if pareto:
        # Pareto efficiency: assign fitness based on Pareto rank
        # Lower rank = better (non-dominated front has rank 0)
        ranks = _compute_pareto_ranks(genome_objectives)
        max_rank = max(ranks) if ranks else 0
        
        for i, (genome_id, genome) in enumerate(genomes):
            # Convert rank to fitness: higher fitness for lower rank
            # Fitness ranges from 0 (worst rank) to 1 (rank 0)
            if max_rank > 0:
                genome.fitness = 1.0 - (ranks[i] / max_rank)
            else:
                genome.fitness = 1.0  # All genomes are non-dominated
    else:
        # Standard: arithmetic mean of objectives
        for i, (genome_id, genome) in enumerate(genomes):
            norm_1, norm_2 = genome_objectives[i]
            genome.fitness = (norm_1 + norm_2) / 2.0