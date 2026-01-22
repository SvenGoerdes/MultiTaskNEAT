import gymnasium as gym
import neat
env_cartpole = gym.make("CartPole-v1")
env_mountaincar = gym.make("MountainCar-v0")


def normalize_reward(shaped_fitness, env_type="cartpole"):
    """
    Normalisiert die Fitness auf den Bereich [0, 1].
    Speziell angepasst an das MountainCar Fitness-Shaping.
    """
    if env_type == "cartpole":
        # CartPole-v1: Standard-Belohnung ist die Anzahl der Schritte (max 500)
        r_min, r_max = 0.0, 500.0
    
    elif env_type == "mountaincar":
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

def eval_genomes(genomes, config, env_1='CartPole-v1', env_2='MountainCar-v0'):
    """
    Evaluates the genomes on multiple environments.

    Args:
        genomes (list): List of genomes to be evaluated.
        config (neat.Config): Configuration for the NEAT algorithm.
        env_1 (str): Name of the first environment.
        env_2 (str): Name of the second environment.
    
    """
    # Wir erstellen die Environments einmal außerhalb der Genome-Schleife
    env_1 = gym.make(env_1)
    env_2 = gym.make(env_2)

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

        # --- NORMALISIERUNG & KOMBINATION ---
        norm_1 = normalize_reward(fitness_1, "cartpole")
        norm_2 = normalize_reward(fitness_2, "mountaincar")

        # Wir nutzen das arithmetische Mittel: 0.5 * CP + 0.5 * MC
        # So kann die Evolution erst eine Aufgabe lernen, ohne sofort auszusterben.
        genome.fitness = (norm_1 + norm_2) / 2.0

    env_1.close()
    env_2.close()