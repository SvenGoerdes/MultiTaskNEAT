import gymnasium as gym

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

def eval_genomes(genomes, config):
    # Wir erstellen die Environments einmal außerhalb der Genome-Schleife
    env_cp = gym.make("CartPole-v1")
    env_mc = gym.make("MountainCar-v0")

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # --- TASK 1: CARTPOLE ---
        obs_cp, _ = env_cp.reset()
        fitness_cp = 0.0
        for _ in range(500): # Max Steps für CartPole
            # Input Padding: 4 Obs + 2 Nullen für MountainCar
            inputs = list(obs_cp) + [0.0, 0.0]
            outputs = net.activate(inputs)
            
            # Nutze nur die ersten 2 Output-Neuronen
            action = np.argmax(outputs[0:2])
            obs_cp, reward, terminated, truncated, _ = env_cp.step(action)
            fitness_cp += reward
            if terminated or truncated:
                break
        
        # --- TASK 2: MOUNTAIN CAR (mit deinem Shaping) ---
        obs_mc, _ = env_mc.reset()
        max_pos = -1.2
        max_vel = 0.0
        steps_mc = 0
        reached_goal = False
        
        for _ in range(200):
            # Input Padding: 4 Nullen für CartPole + 2 Obs
            inputs = [0.0, 0.0, 0.0, 0.0] + list(obs_mc)
            outputs = net.activate(inputs)
            
            # Nutze die Output-Neuronen Index 2, 3 und 4
            action = np.argmax(outputs[2:5])
            obs_mc, _, terminated, truncated, _ = env_mc.step(action)
            
            pos, vel = obs_mc
            if pos > max_pos: max_pos = pos
            if vel > max_vel: max_vel = vel
            steps_mc += 1
            
            if terminated:
                reached_goal = True
                break
            if truncated:
                break
        
        # Dein Fitness-Shaping für MountainCar
        pos_score = (max_pos + 1.2) / 1.7 * 100
        vel_score = max_vel * 500
        goal_bonus = (200 + (200 - steps_mc)) if reached_goal else 0
        fitness_mc = pos_score + vel_score + goal_bonus

        # --- NORMALISIERUNG & KOMBINATION ---
        norm_cp = normalize_reward(fitness_cp, "cartpole")
        norm_mc = normalize_reward(fitness_mc, "mountaincar")

        # Wir nutzen das arithmetische Mittel: 0.5 * CP + 0.5 * MC
        # So kann die Evolution erst eine Aufgabe lernen, ohne sofort auszusterben.
        genome.fitness = (norm_cp + norm_mc) / 2.0

    env_cp.close()
    env_mc.close()