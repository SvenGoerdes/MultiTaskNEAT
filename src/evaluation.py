import gymnasium as gym

env_cartpole = gym.make("CartPole-v1")
env_mountaincar = gym.make("MountainCar-v0")


def normalize_reward(reward, env_type="cartpole"):
    """
    Normalisiert den Reward/Fitness auf den Bereich [0, 1].
    
    Args:
        reward (float): Der rohe Reward oder die shaped Fitness.
        env_type (str): "cartpole" oder "mountaincar".
    """
    if env_type == "cartpole":
        # CartPole-v1: Reward geht von 0 bis 500
        r_min, r_max = 0, 500
    
    elif env_type == "mountaincar":
        # Falls du Standard-Rewards nutzt: -200 bis 0
        # Falls du mein Shaping nutzt (max_pos + 0.5): ca. -0.7 bis 1.1
        # Wir definieren hier weite Grenzen, um sicher zu gehen:
        r_min, r_max = -200.0, 1.1 
    else:
        raise ValueError("Unbekannter Environment-Typ")

    # Min-Max Skalierung
    normalized = (reward - r_min) / (r_max - r_min)
    
    # Sicherstellen, dass der Wert strikt in [0, 1] bleibt (Clamping)
    return max(0.0, min(1.0, normalized))

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # --- TASK 1: CartPole ---
        observation = env_cartpole.reset()
        # WICHTIG: MountainCar Inputs (2 Stück) mit 0.0 auffüllen!
        inputs = list(observation) + [0.0, 0.0] 
        # ... Run simulation ...
        # Nur Output-Neuronen 0 und 1 abfragen
        score_cartpole = fitness_cartpole
        score_cartpole = normalize_reward(score_cartpole, env_type="cartpole")
        
        # --- TASK 2: MountainCar ---
        observation = env_mountaincar.reset()
        # WICHTIG: CartPole Inputs (4 Stück) vorne mit 0.0 auffüllen!
        inputs = [0.0, 0.0, 0.0, 0.0] + list(observation)
        # ... Run simulation ...
        # Nur Output-Neuronen 2, 3 und 4 abfragen
        score_mountaincar = fitness_mountaincar_shaped # Dein Shaping von vorhin!
        score_mountaincar = normalize_reward(score_mountaincar, env_type="mountaincar")        
        # --- KOMBINATION ---
        # Hier entscheidet sich, ob wir einen Multi-Tasker züchten oder einen Spezialisten.
        # Vorschlag: Multiplikation bestraft Nullen extrem hart.
        # Wenn er CartPole perfekt kann (500) aber MountainCar nicht (0), ist die Fitness 0.
        genome.fitness = score_cartpole * score_mountaincar