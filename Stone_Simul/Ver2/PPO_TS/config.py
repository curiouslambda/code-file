# config.py

class Config:
    def __init__(self):
        self.learning_rate = 0.0003
        self.gamma = 0.95
        self.ppo_clip = 0.2
        self.update_timestep = 2000
        self.max_timesteps = 1000000
        self.betas = (0.9, 0.999)
        self.eps_clip = 0.2
        self.env_name = "GameEnv"
        self.alpha_init = 7.5  # Initial alpha for Thompson Sampling (for success rate of 0.75)
        self.beta_init = 2.5   # Initial beta for Thompson Sampling (for success rate of 0.25)
        self.action_space = 3  # A, B, C
        self.state_space = 10  # Custom state space for A, B, C attempts, success, failures, and success prob
        self.max_game_rounds = 30  # Total number of game opportunities
