# config.py

class Config:
    # 게임 설정
    max_attempts = 10
    max_choices = 30
    initial_success_rate = 0.75
    min_success_rate = 0.25
    max_success_rate = 0.75
    success_rate_change = 0.1
    total_episodes = 100
    initial_steps = (total_episodes*30)/4
    
    # DQN 설정
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    learning_rate = 0.001
    batch_size = 64
    target_update = 10
    memory_size = 10000
