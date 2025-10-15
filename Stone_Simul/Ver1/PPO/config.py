# config.py

import torch

class Config:
    # 게임 설정
    max_attempts = 10
    max_choices = 30
    initial_success_rate = 0.75
    min_success_rate = 0.25
    max_success_rate = 0.75
    success_rate_change = 0.1
    
    # DQN 설정
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    learning_rate = 0.001
    batch_size = 64
    target_update = 10
    memory_size = 10000

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PPO 설정
    clip_param = 0.2  # Clipping parameter for PPO
    ppo_epochs = 10  # Number of epochs to update policy
    ppo_batch_size = 64  # Batch size for PPO updates
    learning_rate = 3e-4  # Learning rate for optimizer
    gamma = 0.99  # Discount factor
    lam = 0.95  # GAE lambda
    entropy_coef = 0.01  # Entropy coefficient
    value_loss_coef = 0.5  # Value loss coefficient
    max_grad_norm = 0.5  # Max gradient norm for clipping