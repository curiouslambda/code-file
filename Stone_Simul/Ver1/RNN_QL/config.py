class Config:
    # Environment configuration
    success_prob = 0.75
    min_prob = 0.25
    max_prob = 0.75
    num_categories = 3
    num_trials = 30

    # Q-learning configuration
    num_episodes = 10
    epsilon_decay = 0.99
    alpha = 0.1
    gamma = 0.99

    # RNN configuration
    hidden_size = 50
    input_size = 5  # 5개의 특징
    output_size = 2  # 골라야 할 카테고리, 승리할 확률
