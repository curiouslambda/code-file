# config.py

class Config:
    INPUT_SIZE = 7  # 각 카테고리의 남은 시도 횟수 3개, 성공 횟수 3개, 현재 성공 확률 1개
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 3  # 가능한 행동의 수 (A, B, C)
    LEARNING_RATE = 0.001
    NUM_EPISODES = 10
    NUM_EPOCHS = 100
