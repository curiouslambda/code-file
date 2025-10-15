# config.py

# 환경 설정
GRID_SIZE = 5
START_STATE = (0, 0)
GOAL_STATE = (4, 4)

# 에이전트 설정
AGENT_PARAMS = {
    "STATE_SIZE": 2,
    "ACTION_SIZE": 4,  # 고정된 액션 크기를 가정합니다.
    "REPLAY_MEMORY_CAPACITY": 2000,
    "BATCH_SIZE": 32,
    "GAMMA": 0.95,
    "EPSILON": 0.05,
    "EPSILON_MIN": 0.01,
    "EPSILON_DECAY": 0.995,
    "LEARNING_RATE": 0.001,
    "HIDDEN_LAYER_SIZE": 24,  # 은닉층의 뉴런 수
    "EPISODE" : 5000
}

# 파일 경로
SAVED_MODEL_PATH = 'saved_model.pth'
SAVED_PLOT_PATH = 'dqn_learning.png'
