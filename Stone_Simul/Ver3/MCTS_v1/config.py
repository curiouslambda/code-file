# 게임 환경 설정
GAME_CONFIG = {
    "initial_success_prob": 0.75,  # 초기 성공 확률
    "max_success_prob": 0.75,     # 최대 성공 확률
    "min_success_prob": 0.25,     # 최소 성공 확률
    "max_attempts_per_category": 10,  # 각 카테고리별 최대 시도 횟수
    "categories": ["A", "B", "C"],    # 카테고리 리스트
    "reward_win": 100,           # 승리 시 보상
    "reward_loss": -100,         # 패배 시 보상
    "reward_step_success": 1,    # A, B의 성공 시 보상
    "reward_step_fail": -1,      # C의 성공 시 패널티
    "failure_threshold": {       # 실패 조건
        "C": 5,                  # C의 성공이 5 이상일 때 종료
        "A": 4,                  # A의 실패가 4 이상일 때 종료
        "B": 4,                  # B의 실패가 4 이상일 때 종료
    }
}

# MCTS 설정
MCTS_CONFIG = {
    "simulations": 3,        # MCTS 시뮬레이션 횟수
    "exploration_weight": 1.0  # 탐험-이용 균형을 조정하는 가중치 (UCB1)
}

# 기타 설정
TRAINING_CONFIG = {
    "episodes": 10,            # 학습 에피소드 수
}
