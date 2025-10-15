# config.py

class Config:
    # 게임 관련 파라미터
    MAX_ATTEMPTS = 10  # 카테고리별 최대 시도 횟수
    START_SUCCESS_PROB = 0.75  # 초기 성공 확률
    SUCCESS_PROB_STEP = 0.1  # 성공 혹은 실패 시 성공 확률 증감 폭
    MIN_SUCCESS_PROB = 0.25  # 최소 성공 확률
    MAX_SUCCESS_PROB = 0.75  # 최대 성공 확률

    REQUIRED_SUCCESS_A = 7  # A 카테고리 승리 조건(최소 성공 수)
    REQUIRED_SUCCESS_B = 7  # B 카테고리 승리 조건(최소 성공 수)
    MAX_SUCCESS_C = 4       # C 카테고리 승리 조건(최대 성공 수)

    # 조기 종료 조건
    MAX_FAILURE_A = 4  # A에서 이 이상 실패 시 게임 종료 (실패 조건 충족)
    MAX_FAILURE_B = 4  # B에서 이 이상 실패 시 게임 종료 (실패 조건 충족)
    MAX_SUCCESS_C_FAIL = 5  # C에서 이 이상 성공 시 게임 종료 (실패 조건 충족)

    TOTAL_CATEGORIES = ["A", "B", "C"]  # 카테고리 목록

    # MCTS 관련 파라미터
    MCTS_SIMULATIONS = 1000  # MCTS 시뮬레이션 횟수
    EXPLORATION_CONSTANT = 1.4  # UCB1 계수
    ROLLOUT_DEPTH = 10  # Rollout 제한(샘플)
    
    # 랜덤 시드 등을 고정하고 싶다면 설정
    SEED = 42
