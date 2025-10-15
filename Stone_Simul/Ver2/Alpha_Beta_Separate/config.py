# config.py

class Config:
    # 게임 환경 설정
    total_trials = 30  # 총 시도 횟수
    max_attempts_per_category = 10  # 각 카테고리별 최대 시도 횟수
    win_conditions = {
        'A': 7,  # 카테고리 A에서 최소 성공 횟수
        'B': 7,  # 카테고리 B에서 최소 성공 횟수
        'C': 4   # 카테고리 C에서 최대 성공 횟수 (초과하면 패배)
    }
    
    # 베타 분포 초기 설정 (성공 확률)
    initial_alpha = 7  
    initial_beta = 3   
    initial_alpha_c = 7
    initial_beta_c = 3

    # 성공 확률의 최소/최대 범위
    success_prob_min = 0.25
    success_prob_max = 0.75
