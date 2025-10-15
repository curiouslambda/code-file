# environment.py

import random
from config import Config

class GameEnvironment:
    def __init__(self, alpha, beta, alpha_c, beta_c):
        self.success_prob = Config.success_prob_max  # 초기 성공 확률
        self.attempts_left = {'A': Config.max_attempts_per_category, 
                              'B': Config.max_attempts_per_category, 
                              'C': Config.max_attempts_per_category}
        self.successes = {'A': 0, 'B': 0, 'C': 0}
        self.failures = {'A': 0, 'B': 0, 'C': 0}
        self.alpha = alpha  # A 카테고리 알파
        self.beta = beta    # A 카테고리 베타
        self.alpha_c = alpha_c   # C 카테고리 알파
        self.beta_c = beta_c     # C 카테고리 베타

    def get_success_prob(self):
        return self.success_prob

    def attempt(self, category):
        """카테고리에서 시도하고 성공/실패를 반환"""
        if self.attempts_left[category] <= 0:
            raise ValueError(f"카테고리 {category}에서 더 이상 시도할 수 없습니다.")

        # 성공 여부는 현재 성공 확률에 따라 결정
        success = random.random() < self.success_prob

        if success:
            self.successes[category] += 1
            if category == 'C':
                # C에서는 알파를 증가시키지 않음, 성공을 덜 유도
                self.beta_c += 1
                self.success_prob = max(self.success_prob - 0.1, Config.success_prob_min)  # 성공 확률 감소
            else:
                self.alpha += 1  # A, B에서 성공하면 알파 증가
                self.success_prob = max(self.success_prob - 0.1, Config.success_prob_min)  # 성공 확률 감소
        else:  
            self.failures[category] += 1
            self.success_prob = min(self.success_prob + 0.1, Config.success_prob_max)  # 실패시 확률 증가
            if category == 'C':
                self.alpha_c += 1
            else:
                self.beta += 1
    
        self.attempts_left[category] -= 1

        return success

    def is_game_over(self):
        """30번의 시도 후 게임 종료 여부 판단"""
        total_attempts = sum(self.attempts_left.values())
        return total_attempts == 0

    def check_win(self):
        """게임에서 이겼는지 여부 확인"""
        return (self.successes['A'] >= Config.win_conditions['A'] and
                self.successes['B'] >= Config.win_conditions['B'] and
                self.successes['C'] <= Config.win_conditions['C'])
    

    # def failure_count_c(self):
    #     failure_count = 0

    #     if self.attempts_left['A'] == 0 and self.attempts_left['B'] == 0 and self.attempts_left['C'] == 0:
    #         if self.failures['C'] >= 5:
    #             failure_count += 1
        
    #     print(failure_count)
    #     return failure_count