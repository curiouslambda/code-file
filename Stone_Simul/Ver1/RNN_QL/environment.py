import numpy as np
from config import Config

class GameEnvironment:
    def __init__(self):
        self.success_prob = Config.success_prob
        self.min_prob = Config.min_prob
        self.max_prob = Config.max_prob
        self.num_categories = Config.num_categories
        self.num_trials = Config.num_trials
        self.trial_count = 0
        self.attempt_A = 0
        self.attempt_B = 0
        self.attempt_C = 0
        self.success_count_A = 0
        self.success_count_B = 0
        self.success_count_C = 0
        self.current_category = 0

    def reset(self):
        self.success_prob = Config.success_prob
        self.trial_count = 0
        self.attempt_A = 0
        self.attempt_B = 0
        self.attempt_C = 0
        self.success_count_A = 0
        self.success_count_B = 0
        self.success_count_C = 0
        self.current_category = 0

    def get_state(self):
        return np.array([
            self.current_category,
            self.success_count_A,
            self.success_count_B,
            self.success_count_C,
            self.success_prob
        ])

    def step(self, action):
        reward = 0
        self.current_category = action  # 현재 선택된 카테고리 업데이트

        if action == 0:
            if self.attempt_A <= 10:
                self.attempt_A += 1
                if np.random.random() < self.success_prob:
                    self.success_count_A += 1
                    reward = 1
                    self.success_prob = max(self.min_prob, self.success_prob - 0.1)
                else:
                    reward = -1
                    self.success_prob = min(self.max_prob, self.success_prob + 0.1)
            else:
                action = np.random.randint(1,3)
        elif action == 1:
            if np.random.random() < self.success_prob:
                self.success_count_B += 1
                reward = 1
                self.success_prob = max(self.min_prob, self.success_prob - 0.1)
            else:
                reward = -1
                self.success_prob = min(self.max_prob, self.success_prob + 0.1)
            
        elif action == 2:
            if np.random.random() < self.success_prob:
                self.success_count_C += 1
                reward = -1
                self.success_prob = max(self.min_prob, self.success_prob - 0.1)
            else:
                reward = 1
                self.success_prob = min(self.max_prob, self.success_prob + 0.1)
            
        self.trial_count += 1

        return reward, self.trial_count

    def is_episode_done(self):
        success_sum = self.success_count_A + self.success_count_B
        if self.trial_count == self.num_trials:
            return True
        elif (success_sum >= 14 and success_sum <= 15 and self.success_count_C <= 4):
            return False, 5  # 적당한 보상 반환
        elif (success_sum >= 16 and self.success_count_C <= 4):
            return True, 10  # 큰 보상 반환
        else:
            return False, -5  # 보상 없음