# environment.py

import numpy as np

class Environment:
    def __init__(self):
        self.categories = ['A', 'B', 'C']
        self.success_thresholds = {'A': 16, 'B': 16, 'C': 4}
        self.max_attempts = 30
        self.success_prob = 0.75
        self.min_prob = 0.25
        self.max_prob = 0.75
        self.attempts = {category: 10 for category in self.categories}
        self.success_counts = {category: 0 for category in self.categories}
        self.current_category = None
        self.game_over = False

    def reset(self):
        self.attempts = {category: 10 for category in self.categories}
        self.success_counts = {category: 0 for category in self.categories}
        self.current_category = None
        self.game_over = False

    def select_category(self, category):
        if category not in self.categories:
            raise ValueError("Invalid category")
        
        # 남은 시도 횟수가 있는 카테고리만 선택 가능
        if self.attempts[category] > 0:
            self.current_category = category
        else:
            available_categories = [cat for cat in self.categories if self.attempts[cat] > 0]
            if not available_categories:
                raise ValueError("No categories with remaining attempts")
            self.current_category = np.random.choice(available_categories)

    def play(self):
        if self.current_category is None:
            raise ValueError("No category selected")
        success = np.random.choice([True, False], p=[self.success_prob, 1 - self.success_prob])
        if success:
            self.success_counts[self.current_category] += 1
            # 성공 확률을 조정하는 방식 변경
            self.success_prob = max(self.success_prob - 0.05, self.min_prob)
        else:
            # 성공 확률을 조정하는 방식 변경
            self.success_prob = min(self.success_prob + 0.05, self.max_prob)
        self.attempts[self.current_category] -= 1
        if sum(self.attempts.values()) <= 0:
            self.game_over = True
            win_condition = all(self.success_counts[category] >= self.success_thresholds[category] for category in self.categories if category != 'C') and self.success_counts['C'] <= self.success_thresholds['C']
            return win_condition