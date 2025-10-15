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
        
        # 선택한 카테고리의 남은 시도 횟수가 0 이하일 때까지 반복하여 다른 카테고리 선택
        while self.attempts[category] <= 0:
            category = self.categories[np.random.randint(len(self.categories))]
    
        self.current_category = category

    def play(self):
        if self.current_category is None:
            raise ValueError("No category selected")
        success = np.random.choice([True, False], p=[self.success_prob, 1 - self.success_prob])
        if success:
            self.success_counts[self.current_category] += 1
            self.success_prob = max(self.success_prob - 0.1, self.min_prob)
        else:
            self.success_prob = min(self.success_prob + 0.1, self.max_prob)
        self.attempts[self.current_category] -= 1
        if sum(self.attempts.values()) <= 0:
            self.game_over = True
            win_condition = all(self.success_counts[category] >= self.success_thresholds[category] for category in self.categories if category != 'C') and self.success_counts['C'] <= self.success_thresholds['C']
            return win_condition

