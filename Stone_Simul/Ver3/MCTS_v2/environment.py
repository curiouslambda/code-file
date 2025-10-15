### environment.py

import random
from config import *

class Environment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.success_prob = INITIAL_SUCCESS_PROB
        self.attempts_left = {category: MAX_ATTEMPTS_PER_CATEGORY for category in CATEGORIES}
        self.successes = {category: 0 for category in CATEGORIES}
        self.failures = {category: 0 for category in CATEGORIES}
        return self.get_state()

    def get_state(self):
        return (self.success_prob, self.successes.copy(), self.failures.copy(), self.attempts_left.copy())

    def step(self, category):
        if self.attempts_left[category] <= 0:
            return None, 0, True  # Invalid move

        success = random.random() < self.success_prob
        self.attempts_left[category] -= 1

        if success:
            self.successes[category] += 1
            self.success_prob = max(MIN_SUCCESS_PROB, self.success_prob - SUCCESS_PROB_CHANGE)
        else:
            self.failures[category] += 1
            self.success_prob = min(MAX_SUCCESS_PROB, self.success_prob + SUCCESS_PROB_CHANGE)

        done = self.check_termination()
        reward = self.calculate_reward(done)

        return self.get_state(), reward, done

    def check_termination(self):
        if self.successes['C'] >= FAIL_CONDITION_C:
            return True
        if self.failures['A'] >= FAIL_CONDITION_A_B or self.failures['B'] >= FAIL_CONDITION_A_B:
            return True
        if all(attempts == 0 for attempts in self.attempts_left.values()):
            return True
        return False

    def calculate_reward(self, done):
        if not done:
            return 0
        if (self.successes['A'] >= WIN_CONDITION_A_B and 
            self.successes['B'] >= WIN_CONDITION_A_B and 
            self.successes['C'] <= WIN_CONDITION_C):
            return 1
        return -1