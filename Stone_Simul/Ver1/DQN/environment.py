# environment.py
import numpy as np
import random
from config import Config

class GameEnv:
    def __init__(self):
        self.config = Config()
        self.reset()

    def reset(self):
        self.choices_left = {'A': self.config.max_attempts, 'B': self.config.max_attempts, 'C': self.config.max_attempts}
        self.successes = {'A': 0, 'B': 0, 'C': 0}
        self.failures = {'A': 0, 'B': 0, 'C': 0}
        self.total_attempts = 0
        self.current_success_rate = self.config.initial_success_rate
        return self._get_state()

    def step(self, action):
        category = ['A', 'B', 'C'][action]
        # print(f"선택된 카테고리 : {category}")

         # 선택 가능 횟수가 없는 카테고리를 선택한 경우
        if self.choices_left[category] == 0:
        # 남은 선택 가능 횟수가 있는 다른 카테고리를 선택하도록 유도
            valid_actions = [i for i in range(3) if self.choices_left[['A', 'B', 'C'][i]] > 0]
            # print(valid_actions)
            choiced_action = random.choice(valid_actions)
            category = ['A', 'B', 'C'][choiced_action]
            # print(f"제대로 되려나 : {category}")
            self.choices_left[category] -= 1
            if not valid_actions:
            # 만약 모든 카테고리의 선택 가능 횟수가 0이라면 게임 종료
                return self._get_state(), -1, True, {}
            # 잘못된 선택에 대한 페널티를 주고 게임 계속 진행
            # return self._get_state(), -1, False, {}
            elif self.successes['C'] >= 5:
                return self._get_state(), -5, True, {}
        else: 
            self.choices_left[category] -= 1
        
        
        self.total_attempts += 1
        # self.choices_left[category] -= 1
        # print(f"총 시도 횟수 {self.total_attempts}")
        # print(f"남은 시도 횟수 {self.choices_left}")
        success = random.random() < self.current_success_rate
        
        if success:
            self.successes[category] += 1
            self.current_success_rate = max(self.config.min_success_rate, self.current_success_rate - self.config.success_rate_change)
            reward = 0
            # if category in ['A', 'B']:
            #     reward = 1  # A와 B의 성공은 +0.1
            # else:
            #     reward = -1  # C의 성공은 보상 없음
        else:
            self.failures[category] += 1
            self.current_success_rate = min(self.config.max_success_rate, self.current_success_rate + self.config.success_rate_change)
            reward = 0

        done = self.total_attempts == self.config.max_choices
        # print(f"대체 어디서 던이 되는거냐고 {done}")
        if done:
            reward += self._get_reward()  # 최종 보상을 추가
            # print(f"일단 한 에피 끝 : {reward}")
        # print(f"성공 : {self.successes}")
        # print(f"실패 : {self.failures}")
        
        return self._get_state(), reward, done, {}

    def _get_state(self):
        return (
            self.choices_left['A'],
            self.choices_left['B'],
            self.choices_left['C'],
            self.successes['A'],
            self.successes['B'],
            self.successes['C'],
            self.failures['A'],
            self.failures['B'],
            self.failures['C'],
            self.current_success_rate,
        )
    
    def _get_reward(self):
        if self.total_attempts < self.config.max_choices:
            return 0
        if self.successes['A'] + self.successes['B'] >= 16 and self.successes['C'] <= 4:
            print("개쩌는돌 우마이!!")
            return 10
        if 14 <= self.successes['A'] + self.successes['B'] <= 15 and self.successes['C'] <= 4:
            print("그래도 선방은 함")
            return 5
        if self.successes['A'] + self.successes['B'] == 13 and self.successes['C'] <= 4:
            print("13....돌....? 어따써.....")
            return 2
        if self.successes['C'] <=4:
            return 1
        # if self.successes['C'] >= 5:
        #     return -5
        return -1
