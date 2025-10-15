# environment.py
import numpy as np
import random
import torch
from config import Config
from agent import DQNAgent

class GameEnv:
    def __init__(self, agent, state_dim, action_dim):
        self.config = Config()
        self.agent = agent
        self.reset()

    def reset(self):
        self.choices_left = {'A': self.config.max_attempts, 'B': self.config.max_attempts, 'C': self.config.max_attempts}
        self.successes = {'A': 0, 'B': 0, 'C': 0}
        self.failures = {'A': 0, 'B': 0, 'C': 0}
        self.total_attempts = 0
        self.current_success_rate = self.config.initial_success_rate
        return self._get_state()

    def step(self, action):
        reward = 0
        category = ['A', 'B', 'C'][action]

        # 선택 가능 횟수가 없는 카테고리를 선택한 경우
        if self.choices_left[category] == 0:
        # 남은 선택 가능 횟수가 있는 다른 카테고리를 선택하도록 유도
            valid_actions = [i for i in range(3) if self.choices_left[['A', 'B', 'C'][i]] > 0]
            # choiced_action = random.choice(valid_actions)
            # category = ['A', 'B', 'C'][choiced_action]
            # self.choices_left[category] -= 1

            if valid_actions:
                # 유효한 액션들 중에서 Q값이 더 높은 액션을 선택
                state = self._get_state()
                valid_q_values = [self.agent.model(torch.FloatTensor(state).unsqueeze(0).to(self.agent.device))[0][i].item() for i in valid_actions]
                best_action = valid_actions[np.argmax(valid_q_values)]
                category = ['A', 'B', 'C'][best_action]
                self.choices_left[category] -= 1


            if not valid_actions:
            # 만약 모든 카테고리의 선택 가능 횟수가 0이라면 게임 종료
                return self._get_state(), 0, True, {}
            # 잘못된 선택에 대한 페널티를 주고 게임 계속 진행
            # return self._get_state(), -1, False, {}
        else: 
            self.choices_left[category] -= 1
        
        
        self.total_attempts += 1
        success = random.random() < self.current_success_rate

        if category == 'A':
            if success:
                self.successes['A'] += 1
                reward = 5  # A 성공에 대한 보상
            else:
                self.failures['A'] += 1
        elif category == 'B':
            if success:
                self.successes['B'] += 1
                reward = 5  # B 성공에 대한 보상
            else:
                self.failures['B'] += 1
        elif category == 'C':
            if success:
                self.successes['C'] += 1
                reward = 2  # C 성공에 대한 작은 보상
            else:
                self.failures['C'] += 1
                reward = 3  # C 실패에 대한 보상으로 선택 유도
        
        if success:
            # self.successes[category] += 1
            self.current_success_rate = max(self.config.min_success_rate, self.current_success_rate - self.config.success_rate_change)
            # reward = 0

        else:
            # self.failures[category] += 1
            self.current_success_rate = min(self.config.max_success_rate, self.current_success_rate + self.config.success_rate_change)
            # reward = 0

        if self.successes['C'] >= 5:
            return self._get_state(), -100, True, {}
        
        done = self.total_attempts == self.config.max_choices
        
        if done:
            reward += self._get_reward()  # 최종 보상을 추가

        
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
        # if self.total_attempts < self.config.max_choices:
        #     return 0
        # if self.successes['A'] + self.successes['B'] >= 16 and self.successes['C'] <= 4:
        #     print("개쩌는돌 우마이!!")
        #     return 10
        # if 14 <= self.successes['A'] + self.successes['B'] <= 15 and self.successes['C'] <= 4:
        #     print("그래도 선방은 함")
        #     return 5
        # if self.successes['A'] + self.successes['B'] == 13 and self.successes['C'] <= 4:
        #     print("13....돌....? 어따써.....")
        #     return 2
        # if self.successes['C'] <=4:
        #     return 1
        # # if self.successes['C'] >= 5:
        # #     return -5
        # return -1

        # if self.total_attempts < self.config.max_choices:
        #     return 0
        # # if self.successes['A'] >=7 and self.successes['B'] <7 and self.successes['C'] <= 4:
        # #     print("A는 7넘음. B는 쓰읍...")
        # #     return 2
        # # if self.successes['A'] <7 and self.successes['B'] >=7 and self.successes['C'] <= 4:
        # #     print("--A는 쓰읍... B는 7넘음")
        # #     return 2
        # if 7<= self.successes['A'] <9 and 7<= self.successes['B'] <9 and self.successes['C'] <= 4:
        #     print("=====7 7 돌인데!! !!!!=====")
        #     return 7
        # if 9<= self.successes['A'] and 7<= self.successes['B'] <9 and self.successes['C'] <= 4:
        #     print("========= 9 7 돌 !!!!! =========")
        #     return 9
        # if 7<= self.successes['A'] <9 and 9<= self.successes['B'] and self.successes['C'] <= 4:
        #     print("========= 9 7 돌 !!!!! =========")
        #     return 9
        # if self.successes['C'] <=4:
        #     return 1
        # # if self.successes['C'] >= 5:
        # #     return -5
        # return -5

        if self.total_attempts < self.config.max_choices:
            return 0
    
        reward = 0

        # # 중간 목표 보상
        # if 7 <= self.successes['A'] < 9:
        #     reward += 2
        # if 7 <= self.successes['B'] < 9:
        #     reward += 2

        # # 최종 목표 보상
        # if 9 <= self.successes['A']:
        #     reward += 4
        # if 9 <= self.successes['B']:
        #     reward += 4

        # 동시 만족 보상
        if 7 <= self.successes['A'] < 9 and 7 <= self.successes['B'] < 9:
            reward += 30

        # 동시 만족 보상
        if 9 <= self.successes['A'] and 7 <= self.successes['B'] < 9:
            reward += 50
        if 7 <= self.successes['A'] < 9 and 9 <= self.successes['B']:
            reward += 50
        if 9 <= self.successes['A'] and 9 <= self.successes['B']:
            reward += 100

        # C 성공 횟수 제한 보상
        if self.successes['C'] <= 4:
            reward += 10
        else:
            reward -= 50

        # # 성공/실패에 따른 최종 보상
        # if self.successes['A'] + self.successes['B'] >= 16 and self.successes['C'] <= 4:
        #     print("개쩌는돌 우마이!!")
        #     reward += 10
        # elif 14 <= self.successes['A'] + self.successes['B'] <= 15 and self.successes['C'] <= 4:
        #     print("그래도 선방은 함")
        #     reward += 5
        # else:
        #     reward -= 1
        
        return reward