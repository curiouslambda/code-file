# agent.py

import numpy as np

class QAgent:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # 상태 공간의 크기를 정의
        self.state_space_size = 11 * 11 * 11  # 예시로 상태 공간의 크기를 1331로 설정
        
        # Q 테이블을 초기화할 때 상태 공간의 크기를 고려하여 크기를 설정
        self.q_table = np.zeros((self.state_space_size, num_actions))

    def convert_state_to_index(self, state):
        # 각 카테고리의 최대 성공 횟수
        max_success_count = 10
        
        # 상태에서 각 카테고리의 성공 횟수 추출
        success_counts = state
        
        # 각 카테고리의 성공 횟수를 인덱스로 변환하여 반환
        index = success_counts[0] + max_success_count * success_counts[1] + max_success_count * max_success_count * success_counts[2]
        return index

    def select_action(self, state):
        state_index = self.convert_state_to_index(state)  # 상태를 인덱스로 변환
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[state_index, :])

    def update_q_table(self, state, action, reward, next_state):
        state_index = self.convert_state_to_index(state)  # 상태를 인덱스로 변환
        next_state_index = self.convert_state_to_index(next_state)  # 다음 상태를 인덱스로 변환
        best_next_action = np.argmax(self.q_table[next_state_index, :])
        td_target = reward + self.discount_factor * self.q_table[next_state_index, best_next_action]
        td_error = td_target - self.q_table[state_index, action]
        self.q_table[state_index, action] += self.learning_rate * td_error
