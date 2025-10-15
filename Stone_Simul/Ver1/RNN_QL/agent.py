import numpy as np

class QLearningAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.q_table = {}  # Q-테이블 초기화

    def choose_action(self, state, epsilon):
        # 상태를 문자열로 변환하여 Q-테이블의 키로 사용
        state_str = str(state)

        # epsilon-greedy 전략 사용
        if np.random.rand() < epsilon:
            # 탐험: 무작위로 액션 선택
            action = np.random.randint(self.num_actions)
            # print(f"Exploration: Random action {action} selected")
        else:
            # 활용: 현재 Q-테이블에서 가장 높은 가치를 가지는 액션 선택
            if state_str not in self.q_table:
                # 만약 현재 상태가 Q-테이블에 없다면, 모든 액션에 대해 초기값 0 설정
                self.q_table[state_str] = np.zeros(self.num_actions)
            action = np.argmax(self.q_table[state_str])
            # print(f"Exploitation: Best action {action} selected based on Q-table")
        
        return action

    def update_q_table(self, state, action, reward, next_state, alpha, gamma):
        state_str = str(state)
        next_state_str = str(next_state)

        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(self.num_actions)
        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = np.zeros(self.num_actions)

        # Q-학습의 갱신 식
        best_next_action = np.argmax(self.q_table[next_state_str])
        td_target = reward + gamma * self.q_table[next_state_str][best_next_action]
        td_error = td_target - self.q_table[state_str][action]
        self.q_table[state_str][action] += alpha * td_error
