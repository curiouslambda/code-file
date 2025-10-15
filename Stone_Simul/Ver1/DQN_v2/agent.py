# agent.py
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from models import DQN
from config import Config

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.config = Config()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=self.config.memory_size)
        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon_start
        self.epsilon_min = self.config.epsilon_end
        self.epsilon_decay = self.config.epsilon_decay
        self.initial_steps = self.config.initial_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.config.memory_size:
            del self.memory[0]

    # def act(self, state):
    #     if random.random() <= self.epsilon:
    #         return random.randrange(self.action_dim)
    #     state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    #     act_values = self.model(state)
    #     return torch.argmax(act_values[0]).item()

    def act(self, state):
        current_success_rate = state[-1]  # 현재 성공 확률
        # if self.initial_steps > 0:
        #     # print("몇번이나 나오냐 --")
        #     # 초기 학습 단계
        #     self.initial_steps -= 1
        #     # print(f"확률 : {current_success_rate}")
        #     if current_success_rate >= 0.60:
        #         # print("0또는 1 선택해야됨")
        #         return random.choice([0, 1])  # A, B 중 하나 선택
        #     elif current_success_rate <= 0.40:
        #         # print("-----2선택해야됨")
        #         return 2  # C 선택
        if random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self):
        if len(self.memory) < self.config.batch_size:
            return

        batch = random.sample(self.memory, self.config.batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            target = reward
            if not done:
                target = (reward + self.gamma * torch.max(self.target_model(next_state)[0]).item())
            target_f = self.model(state)
            # print(f"현재 상태에 대한 모델의 예측 값 : {target_f}")
            target_f[0][action] = target
            # print(f"수행한 행동에 대한 Q값 업데이트 {target_f[0]}")

            self.optimizer.zero_grad()
            loss = F.mse_loss(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay