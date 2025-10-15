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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
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


# import random
# import numpy as np
# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# from collections import deque, namedtuple
# from models import DQN
# from config import Config

# class PrioritizedReplayBuffer:
#     def __init__(self, capacity, alpha=0.6):
#         self.capacity = capacity
#         self.alpha = alpha
#         self.buffer = deque(maxlen=capacity)
#         self.priorities = deque(maxlen=capacity)
#         self.pos = 0
#         self.experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

#     def __len__(self):
#         return len(self.buffer)
    
#     def append(self, state, action, reward, next_state, done):
#         max_priority = max(self.priorities, default=1.0)
#         self.buffer.append(self.experience(state, action, reward, next_state, done))
#         self.priorities.append(max_priority)

#     def sample(self, batch_size, beta=0.4):
#         if len(self.buffer) == len(self.priorities):
#             priorities = np.array(self.priorities, dtype=np.float32)
#             priorities = priorities ** self.alpha
#             probabilities = priorities / priorities.sum()

#             indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
#             experiences = [self.buffer[idx] for idx in indices]

#             total = len(self.buffer)
#             weights = (total * probabilities[indices]) ** (-beta)
#             weights /= weights.max()
#             weights = np.array(weights, dtype=np.float32)

#             batch = [self.buffer[idx] for idx in indices]  # 수정된 부분

#             return batch, indices, weights
#         else:
#             return [], [], []

#     def update_priorities(self, batch_indices, batch_priorities):
#         for idx, priority in zip(batch_indices, batch_priorities):
#             self.priorities[idx] = priority

# class DQNAgent:
#     def __init__(self, state_dim, action_dim):
#         self.config = Config()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.memory = PrioritizedReplayBuffer(self.config.memory_size)
#         self.gamma = self.config.gamma
#         self.epsilon = self.config.epsilon_start
#         self.epsilon_min = self.config.epsilon_end
#         self.epsilon_decay = self.config.epsilon_decay
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = DQN(state_dim, action_dim).to(self.device)
#         self.target_model = DQN(state_dim, action_dim).to(self.device)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
#         self.update_target_model()

#     def update_target_model(self):
#         self.target_model.load_state_dict(self.model.state_dict())

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append(state, action, reward, next_state, done)

#     def act(self, state):
#         self.model.eval()  # 평가 모드 설정
#         with torch.no_grad():  # 평가 모드에서 gradient 계산 비활성화
#             state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#             act_values = self.model(state)
#         self.model.train()  # 다시 학습 모드로 전환
#         if random.random() <= self.epsilon:
#             return random.randrange(self.action_dim)
#         return torch.argmax(act_values[0]).item()

#     def replay(self):
#         if len(self.memory.buffer) < self.config.batch_size:  # 수정된 부분
#             return

#         batch = random.sample(self.memory, self.config.batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)

#         states = torch.FloatTensor(states).to(self.device)
#         actions = torch.LongTensor(actions).to(self.device)
#         rewards = torch.FloatTensor(rewards).to(self.device)
#         next_states = torch.FloatTensor(next_states).to(self.device)
#         dones = torch.FloatTensor(dones).to(self.device)

#         q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
#         next_q_values = self.target_model(next_states).max(1)[0]
#         expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

#         loss = F.mse_loss(q_values, expected_q_values.detach())

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
