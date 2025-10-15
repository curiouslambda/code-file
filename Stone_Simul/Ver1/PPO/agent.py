# agent.py
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from models import PolicyNetwork, ValueNetwork
from config import Config

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.config = Config()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=self.config.memory_size)
        self.gamma = self.config.gamma
        self.lam = self.config.lam
        self.epsilon = self.config.epsilon_start
        self.epsilon_decay = self.config.epsilon_decay
        self.epsilon_end = self.config.epsilon_end
        self.device = self.config.device
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.config.device)
        self.value_net = ValueNetwork(state_dim).to(self.config.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.config.learning_rate)
        self.clip_param = self.config.clip_param
        self.ppo_epochs = self.config.ppo_epochs
        self.ppo_batch_size = self.config.ppo_batch_size

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # def act(self, state):
    #     state = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
    #     with torch.no_grad():
    #         probs = self.policy_net(state)
    #         print(f"확률? {probs}")
    #     action = torch.multinomial(probs, num_samples=1).item()
    #     print(f"행동 :{action}")
    #     return action

    def act(self, state):
        # Epsilon 값을 결정합니다.
        epsilon = self.epsilon
        
        # 랜덤한 값이 epsilon 보다 작으면 랜덤한 행동을 선택합니다.
        if random.random() < epsilon:
            action = random.randint(0, self.action_dim - 1)
            # print(f"엡실론으로 나온 액션 {action}")
        else:
            # 그 외의 경우에는 확률 분포를 통해 행동을 선택합니다.
            state = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
            with torch.no_grad():
                probs = self.policy_net(state)
                # print(f'확률? {probs}')
            action = torch.multinomial(probs, num_samples=1).item()
            # print(f"액션 {action}")
        
        return action
    
    def update_epsilon(self, step):
        # 매 스텝마다 Epsilon 값을 감소시킵니다.
        self.epsilon *= self.config.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_end)


    def learn(self):
        states, actions, rewards, next_states, dones = zip(*self.memory)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(len(states))
            for i in range(0, len(states), self.ppo_batch_size):
                idx = indices[i:i + self.ppo_batch_size]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_rewards = rewards[idx]
                batch_next_states = next_states[idx]
                batch_dones = dones[idx]

                advantages = self._calculate_advantages(batch_rewards, batch_next_states, batch_dones)
                old_log_probs = self._get_log_probs(batch_states, batch_actions).detach()
                log_probs = self._get_log_probs(batch_states, batch_actions)

                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()

                values = self.value_net(batch_states).squeeze()
                value_loss = F.mse_loss(values, batch_rewards)

                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.max_grad_norm)
                self.value_optimizer.step()

        self.memory.clear()

    def _calculate_advantages(self, rewards, next_states, dones):
        values = self.value_net(next_states).detach().squeeze()
        returns = rewards + self.gamma * values * (1 - dones)
        advantages = returns - values
        return advantages

    def _get_log_probs(self, states, actions):
        probs = self.policy_net(states)
        log_probs = torch.log(probs)
        return log_probs.gather(1, actions.unsqueeze(1)).squeeze()