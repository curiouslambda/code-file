# agent.py
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from scipy.stats import beta  # for Thompson Sampling

class ActorCritic(nn.Module):
    def __init__(self, state_space, action_space):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU(),
            nn.Linear(128, action_space),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value



class PPOAgent:
    def __init__(self, config):
        self.policy = ActorCritic(config.state_space, config.action_space).to(torch.device('cpu'))
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate, betas=config.betas)
        self.config = config
        self.alpha = config.alpha_init
        self.beta = config.beta_init
        self.gamma = config.gamma
        self.ppo_clip = config.ppo_clip
        self.update_timestep = config.update_timestep
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.05
        self.memory = []  # PPO에 필요한 경험을 저장할 메모리

    def select_action(self, state):
        state = torch.FloatTensor(state).to(torch.device('cpu'))
        action_probs, _ = self.policy(state)
        dist = Categorical(action_probs)

        # Thompson Sampling을 적용하여 성공 확률 기반 행동 선택
        sampled_success_prob = beta.rvs(self.alpha, self.beta)  # 베타 분포에서 성공 확률 샘플링
        action = dist.sample().item()
        return action, sampled_success_prob

    def store_transition(self, state, action, log_prob, value, reward, done):
        # 상태, 행동, 로그 확률, 가치 함수 값, 보상, 완료 여부를 메모리에 저장
        self.memory.append((state, action, log_prob, value, reward, done))
    
    def update(self):
        # 메모리에서 데이터 추출

        if len(self.memory) == 0:
            return  # 메모리가 비어있을 경우 업데이트를 하지 않음
    
        states, actions, log_probs, values, rewards, dones = zip(*self.memory)
        
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        log_probs = torch.tensor(log_probs, dtype=torch.float)
        values = torch.tensor(values, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        # Advantage 계산
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float)
        advantages = returns - values.detach()

        for _ in range(4):  # 4번 업데이트 반복
            # 새로운 정책 확률을 계산
            new_action_probs, new_state_values = self.policy(states)
            dist = Categorical(new_action_probs)
            new_log_probs = dist.log_prob(actions)

            # 정책 비율 (new_prob / old_prob)
            ratios = torch.exp(new_log_probs - log_probs)

            # 클리핑 적용
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages

            # 정책 손실
            policy_loss = -torch.min(surr1, surr2).mean()

            # 가치 함수 손실
            value_loss = self.value_loss_coef * nn.MSELoss()(returns, new_state_values)

            # 엔트로피 보너스 (탐색 강화를 위한)
            entropy = dist.entropy().mean()
            entropy_bonus = -self.entropy_coef * entropy

            # 최종 손실 함수
            loss = policy_loss + value_loss + entropy_bonus

            # 역전파 및 업데이트
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 메모리 초기화
        self.memory = []
