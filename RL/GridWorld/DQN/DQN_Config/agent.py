import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from environment import GRID_SIZE, OBSTACLE_STATE, GOAL_STATE
from config import AGENT_PARAMS, SAVED_MODEL_PATH

ACTIONS = {
    'UP': 0,
    'DOWN': 1,
    'LEFT': 2,
    'RIGHT': 3
}

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, AGENT_PARAMS["HIDDEN_LAYER_SIZE"])
        self.fc2 = nn.Linear(AGENT_PARAMS["HIDDEN_LAYER_SIZE"], AGENT_PARAMS["HIDDEN_LAYER_SIZE"])
        self.fc3 = nn.Linear(AGENT_PARAMS["HIDDEN_LAYER_SIZE"], output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = ReplayMemory(AGENT_PARAMS["REPLAY_MEMORY_CAPACITY"])
        self.batch_size = AGENT_PARAMS["BATCH_SIZE"]
        self.gamma = AGENT_PARAMS["GAMMA"]
        self.epsilon = AGENT_PARAMS["EPSILON"]
        self.epsilon_min = AGENT_PARAMS["EPSILON_MIN"]
        self.epsilon_decay = AGENT_PARAMS["EPSILON_DECAY"]
        self.model = DQN(state_size, action_size).to(device)  # 모델을 초기화할 때 GPU로 이동합니다.
        self.optimizer = optim.Adam(self.model.parameters(), lr=AGENT_PARAMS["LEARNING_RATE"])
    
    def act(self, state):
        state = state.to(self.device)  # 상태 데이터를 지정된 장치로 이동합니다.
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                return self.model(state).max(1)[1].view(1, 1).item()
    
    def remember(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(self.device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        state_action_values = self.model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def take_action(state, action):
    next_state = None
    if action == ACTIONS['UP']:
        next_state = max(state[0] - 1, 0), state[1]
    elif action == ACTIONS['DOWN']:
        next_state = min(state[0] + 1, GRID_SIZE - 1), state[1]
    elif action == ACTIONS['LEFT']:
        next_state = state[0], max(state[1] - 1, 0)
    elif action == ACTIONS['RIGHT']:
        next_state = state[0], min(state[1] + 1, GRID_SIZE - 1)

    if next_state == OBSTACLE_STATE:
        return state

    return next_state

def dqn_learning(device):
    start_time = time.time()
    state_size = 2
    action_size = len(ACTIONS)
    agent = DQNAgent(state_size, action_size, device)
    EPISODES = AGENT_PARAMS["EPISODE"]
    steps_to_goal = []

    for e in range(EPISODES):
        state = torch.tensor(np.random.randint(0, GRID_SIZE, size=(1, state_size)), dtype=torch.float).to(device)  # 초기 상태 데이터를 지정된 장치로 이동합니다.
        done = False
        steps = 0
        while not done:
            action = agent.act(state)
            next_state = take_action(tuple(state.tolist()[0]), action)
            reward = 1 if next_state == GOAL_STATE else 0
            if next_state == OBSTACLE_STATE:
                reward = -1
                done = True
            next_state = torch.tensor(next_state, dtype=torch.float).view(1, -1) if not done else None
            reward = torch.tensor([reward], dtype=torch.float)
            next_state = next_state.to(device) if next_state is not None else None  # 다음 상태 데이터를 지정된 장치로 이동합니다.
            reward = reward.to(device)  # 보상 데이터를 지정된 장치로 이동합니다.
            agent.remember(state, torch.tensor([[action]], dtype=torch.long).to(device), next_state, reward)  # 메모리에 저장할 때에도 장치를 명시적으로 지정합니다.
            state = next_state
            steps += 1
            if state is not None and state.tolist()[0] == list(GOAL_STATE):
                done = True
                steps_to_goal.append(steps)
            agent.replay()
            agent.update_epsilon()
        if e % 5000 == 0:
            print(f"episode: {e}/{EPISODES}, steps: {steps}, epsilon: {agent.epsilon}")
    total_time = time.time() - start_time
    print(f'Training Time: {total_time:.2f} seconds')

    # Save the trained model
    torch.save(agent.model.state_dict(), SAVED_MODEL_PATH)

    return agent.model, steps_to_goal
