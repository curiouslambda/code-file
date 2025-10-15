import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from environment import GRID_SIZE, OBSTACLE_STATE, GOAL_STATE

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
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_size)

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
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(2000)
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 0.05
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def act(self, state):
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

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size)
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

def dqn_learning():
    start_time = time.time()
    state_size = 2
    action_size = len(ACTIONS)
    agent = DQNAgent(state_size, action_size)
    EPISODES = 50000
    steps_to_goal = []

    for e in range(EPISODES):
        state = torch.tensor(np.random.randint(0, GRID_SIZE, size=(1, state_size)), dtype=torch.float)
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
            agent.remember(state, torch.tensor([[action]], dtype=torch.long), next_state, reward)
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

    return agent.model, steps_to_goal

if __name__ == "__main__":
    model, steps_to_goal = dqn_learning()
