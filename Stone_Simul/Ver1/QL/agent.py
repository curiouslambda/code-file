# agent.py
import numpy as np
import random
from config import Config

class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.alpha = Config.ALPHA
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON
        self.q_table = {}

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]

    def choose_action(self, state):
        available_actions = [action for action, choice in zip(self.actions, state[0]) if choice > 0]
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        else:
            q_values = [self.get_q_values(state)[self.actions.index(action)] for action in available_actions]
            max_q_value = max(q_values)
            max_actions = [action for action, q_value in zip(available_actions, q_values) if q_value == max_q_value]
            return random.choice(max_actions)

    def update(self, state, action, reward, next_state):
        action_index = self.actions.index(action)
        q_next = max(self.get_q_values(next_state))
        self.get_q_values(state)[action_index] += self.alpha * (reward + self.gamma * q_next - self.get_q_values(state)[action_index])
