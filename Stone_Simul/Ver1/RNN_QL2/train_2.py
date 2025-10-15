# train.py

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from models import RNNModel
from environment_2 import Environment
from agent import QAgent
from config import Config

def train_rnn(num_episodes):
    # 모델 초기화
    input_size = Config.INPUT_SIZE
    hidden_size = Config.HIDDEN_SIZE
    output_size = Config.OUTPUT_SIZE
    learning_rate = Config.LEARNING_RATE
    num_epochs = Config.NUM_EPOCHS

    rnn_model = RNNModel(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)

    env = Environment()
    for episode in range(num_episodes):
        state = [
            env.attempts['A'],
            env.attempts['B'],
            env.attempts['C'],
            env.success_counts['A'],
            env.success_counts['B'],
            env.success_counts['C'],
            env.success_prob
        ]
        env.reset()
        states = []
        actions = []
        done = False
        while not done:
            states.append(state)
            action = rnn_model(torch.Tensor(state).unsqueeze(0).unsqueeze(0))
            action = action.data.max(1)[1].item()
            actions.append(action)
            env.select_category(env.categories[action])
            win = env.play()
            next_state = [
                env.attempts['A'],
                env.attempts['B'],
                env.attempts['C'],
                env.success_counts['A'],
                env.success_counts['B'],
                env.success_counts['C'],
                env.success_prob
            ]
            state = next_state
            done = env.game_over

        # 훈련
        optimizer.zero_grad()
        states = torch.Tensor(states).unsqueeze(1)  # 여기에 unsqueeze(1) 추가
        actions = torch.Tensor(actions).long().unsqueeze(1)
        output = rnn_model(states)
        loss = criterion(output, actions)
        loss.backward()
        optimizer.step()
        print(f'Episode {episode+1}, Loss: {loss.item()}')

def train_q_learning(num_episodes):
    env = Environment()
    agent = QAgent(num_actions=3)

    for episode in range(num_episodes):
        state = [
            env.attempts['A'],
            env.attempts['B'],
            env.attempts['C'],
            env.success_counts['A'],
            env.success_counts['B'],
            env.success_counts['C'],
            env.success_prob
        ]
        env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            env.select_category(env.categories[action])
            win = env.play()
            reward = 1 if win else -1
            next_state = [
                env.attempts['A'],
                env.attempts['B'],
                env.attempts['C'],
                env.success_counts['A'],
                env.success_counts['B'],
                env.success_counts['C'],
                env.success_prob
            ]
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            done = env.game_over

if __name__ == "__main__":
    print("Training RNN model...")
    train_rnn(num_episodes=Config.NUM_EPISODES)
    print("Training Q-Learning agent...")
    train_q_learning(num_episodes=Config.NUM_EPISODES)