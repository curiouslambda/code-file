# train.py

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from models import RNNModel
from environment import Environment
from agent import QAgent
from config import Config

def generate_training_data(num_samples=1000):
    X = []
    y = []
    for _ in range(num_samples):
        env = Environment()
        states = []
        actions = []
        for _ in range(30):  # 30번의 선택
            state = [
                env.attempts['A'],
                env.attempts['B'],
                env.attempts['C'],
                env.success_counts['A'],
                env.success_counts['B'],
                env.success_counts['C'],
                env.success_prob
            ]
            states.append(state)
            valid_actions = [i for i, category in enumerate(env.categories) if env.attempts[category] > 0]
            action = np.random.choice(valid_actions)
            actions.append(action)
            env.select_category(env.categories[action])
            env.play()
        X.append(states)
        y.append(actions)
    return np.array(X), np.array(y)

def train_rnn():
    # 모델 초기화
    input_size = Config.INPUT_SIZE
    hidden_size = Config.HIDDEN_SIZE
    output_size = Config.OUTPUT_SIZE
    learning_rate = Config.LEARNING_RATE
    num_epochs = Config.NUM_EPOCHS

    rnn_model = RNNModel(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)

    # 데이터 생성 및 전처리
    X_train, y_train = generate_training_data()
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train).long()

    # 훈련
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = rnn_model(X_train)
        # 출력과 대상 데이터의 크기 확인
        # print("Output shape:", output.shape)
        # print("Target shape:", y_train.shape)
        output = output.view(-1, output_size)  # (batch_size * sequence_length, output_size)
        y_train_flat = y_train.view(-1)  # (batch_size * sequence_length)
        # print("Step:", epoch+1)
        # print("Output shape:", output.shape)
        # print("Target shape:", y_train_flat.shape)
        loss = criterion(output, y_train_flat)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def train_q_learning():
    num_episodes = Config.NUM_EPISODES
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
            print(f"처음 상태 : {state}, \n다음 상태 : {next_state}")
            state = next_state
            done = env.game_over

# if __name__ == "__main__":
#     print("Training RNN model...")
#     train_rnn()
#     print("Training Q-Learning agent...")
#     train_q_learning()
