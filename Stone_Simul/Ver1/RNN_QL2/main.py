# main.py

from train_2 import train_rnn, train_q_learning
from config import Config

if __name__ == "__main__":
    print("Training RNN model...")
    train_rnn(num_episodes=Config.NUM_EPISODES)
    print("Training Q-Learning agent...")
    train_q_learning(num_episodes=Config.NUM_EPISODES)