# main.py
from train import train_dqn
from config import Config

if __name__ == "__main__":
    episodes = Config.total_episodes
    scores = train_dqn(episodes)
