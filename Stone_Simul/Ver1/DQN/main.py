# main.py
from train import train_dqn

if __name__ == "__main__":
    episodes = 100
    scores = train_dqn(episodes)
