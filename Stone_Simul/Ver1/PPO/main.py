# main.py
from train import train_ppo

if __name__ == "__main__":
    episodes = 100
    scores = train_ppo(episodes)
