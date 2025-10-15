# utils.py
import matplotlib.pyplot as plt
import json
from config import Config

def plot_training_statistics(episodes, rewards, stone_14_counts, stone_16_counts):
    plt.figure(figsize=(14, 7))

    plt.subplot(3, 1, 1)
    plt.plot(episodes, rewards, label='Total Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(episodes, stone_14_counts, label='Stone 14 Count')
    plt.xlabel('Episodes')
    plt.ylabel('Stone 14 Count')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(episodes, stone_16_counts, label='Stone 16 Count')
    plt.xlabel('Episodes')
    plt.ylabel('Stone 16 Count')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(f'./plots/{Config.EPISODES}_{Config.ALPHA}_{Config.PERFORMANCE_CEHCK_INTERVAL}.png')

def save_training_statistics(data, filename):
    """
    훈련 데이터를 JSON 파일로 저장하는 함수
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_training_statistics(filename):
    """
    JSON 파일에서 훈련 데이터를 불러오는 함수
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return data
