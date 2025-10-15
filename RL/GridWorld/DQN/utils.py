import matplotlib.pyplot as plt
import numpy as np
import os
from environment import GRID_SIZE, GOAL_STATE, OBSTACLE_STATE, START_STATE
from agent import DQNAgent

def plot_gridworld():
    gridworld = np.zeros((GRID_SIZE, GRID_SIZE))
    gridworld[GOAL_STATE] = 2
    gridworld[OBSTACLE_STATE] = -1
    gridworld[START_STATE] = 1

    cmap = plt.cm.get_cmap('coolwarm', 4)
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(gridworld, cmap=cmap, norm=norm, interpolation='nearest')
    plt.title('GridWorld')

def plot_learning_state(steps_to_goal):
    plt.plot(steps_to_goal)
    plt.ylim(0, 100)
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal')
    plt.title('Learning State')

def save_fig(filename):
    # # DQNAgent 인스턴스 생성
    # discount_factor = agent.gamma
    # epsilon_decay = agent.epsilon_decay
    # epsilon_min = agent.epsilon_min

    # folder_name = f"plot/discount_factor_{discount_factor}"
    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)
    plt.savefig(filename)

def visualize_training(steps_to_goal, filename):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plot_gridworld()

    plt.subplot(1, 2, 2)
    plot_learning_state(steps_to_goal)

    save_fig(filename)
    plt.show()

