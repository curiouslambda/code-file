import matplotlib.pyplot as plt
import numpy as np
import os
from environment import GRID_SIZE, GOAL_STATE, OBSTACLE_STATE, START_STATE
from agent import DQNAgent
from config import SAVED_PLOT_PATH, AGENT_PARAMS

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
    plt.ylim(0, AGENT_PARAMS["EPISODE"])  # Set y-axis limit to the total number of episodes
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal')
    plt.title('Learning State')

def save_fig(filename):
    plt.savefig(filename)

def visualize_training(steps_to_goal, filename):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plot_gridworld()

    plt.subplot(1, 2, 2)
    plot_learning_state(steps_to_goal)

    save_fig(filename)
    plt.show()

if __name__ == "__main__":
    # Sample usage to visualize training
    agent = DQNAgent(AGENT_PARAMS["STATE_SIZE"], AGENT_PARAMS["ACTION_SIZE"])
    steps_to_goal = [np.random.randint(0, 100) for _ in range(AGENT_PARAMS["EPISODE"])]  # Sample steps to goal
    visualize_training(steps_to_goal, SAVED_PLOT_PATH)
