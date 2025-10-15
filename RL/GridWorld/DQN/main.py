from agent import dqn_learning, DQNAgent
from utils import visualize_training, save_fig

if __name__ == "__main__":
    _, steps_to_goal = dqn_learning()

    visualize_training(steps_to_goal, 'dqn_learning.png')
    