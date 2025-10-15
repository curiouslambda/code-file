import numpy as np
from agent import q_learning, LEARNING_RATE, DISCOUNT_FACTOR, EPISODES, EPSILON
from utils import visualize_training, save_fig

if __name__ == "__main__":
    Q_table, steps_to_goal, optimal_policy = q_learning()
    print("Final Q Table:")
    print(Q_table)
    print("Optimal Policy:")
    print(optimal_policy)
    visualize_training(Q_table, steps_to_goal, optimal_policy,\
                       f'plot/[LR]{LEARNING_RATE}_[DF]{DISCOUNT_FACTOR}_[EPI]{EPISODES}_[EPS]{EPSILON}.png')