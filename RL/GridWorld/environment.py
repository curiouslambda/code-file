import numpy as np

GRID_SIZE = 5
START_STATE = (0, 0)
GOAL_STATE = (4, 4)

def generate_obstacle():
    obstacle_x = np.random.randint(0, GRID_SIZE)
    obstacle_y = np.random.randint(0, GRID_SIZE)
    while (obstacle_x, obstacle_y) == GOAL_STATE:
        obstacle_x = np.random.randint(0, GRID_SIZE)
        obstacle_y = np.random.randint(0, GRID_SIZE)
    return (obstacle_x, obstacle_y)

OBSTACLE_STATE = generate_obstacle()