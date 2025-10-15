import numpy as np
import time
from environment import GRID_SIZE, OBSTACLE_STATE, GOAL_STATE

# 행동에 대한 인덱스 정의
ACTIONS = {
    'UP': 0,
    'DOWN': 1,
    'LEFT': 2,
    'RIGHT': 3
}

Q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

LEARNING_RATE = 0.03
DISCOUNT_FACTOR = 0.8
EPISODES = 55000
EPSILON = 0.05

steps_to_goal = []

def q_learning():
    start_time = time.time()
    for episode in range(EPISODES):
        state = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
        steps = 0
        while state != GOAL_STATE and state != OBSTACLE_STATE:
            if np.random.rand() < EPSILON:
                action = np.random.choice(list(ACTIONS.values()))
            else:
                action = np.argmax(Q_table[state])
            next_state = take_action(state, action)
            reward = 1 if next_state == GOAL_STATE else 0
            if next_state == OBSTACLE_STATE:
                reward = -1
            else:
                Q_table[state][action] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(Q_table[next_state]) - Q_table[state][action])
            state = next_state
            steps += 1

            if steps % 5000000 == 0:
                print(f'Episode: {episode}, Steps: {steps}')

        if state == GOAL_STATE or state == OBSTACLE_STATE:
            steps_to_goal.append(steps)

        if episode % 5000 == 0:
            print(f'Episode: {episode}, Steps: {steps}, Completed')
    total_time = time.time() - start_time
    print(f'Training Time: {total_time:.2f} seconds')
    
    optimal_policy = np.argmax(Q_table, axis=2)
    return Q_table, steps_to_goal, optimal_policy

# 행동을 수행하고 다음 상태를 반환하는 함수
def take_action(state, action):
    next_state = None
    if action == ACTIONS['UP']:
        next_state = max(state[0] - 1, 0), state[1]
    elif action == ACTIONS['DOWN']:
        next_state = min(state[0] + 1, GRID_SIZE - 1), state[1]
    elif action == ACTIONS['LEFT']:
        next_state = state[0], max(state[1] - 1, 0)
    elif action == ACTIONS['RIGHT']:
        next_state = state[0], min(state[1] + 1, GRID_SIZE - 1)
    
    # 다음 상태가 장애물인지 확인
    if next_state == OBSTACLE_STATE:
        return state
    
    return next_state