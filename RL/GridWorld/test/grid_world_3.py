import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler



# 그래프 저장을 위한 폴더 경로 설정
import os

output_folder = "plot"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# 그리드 월드 크기
GRID_SIZE = 5

# 목표 지점 좌표
GOAL_STATE = (4, 4)

# 최대 벽 개수
MAX_WALLS = 5


# 장애물 생성 함수
def generate_obstacle():
    obstacle_x = np.random.randint(0, GRID_SIZE)
    obstacle_y = np.random.randint(0, GRID_SIZE)
    while (obstacle_x, obstacle_y) == GOAL_STATE:  # 목표 지점과 겹치지 않도록 함
        obstacle_x = np.random.randint(0, GRID_SIZE)
        obstacle_y = np.random.randint(0, GRID_SIZE)
    return (obstacle_x, obstacle_y)

# 장애물 좌표
OBSTACLE_STATE = generate_obstacle()


# 행동에 대한 인덱스 정의
ACTIONS = {
    'UP': 0,
    'DOWN': 1,
    'LEFT': 2,
    'RIGHT': 3
}

# 초기 Q 테이블 초기화
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# 학습 파라미터 설정
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPISODES = 1000
EPSILON = 0.1

# 경험 메모리
# memory = []

# # 학습률 감소 스케줄러
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda=lambda epoch: 1.0 - epoch / EPISODES)



# 그리드 월드를 나타내는 함수
def plot_grid(state):
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    grid[state] = 1
    grid[GOAL_STATE] = 0.5
    for wall in MAX_WALLS:
        grid[wall] = -1
    plt.imshow(grid, cmap='Blues', interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.title('Grid World')
    plt.show()

steps_to_goal = []
# Q-learning 알고리즘
def q_learning():
    for episode in range(EPISODES):
        # 랜덤 벽 생성
        WALLS = generate_walls(MAX_WALLS)
        state = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
        steps = 0
        while state != GOAL_STATE:
            if np.random.rand() < EPSILON:
                action = np.random.choice(list(ACTIONS.values()))
            else:
                action = np.argmax(Q_table[state])
            next_state = take_action(state, action)
            reward = 1 if next_state == GOAL_STATE else 0
            
            # 벽을 통과하지 못하도록 처리
            if next_state in WALLS:
                next_state = state
            
            # 경험 메모리에 저장
            # memory.append((state, action, reward, next_state))
            
            if next_state == GOAL_STATE:
                steps_to_goal.append(steps)
            else:
                Q_table[state][action] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(Q_table[next_state]) - Q_table[state][action])
            state = next_state
            steps += 1

            # 중간 진행 상황 출력
            if steps % 5000000 == 0:
                print(f'Episode: {episode}, Steps: {steps}')
        
        if state == GOAL_STATE:  # 목표 지점에 도달한 경우 단계 수 기록
            steps_to_goal.append(steps)
        
        # 에피소드 완료 상황 출력
        if episode % 100 == 0:
            print(f'Episode: {episode}, Steps: {steps}, Completed')

# 랜덤 벽 생성 함수
def generate_walls(max_walls):
    walls = []
    while len(walls) < max_walls:
        x = np.random.randint(0, GRID_SIZE)
        y = np.random.randint(0, GRID_SIZE)
        if (x, y) != GOAL_STATE and (x, y) not in walls:
            walls.append((x, y))
    return walls

# 행동을 수행하고 다음 상태를 반환하는 함수
def take_action(state, action):
    if action == ACTIONS['UP']:
        return max(state[0] - 1, 0), state[1]
    elif action == ACTIONS['DOWN']:
        return min(state[0] + 1, GRID_SIZE - 1), state[1]
    elif action == ACTIONS['LEFT']:
        return state[0], max(state[1] - 1, 0)
    elif action == ACTIONS['RIGHT']:
        return state[0], min(state[1] + 1, GRID_SIZE - 1)


# Q 테이블 시각화
def plot_q_table(Q_table):
    fig, axs = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(10, 10))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            q_values = Q_table[i, j, :]
            max_action = np.argmax(q_values)
            axs[i, j].imshow([[q_values[max_action]]], cmap='Blues', vmin=0, vmax=1)
            axs[i, j].text(0.5, 0.5, f'{q_values[max_action]:.2f}', fontsize=12, ha='center', color='black')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].set_title(f'State: ({i}, {j})')
    plt.tight_layout()
    plt.show()


# 학습 수행
q_learning()

# 최종 Q 테이블 출력
print("Final Q Table:")
print(Q_table)

# 최적 정책 시각화
optimal_policy = np.argmax(Q_table, axis=2)
print("Optimal Policy:")
print(optimal_policy)

# 시작 지점에서 최적 정책을 따라가며 시각화
plot_grid((0, 0))

# Q 테이블 시각화
plot_q_table(Q_table)


# 그래프를 이미지 파일로 저장
plt.figure(figsize=(10, 5))

# Grid World 그래프
plt.subplot(1, 2, 1)
plot_grid((0, 0))
plt.title('Grid World')

# Q Table 그래프
plt.subplot(1, 2, 2)
plot_q_table(Q_table)
plt.title('Q Table')

# 그래프 저장
plt.savefig(os.path.join(output_folder, f"LR{LEARNING_RATE}_DF{DISCOUNT_FACTOR}_EPI{EPISODES}_EPS{EPSILON}.png"))
plt.close()  # 그래프 창 닫기