import numpy as np
import matplotlib.pyplot as plt
import os
import time

# 그리드 월드 크기
GRID_SIZE = 5

# 시작 지점 좌표
START_STATE = (0, 0)
# 목표 지점 좌표
GOAL_STATE = (4, 4)


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
LEARNING_RATE = 0.05
DISCOUNT_FACTOR = 0.85
EPISODES = 50000
EPSILON = 0.05

steps_to_goal = []
# Q-learning 알고리즘
def q_learning():
    start_time = time.time()
    for episode in range(EPISODES):
        state = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
        steps = 0  # 에피소드 시작 시 단계 수 초기화
        while state != GOAL_STATE and state != OBSTACLE_STATE:  # 목표 지점이나 장애물에 도달하지 않은 동안 반복
            if np.random.rand() < EPSILON:
                action = np.random.choice(list(ACTIONS.values()))
            else:
                action = np.argmax(Q_table[state])
            next_state = take_action(state, action)
            reward = 1 if next_state == GOAL_STATE else 0
            if next_state == OBSTACLE_STATE:  # 장애물에 도달한 경우
                reward = -1
            else:  # 장애물에 도달하지 않은 경우에만 Q 테이블 업데이트
                Q_table[state][action] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(Q_table[next_state]) - Q_table[state][action])
            state = next_state
            steps += 1  # 단계 수 증가

            # 중간 진행 상황 출력
            if steps % 5000000 == 0:
                print(f'Episode: {episode}, Steps: {steps}')
        
        if state == GOAL_STATE or state == OBSTACLE_STATE:  # 목표 지점이나 장애물에 도달한 경우 단계 수 기록 및 에피소드 종료
            steps_to_goal.append(steps)
        
        # 에피소드 완료 상황 출력
        if episode % 5000 == 0:
            print(f'Episode: {episode}, Steps: {steps}, Completed')
    total_time = time.time() - start_time
    print(f'Training Time: {total_time:.2f} seconds')


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

def visualize_training(Q_table, steps_to_goal, optimal_policy):   
    # 하나의 figure와 네 개의 subplot 생성
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 그리드월드 시각화 및 시작, 목표, 장애물 상태의 색상 구분
    gridworld = np.zeros((GRID_SIZE, GRID_SIZE))
    gridworld[GOAL_STATE] = 2
    gridworld[OBSTACLE_STATE] = -1
    gridworld[START_STATE] = 1

    # 시작, 목표, 장애물 상태에 대한 색상 설정
    cmap = plt.cm.get_cmap('coolwarm', 4)
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    axes[0].imshow(gridworld, cmap=cmap, norm=norm, interpolation='nearest')
    axes[0].set_title('GridWorld')

    # 학습 진행 상황 그래프
    axes[1].plot(steps_to_goal)
    axes[1].set_ylim(0, 30)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps to Goal')
    axes[1].set_title('Learning State')

    # Q 테이블 색상 설정
    q_cmap = plt.cm.get_cmap('YlGn', 256)
    q_bounds = np.linspace(0, 1, 256) # 컬러맵 범위 설정
    q_norm = plt.cm.colors.BoundaryNorm(q_bounds, q_cmap.N)

    # Q 테이블 시각화 및 숫자 값 표시
    im = axes[2].imshow(np.max(Q_table, axis=2), cmap=q_cmap, norm=q_norm, interpolation='nearest')

    # 각 셀에 숫자 값 추가
    for i in range(Q_table.shape[0]):
        for j in range(Q_table.shape[1]):
            text = axes[2].text(j, i, np.around(np.max(Q_table[i, j]), decimals=2),
                                 ha="center", va="center", color="black")

    axes[2].set_title('Q Table')
    fig.colorbar(im, ax=axes[2])


    # 그리드월드 시각화 및 시작, 목표, 장애물 상태의 색상 구분
    gridworld = np.zeros((GRID_SIZE, GRID_SIZE))
    gridworld[GOAL_STATE] = 2
    gridworld[OBSTACLE_STATE] = -1
    gridworld[START_STATE] = 1

    # 시작, 목표, 장애물 상태에 대한 색상 설정
    cmap = plt.cm.get_cmap('coolwarm', 4)
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    axes[3].imshow(gridworld, cmap=cmap, norm=norm, interpolation='nearest')
    axes[3].set_title('Optimal Policy')


    # Optimal policy 시각화
    arrows = ['↑', '↓', '←', '→']  # 각 행동에 대응하는 화살표
    for i in range(Q_table.shape[0]):
        for j in range(Q_table.shape[1]):
            action = optimal_policy[i, j]
            axes[3].text(j, i, arrows[action], ha='center', va='center', color='black', fontsize=12)

    axes[3].axis('off')  # Optimal policy는 새로운 subplot에 표시됩니다.
    # axes[3].text(0.5, 0.5, 'Optimal Policy', ha='center', va='center', fontsize=14)

    plt.savefig(f'plot/[LR]{LEARNING_RATE}_[DF]{DISCOUNT_FACTOR}_[EPI]{EPISODES}_[EPS]{EPSILON}.png')
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

# 시각화 함수 호출
visualize_training(Q_table, steps_to_goal, optimal_policy)

