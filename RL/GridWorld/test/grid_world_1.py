import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

WORLD_SIZE = 2
WORLD = np.zeros((WORLD_SIZE, WORLD_SIZE)) 

# 에이전트 시작 위치와 목표 위치 지정
START = [0, 0]  
GOAL = [2, 2]   

# 장애물 랜덤 생성
for i in range(3):
    WORLD[np.random.randint(0, WORLD_SIZE), np.random.randint(0, WORLD_SIZE)] = -1
    print(WORLD)
    
Q = np.zeros((WORLD_SIZE, WORLD_SIZE, 4))
rewards = [[0] * WORLD_SIZE for _ in range(WORLD_SIZE)] 

ep_num = 0
# Q-learning 알고리즘 구현
for episode in tqdm(range(1000)):
    
    state = START
    print(f"지금은 {ep_num+1}번째 에피소드")

    while state != GOAL:

        # 행동 선택 (ε-greedy 방식)
        if np.random.random() < 0.1:
            action = np.random.randint(4)
            # print("랜덤액션 :", action)
        else:
            action = np.argmax(Q[state[0], state[1]])
            # print("Q값 max 액션 : ", action)
        
        if action == 0: # up
            next_state = [state[0] - 1, state[1]] 
            next_state[0] = max(next_state[0], 0)
            #print("action이 0일때 --",next_state)

        elif action == 1: # down
            next_state = [state[0] + 1, state[1]]
            next_state[0] = min(next_state[0], WORLD_SIZE - 1)
            # print("action이 1일때 --",next_state)
            
        elif action == 2: # left
            next_state = [state[0], state[1] - 1]
            next_state[1] = max(next_state[1], 0)
            # print("action이 2일때 --",next_state)

        else: # right
            next_state = [state[0], state[1] + 1] 
            next_state[1] = min(next_state[1], WORLD_SIZE - 1)
            # print("action이 3일때 --",next_state)

        # if next_state == [2,2]:
        #     print("GOAL에 도달")

        # 보상과 상태 업데이트
        reward = -1 if WORLD[next_state[0], next_state[1]] == -1 else 0
        if next_state == GOAL:
            reward = 1
        
        #print("드디어 보상 ----------", reward)
        Q[state[0], state[1], action] = Q[state[0], state[1], action] + 0.5*(reward + np.max(Q[next_state[0], next_state[1]]) - Q[state[0], state[1], action])  
        state = next_state
    
    ep_num += 1

# rewards = [] # 보상 저장

# for episode in range(1000):

#   reward_sum = 0
#   while state != GOAL:
#     # Q-learning 수행
#     reward_sum += reward

#   rewards.append(reward_sum)
  
#   if episode % 100 == 0:
#     plt.clf()
#     plt.bar(range(len(rewards)), rewards)
#     plt.pause(0.1)

# plt.show()



print("Q table:")
print(Q)

# # 학습된 경로 보기
# state = START 
# path = [state]
# while state != GOAL:
#     next_state = np.unravel_index(np.argmax(Q[state[0], state[1]]), (WORLD_SIZE, WORLD_SIZE))
#     path.append(next_state)
#     state = next_state

# print("Path:")
# print(path)



# # 그리드와 경로 시각화 
# WORLD[GOAL[0], GOAL[1]] = 2
# plt.imshow(WORLD, cmap='gray')
# plt.plot([p[1] for p in path], [p[0] for p in path]) 
# plt.show()