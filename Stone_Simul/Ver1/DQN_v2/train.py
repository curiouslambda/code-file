# train.py
from environment import GameEnv
from agent import DQNAgent
from config import Config
import torch
from tqdm import tqdm
import json

def train_dqn(episodes):
    state_dim = 10
    action_dim = 3  # A, B, C
    agent = DQNAgent(state_dim, action_dim)
    env = GameEnv(agent, state_dim, action_dim)
    scores = []

    success_count = 0

    for e in tqdm(range(episodes)):
        state = env.reset()
        total_reward = 0
        states = []
        data = {
            'states' : states
        }
        for time in range(Config.max_choices):
            # print(f"상태 : {state}")
            action = agent.act(state)
            # print(f"확률 : {state[-1]}, 액션 : {action}")
            # print(f"이전 : {state}")
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            # print(f"=== 다음 === {state}")
            states.append(state)
            total_reward += reward
            # print(f"리워드 확인{total_reward}")
            if total_reward >=5:
                success_count += 1
            if done:
                print(f"episode: {e+1}/{episodes}, score: {total_reward}, epsilon: {agent.epsilon:.2}")
                break
            agent.replay()

        print(f"에피소드 {e+1}의 성공 횟수 : {state}")
        # if (e+1) % 20 == 0:
        #     print(f"episode: {e+1}/{episodes}, score: {total_reward}, epsilon: {agent.epsilon:.2}")
            
        filename = './data/' + f'episode{e+1}_data.json'
        with open (filename, 'w') as f:
            json.dump(data, f)
        scores.append(total_reward)

        if e % agent.config.target_update == 0:
            agent.update_target_model()
    
    # print(f"77돌 이상 횟수 : {success_count}")
    # GPU 메모리 사용량 출력
        # print(f"Allocated memory: {torch.cuda.memory_allocated()} bytes")
        # print(f"Reserved memory: {torch.cuda.memory_reserved()} bytes")


    torch.save(agent.model.state_dict(), 'dqn_model.pth')
    return scores
