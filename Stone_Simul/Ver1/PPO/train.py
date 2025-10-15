# train.py
from environment import GameEnv
from agent import PPOAgent
from config import Config
import torch
from tqdm import tqdm
import json

def train_ppo(episodes):
    env = GameEnv()
    state_dim = len(env.reset())
    action_dim = 3  # A, B, C
    
    agent = PPOAgent(state_dim, action_dim)
    scores = []

    for e in tqdm(range(episodes)):
        state = env.reset()
        total_reward = 0
        attempts = []
        success = []
        failure = []
        prob = []
        data = {
            'state_data' : {
                'attempts' : attempts,
                'success' : success,
                'failure' : failure,
                'probability' : prob
            }
        }
        for time in range(Config.max_choices):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            attempts.append(state[0:3])
            success.append(state[3:6])
            failure.append(state[6:9])
            prob.append(state[9:10])
            total_reward += reward

            agent.update_epsilon(time)
            
            if done:
                print(f"episode: {e+1}/{episodes}, score: {total_reward}, epsilon : {agent.epsilon}")
                break
        # print(f"상태 : {state}")
        # JSON 파일로 저장
        with open(f'./data/state_data_episode_{e+1}.json', 'w') as f:
            json.dump(data, f)
        agent.learn()
        scores.append(total_reward)
    
    torch.save(agent.policy_net.state_dict(), 'ppo_policy_model.pth')
    torch.save(agent.value_net.state_dict(), 'ppo_value_model.pth')
    return scores
