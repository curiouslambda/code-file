# main.py
import torch
from environment import GameEnv
from agent import PPOAgent
from config import Config

def train():
    config = Config()
    env = GameEnv(config)
    agent = PPOAgent(config)

    timestep = 0
    for episode in range(10):
        state = env.reset(episode)
        total_reward = 0
        for _ in range(config.max_game_rounds):
            action, success_prob = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # 저장할 값들
            log_prob = torch.log(torch.tensor(success_prob))  # 행동의 로그 확률
            _, value = agent.policy(torch.FloatTensor(state))  # 상태 가치 추정
    
            # 메모리에 상태, 행동, 로그 확률, 가치, 보상, 완료 상태 저장
            agent.store_transition(state, action, log_prob, value.item(), reward, done)
            
            total_reward += reward
            state = next_state

            if done:
                break
        
        # 에피소드 결과 출력
        print(f"Episode {episode}, Total Reward: {total_reward}")
        # if total_reward >= 0:
        print("상태 :", state)

        # 매 에피소드 후 PPO 업데이트
        if timestep % config.update_timestep == 0:
            agent.update()  # 메모리에서 샘플링하여 PPO 업데이트
            timestep = 0
        
        timestep += 1

if __name__ == "__main__":
    train()
