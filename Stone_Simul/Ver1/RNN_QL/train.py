import numpy as np
from agent import QLearningAgent
from environment import GameEnvironment
from config import Config
from tqdm import tqdm

def train():
    env = GameEnvironment()
    agent = QLearningAgent(num_actions=env.num_categories)
    epsilon = 1.0

    for episode in tqdm(range(Config.num_episodes)):
        state = env.get_state()
        done = False
        # print(f"첫번째 state : {state}")

        while not done:
            action = agent.choose_action(state, epsilon)
            reward, _ = env.step(action)
            next_state = env.get_state()
            agent.update_q_table(state, action, reward, next_state, alpha=Config.alpha, gamma=Config.gamma)
            state = next_state
            # print(f"----- 두번째 statae : {state}")

            done = env.is_episode_done()

        env.reset()

    return agent

if __name__ == "__main__":
    trained_agent = train()
    print("Training completed.")
