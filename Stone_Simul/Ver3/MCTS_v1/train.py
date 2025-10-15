from environment import GameEnvironment
from agent import MCTSAgent
from tqdm import tqdm

def train():
    env = GameEnvironment()
    agent = MCTSAgent(env)

    for episode in tqdm(range(1)):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            print("=====\ntrain.py시작\n=====")
            print("\nBefore Action\n================", state)
            action = agent.select_action(state)
            print("\nAfter Action\n=================", state)
            print("=================================")
            print("=====   select_action 끝끝  =====")
            print(f"Action selected: {action}, Remaining steps: {state['remaining_steps']}")
            state, reward, done = env.step(action)
            total_reward += reward

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
