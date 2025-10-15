### train.py

from agent import MCTSAgent
from environment import Environment

if __name__ == "__main__":
    env = Environment()
    agent = MCTSAgent(iterations=1000)

    state = env.reset()
    done = False

    while not done:
        action_node = agent.select_action(env)
        action = [cat for cat in action_node.state[3].keys() if action_node.state[3][cat] > 0][0]
        state, reward, done = env.step(action)

    print(f"Final Result: Successes: {env.successes}, Reward: {reward}")