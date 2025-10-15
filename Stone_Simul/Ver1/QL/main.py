# main.py
from environment import GameEnvironment
from agent import QLearningAgent
from train import train_agent
from models import save_model, load_model
from config import Config

def main():
    episodes = Config.EPISODES
    # Train the agent
    agent = train_agent(episodes)
    
    # Save the model
    save_model(agent, 'q_learning_agent.pkl')

    # Load the model
    q_table = load_model('q_learning_agent.pkl')
    agent.q_table = q_table

    # Test the trained agent
    env = GameEnvironment()
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break

    print(f"Total Reward: {total_reward}")
    print(f"Success counts: {env.success_counts}")
    print(f"Choices left: {env.choices}")

if __name__ == "__main__":
    main()
