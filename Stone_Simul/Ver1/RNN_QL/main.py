import numpy as np
import torch
from agent import QLearningAgent
from environment import GameEnvironment
from config import Config
from models import RNNModel
from train import train

def main():
    # Train Q-learning agent
    trained_agent = train()

    # Create RNN model
    model = RNNModel()

    # Generate sample input sequence
    env = GameEnvironment()
    num_trials = env.num_trials
    batch_size = 1  # 배치 크기
    input_sequence = np.zeros((batch_size, num_trials, Config.input_size))
    for i in range(num_trials):
        state = env.get_state()
        input_sequence[0, i, :] = state
        action = trained_agent.choose_action(state, epsilon=0.0)
        reward, _ = env.step(action)
        # print(f"Trial {i+1}: Action {action} selected, Reward: {reward}")

    # Convert input sequence to tensor
    input_tensor = torch.tensor(input_sequence, dtype=torch.float32)

    # Predict outcome using RNN
    with torch.no_grad():
        category_output, probability_output = model(input_tensor)

    predicted_category = torch.argmax(category_output[0, :]).item()
    winning_probability = probability_output[0, 0].item()

    print(f"Predicted category: {predicted_category}, Winning probability: {winning_probability}")


def print_episode_steps():
    env = GameEnvironment()  # 환경 생성
    agent = QLearningAgent(num_actions=env.num_categories)  # Q-Learning 에이전트 생성
    for episode in range(10):  # 10번의 에피소드 실행
        env.reset()  # 환경 초기화
        done = False  # 에피소드 종료 여부 초기화

        while not done:
            state = env.get_state()  # 현재 상태 가져오기
            action = agent.choose_action(state, epsilon=0.0)  # Q-Learning 에이전트로부터 액션 선택
            reward, trial_count = env.step(action)  # 액션 적용 및 보상 받기
            done, reward_type = env.is_episode_done()  # 에피소드 종료 여부 및 보상 타입 확인

            # 에피소드 내 각 스텝의 정보 출력
            print(f"Episode: {episode+1}, Step: {trial_count}, State: {state}, Action: {action}, Reward: {reward}, Reward Type: {reward_type}")

            if done:
                print(f"Episode {episode+1} completed.")


if __name__ == "__main__":
    main()
    print_episode_steps()
