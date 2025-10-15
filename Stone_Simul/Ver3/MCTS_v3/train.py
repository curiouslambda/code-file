# train.py

import random
from config import Config
from environment import GameEnvironment, GameState, is_terminal_state, get_reward
from agent import MCTSAgent, MCTSNode

def train(num_episodes=10):
    agent = MCTSAgent()
    env = GameEnvironment(seed=Config.SEED)

    for episode in range(num_episodes):
        state = env.reset()  # 초기화
        root = MCTSNode(state)

        # 이미 시작부터 terminal이면?
        if is_terminal_state(state):
            final_reward = get_reward(state)
            print(f"[Episode {episode+1}] Started in terminal state. Reward: {final_reward}")
            continue

        while not is_terminal_state(state):
            # MCTS 탐색으로 액션을 선택
            best_action, best_child = agent.search(root)

            # 만약 best_action이 None이면, 더 이상 진행 X
            if best_action is None:
                print(f"[Episode {episode+1}] No valid action found. Break.")
                break

            # 실제 환경에 적용
            next_state, reward, done = env.step(best_action)

            # 만약 실제 step 후 상태가 root와 달라졌다면,
            # 해당 next_state를 트리 내 노드로 재설정
            if best_child is not None:
                # best_child의 상태가 실제 next_state와 일치한다고 가정
                root = best_child
                root.parent = None  # 새로운 루트 노드이므로
            else:
                # 혹시 best_child가 None이면, 직접 새 노드를 만들어야 할 수도 있음
                root = MCTSNode(next_state)

            state = next_state

            if done:
                final_reward = get_reward(state)
                if final_reward > 0:
                    print(f"[Episode {episode+1}] Final reward: {final_reward}")
                    print(f" Successes: {state.successes}, Failures: {state.failures}, Prob: {state.success_prob:.2f}")
                break

if __name__ == "__main__":
    train(num_episodes=1000)
