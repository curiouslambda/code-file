# agent.py

import math
import random
from copy import deepcopy

from config import Config
from environment import GameState, GameEnvironment, is_terminal_state, get_reward

class MCTSNode:
    def __init__(self, state: GameState, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # action(category) -> MCTSNode
        self.visits = 0
        self.value = 0.0  # 누적 가치 합 (이 노드에서의 추정 리턴)
        
    def is_fully_expanded(self, actions):
        # 가능한 액션들이 모두 children에 있으면 fully expanded
        for action in actions:
            if action not in self.children:
                return False
        return True

    def best_child(self, c_param=Config.EXPLORATION_CONSTANT):
        """
        UCB1 기반으로 자식 노드를 선택
        Q + c * sqrt(ln(N) / n)
        """
        best_score = -float('inf')
        best_action = None
        best_node = None

        for action, child in self.children.items():
            exploitation = child.value / (child.visits + 1e-8)
            exploration = math.sqrt(math.log(self.visits + 1.0) / (child.visits + 1e-8))
            score = exploitation + c_param * exploration

            # 디버깅용
            # print(f"  Action={action}, Exploitation={exploitation:.3f}, Exploration={exploration:.3f}, Score={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_action = action
                best_node = child
        # print(f"[BEST_CHILD] Selected action={best_action}, Score={best_score:.3f}")
        return best_action, best_node


def rollout_policy(state: GameState):
    """
    Rollout 시 임의의 액션(카테고리) 선택
    남은 시도가 있는 카테고리 중에서 랜덤
    """
    possible_actions = [cat for cat in Config.TOTAL_CATEGORIES
                        if state.attempts_left[cat] > 0]
    if not possible_actions:
        return None
    return random.choice(possible_actions)


class MCTSAgent:
    def __init__(self):
        pass

    def select(self, node: MCTSNode):
        """
        트리 내에서 leaf 노드를 찾기 위해 계속해서 best_child를 선택
        """
        current = node
        while True:
            # 종단 상태이거나 자식이 없으면 leaf
            if is_terminal_state(current.state) or len(current.children) == 0:
                return current
            # fully expanded면 best_child로 이동
            _, next_node = current.best_child()
            current = next_node

    def expand(self, node: MCTSNode):
        """
        leaf node 확장
        가능한 액션 중 아직 자식으로 없는 액션 하나를 골라서 자식 노드 생성
        """
        if is_terminal_state(node.state):
            return node

        # 아직 자식이 없는 경우(처음 확장)
        possible_actions = [cat for cat in Config.TOTAL_CATEGORIES
                            if node.state.attempts_left[cat] > 0]
        # 이미 존재하지 않는 자식 액션 찾기
        untried_actions = [action for action in possible_actions if action not in node.children]

        if not untried_actions:
            return node  # 이미 fully expanded

        action = random.choice(untried_actions)
        next_state = self.simulate_step(node.state, action)

        # 디버깅용 로깅
        # print(f"[EXPAND] Creating child for action={action}, state={next_state}")

        child_node = MCTSNode(next_state, parent=node)
        node.children[action] = child_node
        return child_node

    def simulate_step(self, state: GameState, category: str) -> GameState:
        """
        실제 step을 모사(카피)해서 결과 상태를 반환
        """
        next_state = state.copy()
        # 시도 횟수 소모
        next_state.attempts_left[category] -= 1

        # 성공 / 실패 판단
        success = random.random() < next_state.success_prob
        if success:
            next_state.successes[category] += 1
            next_state.success_prob = max(Config.MIN_SUCCESS_PROB,
                                          next_state.success_prob - Config.SUCCESS_PROB_STEP)
        else:
            next_state.failures[category] += 1
            next_state.success_prob = min(Config.MAX_SUCCESS_PROB,
                                          next_state.success_prob + Config.SUCCESS_PROB_STEP)

        # 종료 여부 체크
        next_state.done = is_terminal_state(next_state)
        return next_state

    def rollout(self, state: GameState):
        """
        해당 노드(state)에서부터 rollout_depth만큼 랜덤으로 플레이
        최종 reward를 반환
        """
        current_state = state.copy()
        depth = 0
        while (not is_terminal_state(current_state)) and (depth < Config.ROLLOUT_DEPTH):
            action = rollout_policy(current_state)
            if action is None:
                break
            current_state = self.simulate_step(current_state, action)
            depth += 1
        return get_reward(current_state)

    def backpropagate(self, node: MCTSNode, reward: float):
        """
        MCTS 백업: 루트 방향으로 reward를 전파
        """
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

    def search(self, root: MCTSNode):
        """
        MCTS 탐색
        """
        for _ in range(Config.MCTS_SIMULATIONS):
            # 1. Selection
            leaf = self.select(root)

            # 2. Expansion
            child = self.expand(leaf)

            # 3. Simulation
            reward = self.rollout(child.state)

            # 4. Backpropagation
            self.backpropagate(child, reward)

        # 시뮬레이션 후 최적 자식 선택
        best_action, best_child = root.best_child(c_param=0.0)  # 탐색 계수 0 => exploitation
        return best_action, best_child
