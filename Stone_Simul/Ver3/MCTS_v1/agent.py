import math
import random
from config import MCTS_CONFIG

class MCTSNode:
    def __init__(self, state, parent=None, category=None):
        self.state = state
        self.parent = parent
        self.category = category  # 선택된 category 저장
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        # state["remaining_steps"]에서 값이 > 0인 항목만 고려하여 확장 완료 여부 판단
        remaining_actions = [k for k, v in self.state["remaining_steps"].items() if v > 0]
        return len(self.children) == len(remaining_actions)


    def best_child(self, exploration_weight=1.0):
        best_score = float('-inf')
        best_child = None

        for child in self.children:
            # child.visits == 0 방지 로직 추가
            if child.visits == 0:
                score = float('inf')
            else:
                score = (
                    child.value / child.visits + 
                    exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                )

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

class MCTSAgent:
    def __init__(self, environment):
        self.environment = environment
        self.simulations = MCTS_CONFIG["simulations"]
        self.exploration_weight = MCTS_CONFIG["exploration_weight"]

    def select_action(self, state):
        root = MCTSNode(state)
        for _ in range(self.simulations):
            leaf = self._select(root)
            reward = self._simulate(leaf)
            self._backpropagate(leaf, reward)
        return root.best_child(exploration_weight=0).category

    def _select(self, node):
        while not node.state["done"] and node.is_fully_expanded():
            node = node.best_child()
        if not node.is_fully_expanded():
            return self._expand(node)
        return node

    def _expand(self, node):
        untried_actions = [
            category for category in node.state["remaining_steps"]
            if node.state["remaining_steps"][category] > 0
        ]
        # print(f"Remaining steps: {node.state['remaining_steps']}")

        if not untried_actions:
            raise ValueError("No valid actions to expand.")  # 디버깅용 예외 처리
        
        category = random.choice(untried_actions)
        next_state, _, _ = self.environment.step(category)
        print("&&&&&&& expand에 있는 step &&&&&&")
        print(f"\n{next_state}\n")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        child_node = MCTSNode(next_state, parent=node, category=category)  # category 추가

        node.children.append(child_node)
        return child_node

    def _simulate(self, node):
        # state = node.state.copy()  # 시뮬레이션 중 상태를 안전하게 복사
        simulated_env = self.environment.clone() 
        print("*****************************")
        print("simulate에 있는 상태\n\n") # ✅ 환경 복제
        state = simulated_env.get_state()
        last_category = None  # 마지막으로 선택된 category를 저장
        # print(f"상태 : {state}")
        while not state["done"]:
            # print("***** 이게 나왔으면 while문으로 들어온 것 *****")
            # print(state["done"])
            valid_categories = [k for k, v in state["remaining_steps"].items() if v > 0]

            last_category = random.choice(valid_categories)
            state, _, done = self.environment.step(last_category)
            state["done"] = done

        # 마지막으로 선택된 category를 기반으로 보상을 계산
        return self.environment.get_reward(last_category, state["done"])

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
