import numpy as np
from typing import Dict, List, Tuple
import logging
from common.env import LostArkStoneEnv, State

class Node:
    def __init__(self, state: State, parent=None):
        self.state = state
        self.parent = parent
        self.children: Dict[str, 'Node'] = {}  # action -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.untried_actions = ['A', 'B', 'C']  # 초기에는 모든 행동이 시도되지 않음

class MCTS:
    def __init__(self, env: LostArkStoneEnv, config: Dict):
        self.env = env
        self.config = config
        self.c_puct = float(config.get('c_puct', 1.0))  # 탐색-활용 트레이드오프 파라미터
        self.max_simulations = int(config.get('max_simulations', 1000))
        
    def search(self, root_state: State) -> str:
        """MCTS 알고리즘 실행"""
        root = Node(root_state)
        
        for _ in range(self.max_simulations):
            # 1. Selection
            node = self._select(root)
            
            # 2. Expansion
            if not node.state.remaining_slots == 0 and not self.env.is_terminal(node.state):
                node = self._expand(node)
            
            # 3. Simulation
            value = self._simulate(node.state)
            
            # 4. Backpropagation
            self._backpropagate(node, value)
        
        # 최적 행동 선택
        return self._best_action(root)
        
    def _best_action(self, node: Node) -> str:
        """가장 많이 방문된 행동 선택 (tie-breaking 포함)"""
        best_actions = []
        best_visit_count = -1
        
        for action, child in node.children.items():
            if child.visit_count > best_visit_count:
                best_visit_count = child.visit_count
                best_actions = [action]
            elif child.visit_count == best_visit_count:
                best_actions.append(action)
        
        # 방문된 자식이 없으면 랜덤 선택
        if not best_actions:
            best_action = np.random.choice(['A', 'B', 'C'])
        else:
            # 동률이 있으면 랜덤 선택
            best_action = np.random.choice(best_actions)
            
        return best_action
        
    def _select(self, node: Node) -> Node:
        """트리에서 가장 유망한 노드 선택"""
        while not node.untried_actions and not self.env.is_terminal(node.state):
            node = self._best_child(node, self.c_puct)
        return node
    
    def _expand(self, node: Node) -> Node:
        """새로운 노드 확장"""
        action = np.random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        
        # 다음 상태 계산
        success = np.random.random() < node.state.p
        next_state = self.env.get_next_state(node.state, action, success)
        
        # 새로운 노드 생성
        child = Node(next_state, parent=node)
        node.children[action] = child
        
        return child
    
    def _simulate(self, state: State) -> float:
        """현재 상태에서 시뮬레이션 실행"""
        current_state = state
        while not self.env.is_terminal(current_state):
            action = np.random.choice(['A', 'B', 'C'])
            success = np.random.random() < current_state.p
            current_state = self.env.get_next_state(current_state, action, success)
        
        return self.env.get_reward(current_state)
    
    def _backpropagate(self, node: Node, value: float):
        """가치 역전파"""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent
    
    def _best_child(self, node: Node, c_puct: float) -> Node:
        """표준 UCB1 공식을 사용하여 최적의 자식 노드 선택 (tie-breaking 포함)"""
        best_children = []
        best_score = float('-inf')
        
        for child in node.children.values():
            # 표준 UCB1 공식: Q(s,a) + c_puct * sqrt(ln(N(s)) / N(s,a))
            exploitation = child.value_sum / (child.visit_count + 1e-8)
            # 부모 방문 횟수에 +1을 추가하여 log(1) = 0이 되도록 함
            exploration = c_puct * np.sqrt(np.log(node.visit_count + 1)) / (child.visit_count + 1e-8)
            score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_children = [child]
            elif abs(score - best_score) < 1e-8:  # 동률 처리 (부동소수점 오차 고려)
                best_children.append(child)
        
        # 동률이 있으면 랜덤 선택
        return best_children[np.random.randint(len(best_children))] 