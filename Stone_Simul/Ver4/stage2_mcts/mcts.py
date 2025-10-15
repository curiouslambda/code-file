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
        return self._best_child(root, 0.0).state
        
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
        """UCB1 공식을 사용하여 최적의 자식 노드 선택"""
        best_score = float('-inf')
        best_child = None
        
        for child in node.children.values():
            # UCB1 공식: Q(s,a) + c_puct * sqrt(N(s)) / (1 + N(s,a))
            exploitation = child.value_sum / (child.visit_count + 1e-8)
            exploration = c_puct * np.sqrt(node.visit_count) / (child.visit_count + 1e-8)
            score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child 