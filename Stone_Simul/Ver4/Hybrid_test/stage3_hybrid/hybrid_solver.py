"""
DP 가이던스 기반 하이브리드 MCTS 솔버
DP 결과를 MCTS 트리 탐색 초반에 가이던스로 활용
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from common.env import LostArkStoneEnv, State
from .utils import DPGuidanceUtils

class HybridNode:
    """하이브리드 MCTS 노드"""
    def __init__(self, state: State, parent=None):
        self.state = state
        self.parent = parent
        self.children: Dict[str, 'HybridNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.untried_actions = ['A', 'B', 'C']

class HybridSolver:
    """DP 가이던스 기반 하이브리드 MCTS 솔버"""
    
    def __init__(self, env: LostArkStoneEnv, config: Dict):
        """
        Args:
            env: LostArkStoneEnv 인스턴스
            config: 설정 딕셔너리
        """
        self.env = env
        self.config = config
        
        # MCTS 파라미터
        self.c_puct = float(config.get('c_puct', 1.0))
        self.max_simulations = int(config.get('max_simulations', 1000))
        
        # 하이브리드 파라미터
        self.guidance_weight = float(config.get('guidance_weight', 0.3))
        self.guidance_threshold = int(config.get('guidance_threshold', 5))
        self.guidance_temperature = float(config.get('guidance_temperature', 0.5))
        self.exploration_weight = float(config.get('exploration_weight', 0.1))
        
        # DP 가이던스 유틸리티
        dp_values_path = config.get('dp_values_path', 'outputs/dp_values.npz')
        self.dp_guidance = DPGuidanceUtils(env, dp_values_path)
        
        # 통계
        self.guidance_usage_count = 0
        self.guidance_agreement_count = 0
        self.total_simulations = 0
        self.simulation_guidance_count = 0  # 시뮬레이션에서 DP 가이던스 사용 횟수
        self.total_simulation_steps = 0     # 전체 시뮬레이션 스텝 수
        
    def search(self, root_state: State) -> str:
        """하이브리드 MCTS 알고리즘 실행"""
        root = HybridNode(root_state)
        
        for sim in range(self.max_simulations):
            # 1. Selection (DP 가이던스 적용)
            node = self._select(root)
            
            # 2. Expansion
            if not self.env.is_terminal(node.state):
                node = self._expand(node)
            
            # 3. Simulation
            value = self._simulate(node.state)
            
            # 4. Backpropagation
            self._backpropagate(node, value)
        
        # 최적 행동 선택
        best_action = self._best_action(root)
        
        # 가이던스 일치율 계산 (DP 가이던스가 적용되는 경우에만)
        if self.dp_guidance.is_guidance_applicable(root_state, self.guidance_threshold):
            dp_recommended_action = self.dp_guidance.get_dp_action(root_state)
            if best_action == dp_recommended_action:
                self.guidance_agreement_count += 1
        
        return best_action
    
    def _select(self, node: HybridNode) -> HybridNode:
        """DP 가이던스를 적용한 노드 선택"""
        while not node.untried_actions and not self.env.is_terminal(node.state):
            # DP 가이던스 적용 가능 여부 확인
            if self.dp_guidance.is_guidance_applicable(node.state, self.guidance_threshold):
                # DP 가이던스 적용
                selected_node = self._select_with_guidance(node)
                if selected_node is not None:
                    node = selected_node
                    self.guidance_usage_count += 1
            else:
                # 일반 UCB 선택
                best_action = self._best_child(node, self.c_puct)
                if best_action and best_action in node.children:
                    node = node.children[best_action]
                else:
                    # 자식이 없는 경우 랜덤 선택
                    available_actions = list(node.children.keys())
                    if available_actions:
                        random_action = np.random.choice(available_actions)
                        node = node.children[random_action]
                    else:
                        break
        
        return node
    
    def _select_with_guidance(self, node: HybridNode) -> Optional[HybridNode]:
        """DP 가이던스를 적용한 노드 선택"""
        best_score = float('-inf')
        best_child = None
        
        for action, child in node.children.items():
            # 하이브리드 UCB 점수 계산
            score = self.dp_guidance.get_hybrid_ucb_score(
                node, action, self.c_puct, self.guidance_weight
            )
            
            if score > best_score:
                best_score = score
                best_child = child
        
        # 자식이 없는 경우 원본 노드 반환
        if best_child is None:
            return node
        
        return best_child
    
    def _expand(self, node: HybridNode) -> HybridNode:
        """새로운 노드 확장 (표준 MCTS 방식)"""
        if not node.untried_actions:
            return node
        
        # 모든 시도하지 않은 행동에 대해 자식 노드 생성
        for action in list(node.untried_actions):
            # 성공 확률에 따라 성공/실패 결정
            success = np.random.random() < node.state.p
            next_state = self.env.get_next_state(node.state, action, success)
            
            # 새로운 노드 생성
            child = HybridNode(next_state, parent=node)
            node.children[action] = child
        
        # 모든 시도하지 않은 행동을 제거
        node.untried_actions.clear()
        
        # 첫 번째 자식 노드 반환 (어떤 자식이든 상관없음)
        if node.children:
            return list(node.children.values())[0]
        
        return node
    
    def _simulate(self, state: State) -> float:
        """현재 상태에서 시뮬레이션 실행 (DP 가이던스 적용)"""
        current_state = state
        
        while not self.env.is_terminal(current_state):
            # 통계 카운팅: 전체 시뮬레이션 스텝 수 증가
            self.total_simulation_steps += 1
            
            # DP 가이던스 적용 가능 여부 확인
            if self.dp_guidance.is_guidance_applicable(current_state, self.guidance_threshold):
                # DP 기반 행동 선택
                action = self.dp_guidance.get_guided_action_selection(
                    current_state, self.exploration_weight
                )
                # 통계 카운팅: DP 가이던스 사용 횟수 증가
                self.simulation_guidance_count += 1
            else:
                # 일반 랜덤 선택
                action = np.random.choice(['A', 'B', 'C'])
            
            success = np.random.random() < current_state.p
            current_state = self.env.get_next_state(current_state, action, success)
        
        return self.env.get_reward(current_state)
    
    def _backpropagate(self, node: HybridNode, value: float):
        """가치 역전파"""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent
    
    def _best_child(self, node: HybridNode, c_puct: float) -> str:
        """최적의 자식 노드 선택"""
        best_score = float('-inf')
        best_action = None
        
        for action, child in node.children.items():
            # UCB1 공식
            exploitation = child.value_sum / (child.visit_count + 1e-8)
            exploration = c_puct * np.sqrt(node.visit_count) / (child.visit_count + 1e-8)
            score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_action = action
        
        # 자식이 없는 경우 기본값 반환
        if best_action is None:
            return 'A'  # 기본 행동
        
        return best_action

    def _best_action(self, node: HybridNode) -> str:
        """방문 횟수 기반 최적 행동 선택 (tie-breaking)"""
        best_visit_count = -1
        best_action = None
        
        for action, child in node.children.items():
            if child.visit_count > best_visit_count:
                best_visit_count = child.visit_count
                best_action = action
        
        # 자식이 없는 경우 기본값 반환
        if best_action is None:
            return 'A'  # 기본 행동
        
        return best_action
    
    def _build_mcts_tree(self, state: State) -> HybridNode:
        """MCTS 트리 구축 (search와 유사하지만 최적 행동 반환하지 않음)"""
        root = HybridNode(state)
        
        for sim in range(self.max_simulations):
            # 1. Selection (DP 가이던스 적용)
            node = self._select(root)
            
            # 2. Expansion
            if not self.env.is_terminal(node.state):
                node = self._expand(node)
            
            # 3. Simulation
            value = self._simulate(node.state)
            
            # 4. Backpropagation
            self._backpropagate(node, value)
            
            # 총 개별 시뮬레이션 횟수 카운팅
            self.total_simulations += 1
        
        return root
    
    def _get_action_probabilities(self, root: HybridNode) -> List[float]:
        """방문 횟수 기반 행동 확률 분포 계산"""
        total_visits = sum(child.visit_count for child in root.children.values())
        
        if total_visits == 0:
            return [1/3, 1/3, 1/3]  # 균등 분포
        
        probs = []
        for action in ['A', 'B', 'C']:
            if action in root.children:
                prob = root.children[action].visit_count / total_visits
            else:
                prob = 0.0
            probs.append(prob)
        
        # 정규화 (확률 합이 1이 되도록)
        if sum(probs) > 0:
            probs = [p / sum(probs) for p in probs]
        else:
            probs = [1/3, 1/3, 1/3]
        
        return probs
    
    def get_action_statistics(self, root_state: State, num_episodes: int = 100) -> Dict[str, Union[Dict[str, float], float]]:
        """하이브리드 솔버의 행동 통계 수집"""
        action_counts = {'A': 0, 'B': 0, 'C': 0}
        success_rates = {'A': [], 'B': [], 'C': []}
        episode_lengths = []
        success_count = 0
        guidance_agreement_count = 0
        guidance_applicable_count = 0  # DP 가이던스가 적용된 경우의 수
        episode_data = []  # 에피소드 경로 데이터 수집
        
        for episode in range(num_episodes):
            state = root_state
            episode_length = 0
            actions_taken = []
            trajectory = []  # 에피소드 경로 기록
            
            while not self.env.is_terminal(state):
                # 현재 상태 기록
                trajectory.append({
                    'a': state.a,
                    'b': state.b,
                    'c': state.c,
                    'p': state.p,
                    'remaining_slots': state.remaining_slots,
                    'action': None  # 나중에 설정
                })
                
                # 한 번의 MCTS 탐색으로 확률 분포 생성
                root = self._build_mcts_tree(state)
                action_probs = self._get_action_probabilities(root)
                action = np.random.choice(['A', 'B', 'C'], p=action_probs)
                actions_taken.append(action)
                
                # 경로에 행동 기록
                trajectory[-1]['action'] = action
                
                # 가이던스 일치율 계산 (DP 가이던스가 적용되는 경우에만)
                if self.dp_guidance.is_guidance_applicable(state, self.guidance_threshold):
                    guidance_applicable_count += 1
                    dp_recommended_action = self.dp_guidance.get_dp_action(state)
                    if action == dp_recommended_action:
                        guidance_agreement_count += 1
                
                # 환경에서 한 스텝 진행
                success = np.random.random() < state.p
                state = self.env.get_next_state(state, action, success)
                episode_length += 1
            
            # 최종 상태 기록
            trajectory.append({
                'a': state.a,
                'b': state.b,
                'c': state.c,
                'p': state.p,
                'remaining_slots': state.remaining_slots,
                'action': 'END'
            })
            
            # 에피소드 데이터 저장
            episode_data.append({
                'episode': episode,
                'success': self.env.get_reward(state) > 0,
                'trajectory': trajectory,
                'length': episode_length
            })
            
            # 통계 업데이트
            for action in actions_taken:
                action_counts[action] += 1
            
            # 성공 여부 확인
            if self.env.get_reward(state) > 0:
                success_count += 1
                for action in actions_taken:
                    success_rates[action].append(1.0)
            else:
                for action in actions_taken:
                    success_rates[action].append(0.0)
            
            episode_lengths.append(episode_length)
        
        # 통계 계산
        total_actions = sum(action_counts.values())
        action_frequencies = {action: count/total_actions for action, count in action_counts.items()}
        
        avg_success_rates = {}
        for action in ['A', 'B', 'C']:
            if success_rates[action]:
                avg_success_rates[action] = np.mean(success_rates[action])
            else:
                avg_success_rates[action] = 0.0
        
        # 가이던스 일치율 계산 (DP 가이던스가 적용된 경우에만)
        guidance_agreement_rate = guidance_agreement_count / guidance_applicable_count if guidance_applicable_count > 0 else 0.0
        
        return {
            'action_frequencies': action_frequencies,
            'avg_success_rates': avg_success_rates,
            'avg_episode_length': float(np.mean(episode_lengths)),
            'overall_success_rate': float(success_count / num_episodes),
            'guidance_agreement_rate': float(guidance_agreement_rate),
            'guidance_applicable_rate': float(guidance_applicable_count / total_actions) if total_actions > 0 else 0.0,
            'episode_data': episode_data
        }
    
    def save_results(self, results: Dict[str, Union[Dict[str, float], float]], save_path: str):
        """결과 저장"""
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 결과를 npz 파일로 저장
        action_freqs = results['action_frequencies']  # type: Dict[str, float]
        success_rates = results['avg_success_rates']  # type: Dict[str, float]
        
        np.savez(
            save_path,
            action_frequencies=np.array(list(action_freqs.values())),
            avg_success_rates=np.array(list(success_rates.values())),
            avg_episode_length=np.array([results['avg_episode_length']]),
            overall_success_rate=np.array([results['overall_success_rate']]),
            guidance_agreement_rate=np.array([results['guidance_agreement_rate']]),
            guidance_applicable_rate=np.array([results['guidance_applicable_rate']])
        )
        
        logging.info(f"하이브리드 결과 저장 완료: {save_path}")
    
    def get_guidance_statistics(self) -> Dict[str, float]:
        """가이던스 사용 통계 분석"""
        # 시뮬레이션에서의 가이던스 사용률
        simulation_guidance_rate = 0.0
        if self.total_simulation_steps > 0:
            simulation_guidance_rate = self.simulation_guidance_count / self.total_simulation_steps
        
        # MCTS 탐색에서의 가이던스 사용률
        mcts_guidance_rate = 0.0
        if self.total_simulations > 0:
            mcts_guidance_rate = self.guidance_usage_count / self.total_simulations
        
        return {
            'simulation_guidance_rate': simulation_guidance_rate,
            'mcts_guidance_rate': mcts_guidance_rate,
            'total_simulation_steps': self.total_simulation_steps,
            'simulation_guidance_count': self.simulation_guidance_count,
            'total_simulations': self.total_simulations,
            'guidance_usage_count': self.guidance_usage_count
        }
    
    def print_guidance_analysis(self):
        """가이던스 분석 결과 출력"""
        stats = self.get_guidance_statistics()
        
        print("\n=== DP 가이던스 분석 결과 ===")
        print(f"전체 시뮬레이션 스텝 중 {stats['simulation_guidance_rate']*100:.1f}%가 가이던스를 받아 진행")
        print(f"전체 MCTS 탐색 중 {stats['mcts_guidance_rate']*100:.1f}%에서 선택 과정에 가이던스가 개입")
        print(f"\n상세 통계:")
        print(f"  - 전체 시뮬레이션 스텝: {stats['total_simulation_steps']:,}")
        print(f"  - 가이던스 사용 스텝: {stats['simulation_guidance_count']:,}")
        print(f"  - 전체 MCTS 시뮬레이션: {stats['total_simulations']:,}")
        print(f"  - 가이던스 사용 시뮬레이션: {stats['guidance_usage_count']:,}")
        print("=" * 40) 