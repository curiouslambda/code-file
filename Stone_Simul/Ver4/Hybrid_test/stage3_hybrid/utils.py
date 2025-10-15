"""
DP-MCTS 연동 유틸리티 함수들
DP 결과를 MCTS 트리 탐색 초반에 가이던스로 활용하는 헬퍼 함수들
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from common.env import LostArkStoneEnv, State

class DPGuidanceUtils:
    """DP 결과를 MCTS 가이던스로 변환하는 유틸리티 클래스"""
    
    def __init__(self, env: LostArkStoneEnv, dp_values_path: str):
        """
        Args:
            env: LostArkStoneEnv 인스턴스
            dp_values_path: DP 결과가 저장된 .npz 파일 경로
        """
        self.env = env
        self.dp_values_path = dp_values_path
        self.dp_values = None
        self.dp_policy = None
        self.state_space = env.get_state_space_size()
        self._load_dp_results()
    
    def _load_dp_results(self):
        """DP 결과 로드"""
        try:
            data = np.load(self.dp_values_path)
            self.dp_values = data['V']
            self.dp_policy = data['policy']
            logging.info(f"DP 결과 로드 완료: {self.dp_values_path}")
        except Exception as e:
            logging.error(f"DP 결과 로드 실패: {e}")
            self.dp_values = None
            self.dp_policy = None
    
    def _get_state_index(self, state: State) -> Tuple[int, int, int, int, int]:
        """상태를 인덱스로 변환"""
        a = min(max(0, state.a), self.state_space['a'] - 1)
        b = min(max(0, state.b), self.state_space['b'] - 1)
        c = min(max(0, state.c), self.state_space['c'] - 1)
        p_idx = min(max(0, int((state.p - self.env.min_prob) / self.env.prob_change)), 
                   self.state_space['p'] - 1)
        slots = min(max(0, state.remaining_slots), self.state_space['remaining_slots'] - 1)
        
        return (a, b, c, p_idx, slots)
    
    def get_dp_value(self, state: State) -> Optional[float]:
        """주어진 상태의 DP 가치 함수 값 반환"""
        if self.dp_values is None:
            return None
        
        try:
            idx = self._get_state_index(state)
            return float(self.dp_values[idx])
        except (IndexError, ValueError):
            return None
    
    def get_dp_policy(self, state: State) -> Optional[int]:
        """주어진 상태의 DP 정책 반환 (0: A, 1: B, 2: C)"""
        if self.dp_policy is None:
            return None
        
        try:
            idx = self._get_state_index(state)
            return int(self.dp_policy[idx])
        except (IndexError, ValueError):
            return None
    
    def get_dp_action(self, state: State) -> Optional[str]:
        """주어진 상태의 DP 권장 행동 반환 ('A', 'B', 'C')"""
        policy = self.get_dp_policy(state)
        if policy is None:
            return None
        
        actions = ['A', 'B', 'C']
        return actions[policy]
    
    def get_action_priorities(self, state: State, temperature: float = 1.0) -> Dict[str, float]:
        """
        DP 가치 함수를 기반으로 행동 우선순위 계산
        
        Args:
            state: 현재 상태
            temperature: 확률 분포의 온도 (높을수록 균등, 낮을수록 집중)
        
        Returns:
            각 행동의 우선순위 (확률 분포)
        """
        if self.dp_values is None:
            # DP 결과가 없으면 균등 분포
            return {'A': 1/3, 'B': 1/3, 'C': 1/3}
        
        actions = ['A', 'B', 'C']
        action_values = []
        
        for action in actions:
            # 성공/실패에 따른 다음 상태들의 가치 계산
            success_state = self.env.get_next_state(state, action, True)
            fail_state = self.env.get_next_state(state, action, False)
            
            success_value = self.get_dp_value(success_state) or 0.0
            fail_value = self.get_dp_value(fail_state) or 0.0
            
            # 기대 가치 계산
            expected_value = state.p * success_value + (1 - state.p) * fail_value
            action_values.append(expected_value)
        
        # 온도 기반 확률 분포 계산
        if temperature == 0:
            # 최대값만 선택
            max_idx = np.argmax(action_values)
            priorities = [0.0] * 3
            priorities[max_idx] = 1.0
        else:
            # 소프트맥스 적용
            logits = np.array(action_values) / temperature
            exp_logits = np.exp(logits - np.max(logits))  # 수치 안정성
            priorities = exp_logits / np.sum(exp_logits)
        
        return dict(zip(actions, priorities))
    
    def get_guided_action_selection(self, state: State, exploration_weight: float = 0.1) -> str:
        """
        DP 가이던스와 탐색을 결합한 행동 선택
        
        Args:
            state: 현재 상태
            exploration_weight: 탐색 가중치 (0: 순수 DP, 1: 순수 랜덤)
        
        Returns:
            선택된 행동
        """
        if self.dp_values is None or np.random.random() < exploration_weight:
            # DP 결과가 없거나 탐색 모드일 때 랜덤 선택
            return np.random.choice(['A', 'B', 'C'])
        
        # DP 기반 확률 분포 계산
        priorities = self.get_action_priorities(state, temperature=0.5)
        
        # 확률 분포에서 샘플링
        actions = list(priorities.keys())
        probs = list(priorities.values())
        
        return np.random.choice(actions, p=probs)
    
    def get_state_guidance_score(self, state: State) -> float:
        """
        현재 상태에 대한 DP 가이던스 점수 계산
        높은 값일수록 DP가 신뢰할 만한 상태
        
        Returns:
            가이던스 점수 (0.0 ~ 1.0)
        """
        if self.dp_values is None:
            return 0.0
        
        # 1. DP 가치 함수 값이 있는지 확인
        dp_value = self.get_dp_value(state)
        if dp_value is None:
            return 0.0
        
        # 2. 상태가 DP 학습 범위 내에 있는지 확인
        state_space = self.state_space
        if (state.a >= state_space['a'] or 
            state.b >= state_space['b'] or 
            state.c >= state_space['c'] or
            state.remaining_slots >= state_space['remaining_slots']):
            return 0.0
        
        # 3. 확률이 범위 내에 있는지 확인
        p_idx = int((state.p - self.env.min_prob) / self.env.prob_change)
        if p_idx < 0 or p_idx >= state_space['p']:
            return 0.0
        
        # 4. 가치 함수 값의 신뢰도 계산 (높은 가치일수록 신뢰도 높음)
        confidence = min(1.0, dp_value / 1.0)  # 최대 가치 1.0 기준
        
        return confidence
    
    def get_hybrid_ucb_score(self, node, action: str, c_puct: float, 
                           guidance_weight: float = 0.3) -> float:
        """
        DP 가이던스를 결합한 UCB 점수 계산
        
        Args:
            node: MCTS 노드
            action: 평가할 행동
            c_puct: UCB 탐색 파라미터
            guidance_weight: DP 가이던스 가중치
        
        Returns:
            UCB 점수
        """
        if action not in node.children:
            return float('inf')  # 방문하지 않은 행동은 무한대 점수
        
        child = node.children[action]
        
        # 기본 UCB 점수 계산
        exploitation = child.value_sum / (child.visit_count + 1e-8)
        exploration = c_puct * np.sqrt(node.visit_count) / (child.visit_count + 1e-8)
        ucb_score = exploitation + exploration
        
        # DP 가이던스 점수 계산
        guidance_score = 0.0
        if self.dp_values is not None:
            dp_action = self.get_dp_action(node.state)
            if dp_action == action:
                guidance_score = self.get_state_guidance_score(node.state)
        
        # 가이던스 가중 평균
        final_score = (1 - guidance_weight) * ucb_score + guidance_weight * guidance_score
        
        return final_score
    
    def is_guidance_applicable(self, state: State, remaining_slots_threshold: int = 5) -> bool:
        """
        현재 상태에 DP 가이던스를 적용할 수 있는지 확인
        
        Args:
            state: 현재 상태
            remaining_slots_threshold: 가이던스 적용 임계값 (남은 슬롯 수)
        
        Returns:
            가이던스 적용 가능 여부
        """
        # 1. DP 결과가 있는지 확인
        if self.dp_values is None:
            return False
        
        # 2. 남은 슬롯이 충분한지 확인 (초반에만 가이던스 적용)
        if state.remaining_slots > remaining_slots_threshold:
            return False
        
        # 3. 상태가 DP 학습 범위 내에 있는지 확인
        guidance_score = self.get_state_guidance_score(state)
        return guidance_score > 0.1  # 최소 신뢰도 임계값 