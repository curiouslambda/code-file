import numpy as np
from typing import Tuple, Dict, List
import logging
import hashlib
import random
from common.env import LostArkStoneEnv, State
from common.utils import (plot_value_curve, plot_policy_margin_heatmap, plot_policy_decision_map,
                           plot_value_heatmap, plot_policy_heatmap, plot_value_vs_probability,
                           plot_value_vs_remaining_slots, plot_state_transition, plot_multi_c_value_heatmaps)
import os

class DPSolver:
    def __init__(self, env: LostArkStoneEnv, config: Dict):
        self.env = env
        self.config = config
        self.gamma = float(config['gamma'])
        self.theta = float(config['theta'])
        self.max_iterations = int(config['max_iterations'])
        
        # 상태 공간 크기
        self.state_space = env.get_state_space_size()

        # 환경 상수 저장 (매직넘버 제거)
        self.max_attempts_per_category = env.max_slots
        self.total_max_attempts = 3 * env.max_slots

        # 가치 함수와 정책 초기화 - 새로운 상태 벡터: (a, b, c, p_idx, a_try, b_try, c_try)
        self.V = np.zeros((
            self.state_space['a'],
            self.state_space['b'],
            self.state_space['c'],
            self.state_space['p'],
            self.state_space['a_try'],
            self.state_space['b_try'],
            self.state_space['c_try']
        ))
        self.policy = np.zeros_like(self.V, dtype=int)
        
        # 행동 공간
        self.actions = ['A', 'B', 'C']
        
    def _get_state_index(self, state: State) -> Tuple[int, int, int, int, int, int, int]:
        """상태를 인덱스로 변환"""
        # 각 차원의 범위를 체크하고 조정
        a = min(max(0, state.a), self.state_space['a'] - 1)
        b = min(max(0, state.b), self.state_space['b'] - 1)
        c = min(max(0, state.c), self.state_space['c'] - 1)
        p_idx = min(max(0, round((state.p - self.env.min_prob) / self.env.prob_change)),
                   self.state_space['p'] - 1)
        a_try = min(max(0, state.a_try), self.state_space['a_try'] - 1)
        b_try = min(max(0, state.b_try), self.state_space['b_try'] - 1)
        c_try = min(max(0, state.c_try), self.state_space['c_try'] - 1)

        return (a, b, c, p_idx, a_try, b_try, c_try)
    
    def _get_next_state(self, state: State, action: str, success: bool) -> State:
        """다음 상태 계산"""
        next_state = State(
            a=state.a + (1 if action == 'A' and success else 0),
            b=state.b + (1 if action == 'B' and success else 0),
            c=state.c + (1 if action == 'C' and success else 0),
            p=max(self.env.min_prob, min(self.env.max_prob,
                state.p - self.env.prob_change if success else state.p + self.env.prob_change)),
            a_try=state.a_try - (1 if action == 'A' else 0),
            b_try=state.b_try - (1 if action == 'B' else 0),
            c_try=state.c_try - (1 if action == 'C' else 0)
        )
        return next_state
    
    def _compute_state_value(self, a: int, b: int, c: int, p_idx: int,
                           a_try: int, b_try: int, c_try: int) -> float:
        """역방향 DP를 위한 상태 가치 계산 (재귀적, 메모이제이션)"""
        state_key = (a, b, c, p_idx, a_try, b_try, c_try)

        # 이미 계산된 상태라면 캐시에서 반환
        if hasattr(self, '_value_cache') and state_key in self._value_cache:
            return self._value_cache[state_key]

        # 캐시 초기화 (첫 호출시)
        if not hasattr(self, '_value_cache'):
            self._value_cache = {}

        current_p = self.env.min_prob + p_idx * self.env.prob_change
        state = State(a=a, b=b, c=c, p=current_p,
                     a_try=a_try, b_try=b_try, c_try=c_try)

        # 터미널 상태인 경우
        if self.env.is_terminal(state):
            value = self.env.get_reward(state)
            self._value_cache[state_key] = value
            return value

        # 가능한 행동들에 대한 가치 계산
        valid_actions = self.env.get_valid_actions(state)
        action_values = []
        valid_action_indices = []

        for action in valid_actions:
            # 성공/실패에 따른 다음 상태 계산
            success_state = self._get_next_state(state, action, True)
            fail_state = self._get_next_state(state, action, False)

            # 다음 상태의 p_idx 계산
            success_p_idx = min(max(0, round((success_state.p - self.env.min_prob) / self.env.prob_change)),
                               self.state_space['p'] - 1)
            fail_p_idx = min(max(0, round((fail_state.p - self.env.min_prob) / self.env.prob_change)),
                           self.state_space['p'] - 1)

            # 다음 상태의 가치 계산 (재귀 호출)
            success_value = self._compute_state_value(
                success_state.a, success_state.b, success_state.c,
                success_p_idx, success_state.a_try, success_state.b_try, success_state.c_try
            )

            fail_value = self._compute_state_value(
                fail_state.a, fail_state.b, fail_state.c,
                fail_p_idx, fail_state.a_try, fail_state.b_try, fail_state.c_try
            )

            # 보상 추가
            success_value = self.env.get_reward(success_state) + self.gamma * success_value
            fail_value = self.env.get_reward(fail_state) + self.gamma * fail_value

            # 기대 가치 계산
            value = state.p * success_value + (1 - state.p) * fail_value
            action_values.append(value)
            valid_action_indices.append(self.actions.index(action))

        if action_values:  # 가능한 액션이 있는 경우
            # 최적 행동 선택 (타이브레이크 포함)
            max_value = max(action_values)
            best_action_indices = [i for i, val in enumerate(action_values) if val == max_value]

            if len(best_action_indices) > 1:
                # 상태 기반 시드로 타이브레이크
                state_info = f"{a},{b},{c},{p_idx},{a_try},{b_try},{c_try}"
                seed = int(hashlib.md5(state_info.encode()).hexdigest(), 16) % (2**32)
                random.seed(seed)
                selected_idx = random.choice(best_action_indices)
            else:
                selected_idx = best_action_indices[0]

            best_action = valid_action_indices[selected_idx]
            optimal_value = action_values[selected_idx]

            # 결과 저장
            self.V[a, b, c, p_idx, a_try, b_try, c_try] = optimal_value
            self.policy[a, b, c, p_idx, a_try, b_try, c_try] = best_action
        else:
            # 가능한 액션이 없는 경우
            optimal_value = 0.0
            self.V[a, b, c, p_idx, a_try, b_try, c_try] = optimal_value

        # 캐시에 저장
        self._value_cache[state_key] = optimal_value
        return optimal_value

    def backward_dp(self) -> None:
        """역방향 DP 알고리즘 실행"""
        logging.info("Starting Backward DP...")

        # 캐시 초기화
        self._value_cache = {}

        # 터미널 상태들을 미리 초기화 (남은 시도 횟수가 0인 상태들)
        for a_try in range(self.state_space['a_try']):
            for b_try in range(self.state_space['b_try']):
                for c_try in range(min(self.state_space['c_try'], self.total_max_attempts - a_try - b_try)):
                    for a in range(min(self.state_space['a'], self.max_attempts_per_category - a_try)):
                        for b in range(min(self.state_space['b'], self.max_attempts_per_category - b_try)):
                            for c in range(min(self.state_space['c'], self.max_attempts_per_category - c_try)):
                                for p_idx in range(self.state_space['p']):
                                    current_p = self.env.min_prob + p_idx * self.env.prob_change
                                    state = State(a=a, b=b, c=c, p=current_p,
                                                a_try=a_try, b_try=b_try, c_try=c_try)

                                    if self.env.is_terminal(state):
                                        reward = self.env.get_reward(state)
                                        self.V[a, b, c, p_idx, a_try, b_try, c_try] = reward
                                        self._value_cache[(a, b, c, p_idx, a_try, b_try, c_try)] = reward

        # 남은 시도 횟수의 합이 큰 상태부터 작은 상태 방향으로 계산
        # (역방향: 시도 횟수가 많은 상태부터 계산)
        max_total_attempts = self.total_max_attempts

        for total_attempts in range(max_total_attempts, -1, -1):
            for a_try in range(min(self.state_space['a_try'], total_attempts + 1)):
                for b_try in range(min(self.state_space['b_try'], total_attempts - a_try + 1)):
                    c_try = total_attempts - a_try - b_try
                    if c_try < 0 or c_try >= self.state_space['c_try']:
                        continue

                    for a in range(min(self.state_space['a'], self.max_attempts_per_category - a_try)):
                        for b in range(min(self.state_space['b'], self.max_attempts_per_category - b_try)):
                            for c in range(min(self.state_space['c'], self.max_attempts_per_category - c_try)):
                                for p_idx in range(self.state_space['p']):
                                    # 터미널 상태는 이미 초기화했으므로 건너뜀
                                    current_p = self.env.min_prob + p_idx * self.env.prob_change
                                    state = State(a=a, b=b, c=c, p=current_p,
                                                a_try=a_try, b_try=b_try, c_try=c_try)

                                    if not self.env.is_terminal(state):
                                        self._compute_state_value(a, b, c, p_idx, a_try, b_try, c_try)

        logging.info("Backward DP completed")
        self.deltas = []  # 역방향 DP는 delta 계산이 의미가 없으므로 빈 리스트 반환

    def value_iteration(self) -> List[float]:
        """역방향 DP를 사용하는 래퍼 메소드"""
        self.backward_dp()
        return self.deltas
    
    def save_results(self):
        """결과 저장"""
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(self.config['value_save_path']), exist_ok=True)
        os.makedirs(os.path.dirname(self.config['value_curve_plot']), exist_ok=True)
        os.makedirs(os.path.dirname(self.config['value_heatmap_plot']), exist_ok=True)
        
        # 가치 함수와 정책 저장
        np.savez(
            self.config['value_save_path'],
            V=self.V,
            policy=self.policy
        )
        
        # 1. 벨만 잔차 수렴 곡선 (필수)
        plot_value_curve(
            self.deltas,
            self.config['value_curve_plot']
        )

        # 2. 정책 마진 히트맵 (c=2, p=0.75, a_try=5, b_try=5, c_try=5)
        p_idx = int(round((0.75 - self.env.min_prob) / self.env.prob_change))
        plot_policy_margin_heatmap(
            self.V, self.env, c=2, p_idx=p_idx,
            a_try=5, b_try=5, c_try=5,
            save_path='outputs/figures/dp_policy_margin_heatmap.png'
        )

        # 3. 정책 결정 맵 (c=2, p=0.75, a_try=5, b_try=5, c_try=5)
        plot_policy_decision_map(
            self.V, self.env, c=2, p_idx=p_idx,
            a_try=5, b_try=5, c_try=5,
            save_path='outputs/figures/dp_policy_decision_map.png'
        )
        
                # 추가 분석을 위한 여러 정책 맵들 생성
        # 다른 c 값들에 대해서도 정책 분석
        for c_val in [0, 1, 3, 4]:
            plot_policy_decision_map(
                self.V, self.env, c=c_val, p_idx=p_idx,
                a_try=5, b_try=5, c_try=5,
                save_path=f'outputs/figures/dp_policy_decision_map_c{c_val}.png'
            ) 