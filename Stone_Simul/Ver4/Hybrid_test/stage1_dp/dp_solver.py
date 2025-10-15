import numpy as np
from typing import Tuple, Dict, List
import logging
from common.env import LostArkStoneEnv, State
from common.utils import plot_value_curve, plot_value_heatmap, plot_policy_heatmap, plot_value_vs_probability, plot_value_vs_remaining_slots, plot_state_transition, plot_multi_c_value_heatmaps
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
        
        # 가치 함수와 정책 초기화
        self.V = np.zeros((
            self.state_space['a'],
            self.state_space['b'],
            self.state_space['c'],
            self.state_space['p'],
            self.state_space['remaining_slots']
        ))
        self.policy = np.zeros_like(self.V, dtype=int)
        
        # 행동 공간
        self.actions = ['A', 'B', 'C']
        
    def _get_state_index(self, state: State) -> Tuple[int, int, int, int, int]:
        """상태를 인덱스로 변환"""
        # 각 차원의 범위를 체크하고 조정
        a = min(max(0, state.a), self.state_space['a'] - 1)
        b = min(max(0, state.b), self.state_space['b'] - 1)
        c = min(max(0, state.c), self.state_space['c'] - 1)
        p_idx = min(max(0, int((state.p - self.env.min_prob) / self.env.prob_change)), 
                   self.state_space['p'] - 1)
        slots = min(max(0, state.remaining_slots), self.state_space['remaining_slots'] - 1)
        
        return (a, b, c, p_idx, slots)
    
    def _get_next_state(self, state: State, action: str, success: bool) -> State:
        """다음 상태 계산"""
        next_state = State(
            a=state.a + (1 if action == 'A' and success else 0),
            b=state.b + (1 if action == 'B' and success else 0),
            c=state.c + (1 if action == 'C' and success else 0),
            p=max(self.env.min_prob, min(self.env.max_prob, 
                state.p - self.env.prob_change if success else state.p + self.env.prob_change)),
            remaining_slots=state.remaining_slots - 1
        )
        return next_state
    
    def value_iteration(self) -> List[float]:
        """Value Iteration 알고리즘 실행"""
        deltas = []
        
        # 1. 터미널 상태 초기화
        for a in range(self.state_space['a']):
            for b in range(self.state_space['b']):
                for c in range(self.state_space['c']):
                    for p_idx in range(self.state_space['p']):
                        for slots in range(self.state_space['remaining_slots']):
                            state = State(
                                a=a, b=b, c=c,
                                p=self.env.min_prob + p_idx * self.env.prob_change,
                                remaining_slots=slots
                            )
                            if self.env.is_terminal(state):
                                self.V[a, b, c, p_idx, slots] = self.env.get_reward(state)
        
        # 2. Value Iteration
        for iteration in range(self.max_iterations):
            delta = 0
            
            # 모든 상태에 대해 반복
            for a in range(self.state_space['a']):
                for b in range(self.state_space['b']):
                    for c in range(self.state_space['c']):
                        for p_idx in range(self.state_space['p']):
                            for slots in range(self.state_space['remaining_slots']):
                                state = State(
                                    a=a, b=b, c=c,
                                    p=self.env.min_prob + p_idx * self.env.prob_change,
                                    remaining_slots=slots
                                )
                                
                                if self.env.is_terminal(state):
                                    continue
                                
                                # 현재 상태의 가치 저장
                                v = self.V[a, b, c, p_idx, slots]
                                
                                # 모든 행동에 대한 가치 계산
                                action_values = []
                                for action in self.actions:
                                    # 성공/실패에 따른 다음 상태와 보상 계산
                                    success_state = self._get_next_state(state, action, True)
                                    fail_state = self._get_next_state(state, action, False)
                                    
                                    success_idx = self._get_state_index(success_state)
                                    fail_idx = self._get_state_index(fail_state)
                                    
                                    # 성공 시
                                    if self.env.is_terminal(success_state):
                                        success_value = self.env.get_reward(success_state)
                                    else:
                                        success_value = self.env.get_reward(success_state) + self.gamma * self.V[success_idx]
                                    # 실패 시
                                    if self.env.is_terminal(fail_state):
                                        fail_value = self.env.get_reward(fail_state)
                                    else:
                                        fail_value = self.env.get_reward(fail_state) + self.gamma * self.V[fail_idx]
                                    
                                    value = state.p * success_value + (1 - state.p) * fail_value
                                    action_values.append(value)
                                
                                # 최적 행동과 가치 업데이트
                                best_action = np.argmax(action_values)
                                self.policy[a, b, c, p_idx, slots] = best_action
                                self.V[a, b, c, p_idx, slots] = action_values[best_action]
                                
                                # 변화량 계산
                                delta = max(delta, abs(v - self.V[a, b, c, p_idx, slots]))
            
            deltas.append(delta)
            logging.info(f"Iteration {iteration + 1}, Delta: {delta}")
            
            if delta < self.theta:
                logging.info(f"Value Iteration converged at iteration {iteration + 1}")
                break
        
        return deltas
    
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
        
        # 1. 수렴 곡선 플롯
        plot_value_curve(
            self.deltas,
            self.config['value_curve_plot']
        )
        
        # 2. 여러 c 값에 대한 가치 함수 히트맵 (p=0.75, slots=5)
        p_idx = int((0.75 - self.env.min_prob) / self.env.prob_change)
        c_values = [0, 2, 4]  # 비교할 c 값들
        plot_multi_c_value_heatmaps(
            self.V,
            c_values,
            p_idx,
            5,
            'outputs/figures/dp_multi_c_value_heatmap.png'
        )
        
        # 3. 정책 히트맵 (c=0일 때)
        plot_policy_heatmap(
            self.policy[:, :, 0, p_idx, 5],
            'A Success Count',
            'B Success Count',
            'outputs/figures/dp_policy_heatmap.png'
        )
        
        # 4. 성공 확률에 따른 가치 변화
        probs = np.linspace(self.env.min_prob, self.env.max_prob, self.state_space['p'])
        states_to_plot = [
            (0, 0, 0, 30),  # 초기 상태
            (3, 3, 1, 23),  # 중간 상태 1
            (6, 6, 2, 16),  # 중간 상태 2
            (5, 5, 4, 16)   # 중간 상태 3
        ]
        plot_value_vs_probability(
            self.V,
            probs,
            states_to_plot,
            'outputs/figures/dp_value_vs_probability.png'
        )
        
        # 5. 남은 시도 횟수에 따른 가치 변화
        plot_value_vs_remaining_slots(
            self.V[0, 0, 0, :, :],  # a=0, b=0, c=0, 모든 p, 모든 slots
            'outputs/figures/dp_value_vs_slots.png'
        )
        
        # 6. 상태 전이 시각화
        initial_state = self.env.get_initial_state()
        plot_state_transition(
            self.env,
            self.policy,
            initial_state,
            self._get_state_index,
            max_steps=10,
            save_path='outputs/figures/dp_state_transition.png'
        ) 