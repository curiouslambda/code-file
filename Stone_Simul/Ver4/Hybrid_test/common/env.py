import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class State:
    a: int  # A 카테고리 성공 횟수
    b: int  # B 카테고리 성공 횟수
    c: int  # C 카테고리 성공 횟수
    p: float  # 현재 성공 확률
    remaining_slots: int  # 남은 시도 횟수

class LostArkStoneEnv:
    def __init__(self):
        self.initial_prob = 0.75
        self.min_prob = 0.25
        self.max_prob = 0.75
        self.prob_change = 0.10
        self.max_slots = 10  # 각 카테고리당 최대 시도 횟수
        self.win_conditions = {
            'A': 7,  # A >= 7
            'B': 7,  # B >= 7
            'C': 4   # C <= 4
        }

    def get_initial_state(self) -> State:
        return State(a=0, b=0, c=0, p=self.initial_prob, remaining_slots=self.max_slots * 3)

    def is_terminal(self, state: State) -> bool:
        # 승리 조건 확인
        win_condition = (
            state.a >= self.win_conditions['A'] and
            state.b >= self.win_conditions['B'] and
            state.c <= self.win_conditions['C']
        )
        # 모든 슬롯 소진
        no_slots = state.remaining_slots == 0
        return win_condition or no_slots

    def get_reward(self, state: State) -> float:
        if not self.is_terminal(state):
            return 0.0
        
        # 승리 조건 만족 시 보상 1.0
        if (state.a >= self.win_conditions['A'] and
            state.b >= self.win_conditions['B'] and
            state.c <= self.win_conditions['C']):
            return 1.0
        return 0.0

    def step(self, state: State, action: str) -> Tuple[State, float, bool]:
        """
        action: 'A', 'B', 'C' 중 하나
        """
        if state.remaining_slots == 0:
            return state, 0.0, True

        # 성공/실패 결정
        success = np.random.random() < state.p
        
        # 다음 상태 계산
        next_state = State(
            a=state.a + (1 if action == 'A' and success else 0),
            b=state.b + (1 if action == 'B' and success else 0),
            c=state.c + (1 if action == 'C' and success else 0),
            p=max(self.min_prob, min(self.max_prob, 
                state.p - self.prob_change if success else state.p + self.prob_change)),
            remaining_slots=state.remaining_slots - 1
        )

        # 보상 계산
        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)

        return next_state, reward, done

    def get_state_space_size(self) -> Dict[str, int]:
        return {
            'a': self.max_slots + 1,
            'b': self.max_slots + 1,
            'c': self.max_slots + 1,
            'p': int((self.max_prob - self.min_prob) / self.prob_change) + 1,
            'remaining_slots': self.max_slots * 3 + 1
        }
        
    def get_next_state(self, state: State, action: str, success: bool) -> State:
        """다음 상태 계산"""
        return State(
            a=state.a + (1 if action == 'A' and success else 0),
            b=state.b + (1 if action == 'B' and success else 0),
            c=state.c + (1 if action == 'C' and success else 0),
            p=max(self.min_prob, min(self.max_prob, 
                state.p - self.prob_change if success else state.p + self.prob_change)),
            remaining_slots=state.remaining_slots - 1
        ) 