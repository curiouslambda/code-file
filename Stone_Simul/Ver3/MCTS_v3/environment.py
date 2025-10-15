# environment.py

import random
from config import Config

class GameState:
    """
    게임의 상태를 표현하는 클래스
    """
    def __init__(self,
                 success_prob=Config.START_SUCCESS_PROB,
                 attempts_left=None,
                 successes=None,
                 failures=None,
                 done=False):
        if attempts_left is None:
            attempts_left = {"A": Config.MAX_ATTEMPTS,
                             "B": Config.MAX_ATTEMPTS,
                             "C": Config.MAX_ATTEMPTS}
        if successes is None:
            successes = {"A": 0, "B": 0, "C": 0}
        if failures is None:
            failures = {"A": 0, "B": 0, "C": 0}

        self.success_prob = success_prob
        self.attempts_left = attempts_left
        self.successes = successes
        self.failures = failures
        self.done = done

    def copy(self):
        """
        현재 상태를 복사해서 반환
        """
        return GameState(
            success_prob=self.success_prob,
            attempts_left=self.attempts_left.copy(),
            successes=self.successes.copy(),
            failures=self.failures.copy(),
            done=self.done
        )

def is_terminal_state(state: GameState) -> bool:
    """
    게임 종료 조건을 판별
    """
    # 이미 done으로 표시되어 있으면 바로 종료
    if state.done:
        return True

    # 카테고리 A, B, C의 남은 시도가 모두 0인지 (총 30스텝 소진)
    if sum(state.attempts_left.values()) == 0:
        return True

    # C의 성공횟수가 5 이상이면 실패로 종료
    if state.successes["C"] >= Config.MAX_SUCCESS_C_FAIL:
        return True

    # A 혹은 B에서 실패가 4 이상이면 실패로 종료
    if state.failures["A"] >= Config.MAX_FAILURE_A or state.failures["B"] >= Config.MAX_FAILURE_B:
        return True

    return False

def get_reward(state: GameState) -> float:
    """
    최종 스코어(또는 보상)를 계산
    - A, B는 성공이 7 이상 필요
    - C는 성공이 4 이하
    """
    # 게임이 최종 종료된 시점에서 평가
    # (단순히 승패를 1, 0으로 처리)
    if (state.successes["A"] >= Config.REQUIRED_SUCCESS_A and
        state.successes["B"] >= Config.REQUIRED_SUCCESS_B and
        state.successes["C"] <= Config.MAX_SUCCESS_C):
        return 1.0  # 승리
    else:
        return 0.0  # 패배

class GameEnvironment:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.state = GameState()

    def reset(self):
        """
        게임을 초기화하고 초기 상태를 반환
        """
        self.state = GameState()
        return self.state

    def step(self, category: str):
        """
        주어진 category(A, B, C)에 대한 행동을 취했을 때
        다음 상태, 보상, 종료 여부를 반환
        """
        # 상태 복사 후 업데이트
        next_state = self.state.copy()
        # print(next_state)

        # 남은 횟수가 없으면 그대로 반환 (실제로는 불가능한 액션 처리)
        if next_state.attempts_left[category] <= 0:
            return next_state, 0.0, True

        # 시도 횟수 소모
        next_state.attempts_left[category] -= 1

        # 성공 / 실패 판단
        # 모든 카테고리는 동일한 성공 확률
        success = random.random() < next_state.success_prob
        if success:
            next_state.successes[category] += 1
            # 성공 시 성공확률 감소
            next_state.success_prob = max(Config.MIN_SUCCESS_PROB,
                                          next_state.success_prob - Config.SUCCESS_PROB_STEP)
        else:
            next_state.failures[category] += 1
            # 실패 시 성공확률 증가
            next_state.success_prob = min(Config.MAX_SUCCESS_PROB,
                                          next_state.success_prob + Config.SUCCESS_PROB_STEP)

        # 종료 조건 검사
        next_state.done = is_terminal_state(next_state)

        # 보상 계산 (여기서는 최종 종료 시점에만 보상을 주는 구조로 예시)
        if next_state.done:
            reward = get_reward(next_state)
        else:
            reward = 0.0

        self.state = next_state  # 환경 내부 상태 갱신
        return next_state, reward, next_state.done
