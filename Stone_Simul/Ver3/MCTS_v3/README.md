## MCTS_Ver3

### 파일 구성

```
MCTS_Ver3\
  - agent.py
  - config.py
  - environment.py
  - train.py
```

### 핵심 개념

- **Config 기반 설정**: 클래스 `Config`에 모든 상수 집약(시도 횟수, 확률 범위, 보상/종료 조건, MCTS 파라미터)
- **상태 모델링**: `GameState` 클래스로 상태 보존 및 `copy()` 제공
- **환경 전이**: `GameEnvironment.step(category)`는 상태 사본을 갱신해 종료 시에만 보상(1.0/0.0)을 반환하고 내부 상태를 최신화
- **종료/보상 함수 분리**: `is_terminal_state`, `get_reward`로 규칙을 명확히 분리
- **MCTS 구조**: Selection→Expansion→Simulation(rollout)→Backprop. UCB1 사용, 탐색 상수 `EXPLORATION_CONSTANT`

### 모듈별 요약

- **config.py (Config)**
  - MAX_ATTEMPTS=10, START_SUCCESS_PROB=0.75, SUCCESS_PROB_STEP=0.1, MIN/MAX=0.25/0.75
  - 승리: A/B ≥ 7, C ≤ 4. 실패: A 실패 ≥ 4, B 실패 ≥ 4, C 성공 ≥ 5
  - MCTS_SIMULATIONS=1000, EXPLORATION_CONSTANT=1.4, ROLLOUT_DEPTH=10, SEED=42
- **environment.py**
  - `GameState` 정의/복사, `is_terminal_state`, `get_reward` 구현
  - `GameEnvironment.step`: 선택 카테고리 시도 차감→성공/실패에 따라 확률 조정→종료/보상 계산→내부 상태 갱신
- **agent.py**
  - `MCTSNode`: action→child 매핑, 방문수/가치 누적
  - `best_child`: UCB1 점수로 자식 선택
  - `simulate_step`: 환경 독립적 상태 전이(사본 기반)
  - `rollout`: 무작위 정책으로 `ROLLOUT_DEPTH`까지 시뮬레이션 후 보상 평가
  - `search`: N회 시뮬레이션 후 최적 액션과 해당 자식 노드 반환
- **train.py**
  - 에피소드 루프에서 `search`로 액션 선택→실제 환경에 적용→루트 업데이트→종료 시 결과 로그




