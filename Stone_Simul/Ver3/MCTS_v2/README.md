## MCTS_Ver2

### 파일 구성

```
MCTS_Ver2\
  - agent.py
  - config.py
  - environment.py
  - train.py
```

### 핵심 개념

- **상태 표현**: 튜플 (success_prob, successes, failures, attempts_left)
- **성공확률 전이**: 성공 시 −0.1, 실패 시 +0.1 (범위 [0.25, 0.75])
- **종료 조건**: C 성공 ≥ 5, A/B 실패 ≥ 4, 모든 시도 소진
- **보상 체계**: 진행 중 0, 종료 시 승리 1, 패배 −1
- **MCTS 개요**: UCB1 선택/확장/시뮬레이션/백업 구현이나, 액션/노드 반환과 선택 로직에 잠재적 결함

### 모듈별 요약

- **config.py**
  - MAX_ATTEMPTS_PER_CATEGORY=10, CATEGORIES=['A','B','C']
  - INITIAL_SUCCESS_PROB=0.75, SUCCESS_PROB_CHANGE=0.1, MIN/MAX=0.25/0.75
  - 승리 조건: A/B ≥ 7, C ≤ 4; 실패 조건: A/B 실패 ≥ 4, C 성공 ≥ 5
- **environment.py (Environment)**
  - step(category): 시도 차감, 성공/실패에 따라 확률 조정, done/보상 계산
  - calculate_reward(done): 종료 시 1 또는 −1, 진행 중 0
- **agent.py (MCTSAgent/Node)**
  - best_child: UCB1
  - selection: `env_copy` 사용하지만 `env_copy.step(list(node.state[3].keys())[0])`로 임의 키 사용
  - expansion/simulation: `category` 변수 의존 및 노드/액션 혼용으로 논리 오류 가능
  - 반환값: `root.best_child(c_param=0).parent`로 액션이 아닌 노드를 반환
- **train.py**
  - `select_action` 반환 노드에서 첫 가용 카테고리를 추출해 `env.step` 수행



