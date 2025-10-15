## MCTS_Ver1

### 파일 구성

```
MCTS_Ver1\
  - agent.py
  - config.py
  - environment.py
  - main.py
  - train.py
```

### 핵심 개념

- **성공확률 전이**: 성공 시 성공확률 −0.1, 실패 시 +0.1 (범위 [0.25, 0.75])
- **종료 조건**: C 성공 ≥ 5, A/B 실패 ≥ 4, 모든 시도 소진(sum==0), 또는 카테고리 중 하나라도 0(조기 종료 로그 출력)
- **보상 체계**: 진행 중에는 A/B 선택 시 +1, C 선택 시 −1. 종료 시 A/B ≥ 7, C ≤ 4이면 +100, 아니면 −100
- **MCTS 개요**: UCB1로 자식 선택, 무방문 자식 무한 가중치 처리. 확장/시뮬레이션 시 실제 환경을 변형할 수 있는 버그 가능성 존재

### 모듈별 요약

- **config.py**
  - GAME_CONFIG: 초기 성공확률 0.75, 최소/최대 0.25/0.75, 카테고리 A/B/C, 각 10회 시도
  - 실패 조건: C 성공 ≥ 5, A/B 실패 ≥ 4
  - 보상: 승리 +100, 패배 −100, 스텝 보상(A/B +1, C −1)
  - MCTS_CONFIG: simulations=3, exploration_weight=1.0
- **environment.py**
  - 상태: success_prob, remaining_steps, success, failure
  - step(category): 시도 차감 후 성공/실패에 따라 확률 조정, 종료/보상 계산
  - check_done(): 실패/성공 한계, 합계 0, 또는 하나라도 0일 때 종료
- **agent.py**
  - MCTSNode: UCB1 기반 best_child, is_fully_expanded는 남은 시도>0인 액션 기준
  - _expand: 무작위 유효 액션으로 env.step 수행 후 자식 생성.(원본 env 사용)
  - _simulate: env.clone()를 만들지만 self.environment.step을 호출해 원본 env를 변형할 가능성 존재재
  - _backpropagate: 방문/가치 누적
- **train.py**
  - 1 에피소드 루프로 상태 초기화→에이전트 액션 선택→env.step→누적 보상/종료
- **main.py**
  - `train()` 실행 엔트리


