## Stage 2 — Monte Carlo Tree Search (MCTS)

### 개요
- 표준 MCTS 파이프라인(Selection → Expansion → Simulation → Backpropagation)으로 다음 행동을 탐색
- 루트에서 `max_simulations`만큼 시뮬레이션 후 방문수가 가장 많은 행동을 선택

### 주요 파일
- `mcts.py`
  - `Node`: 방문수/가치합, 자식 노드, 미시도 액션 관리
  - `MCTS`: `search`, `_select`, `_expand`, `_simulate`, `_backpropagate`, `_best_child(UCB1)`
- `run_mcts.py`
  - 설정 로드(`config/mcts_config.yaml`), 시드/로깅 설정, 다중 에피소드 실행 및 통계 집계
  - 결과 시각화: `common.utils.plot_mcts_results()`

### 설정
- `config/mcts_config.yaml`
  - `c_puct=5.0`, `max_simulations=2000`, `n_episodes=100`
  - 로그 경로: `logs/mcts.log`

### 실행 방법
```bash
python stage2_mcts/run_mcts.py
```

### 출력물
- `outputs/figures/mcts/` 
  - `cumulative_success_rate*.png`, `rolling_success_rate*.png`
  - `episode_lengths*.png`, `success_fail_comparison*.png`
  - `action_selection_frequency*.png`


