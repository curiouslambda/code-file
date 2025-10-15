# Stage2_mcts

### 목적
- 표준 MCTS 탐색 구현 및 성능 로그/시각화

### 주요 파일
- `mcts.py`: Node 구조, Selection/Expansion/Simulation/Backpropagation, UCB1, 최다 방문 행동 선택
- `run_mcts.py`: 설정/시드/로깅, 다수 에피소드 실행, 성공률/스텝 통계, 결과 그래프 저장

### 실행
```bash
python stage2_mcts/run_mcts.py
```

### 산출물
- `outputs/figures/mcts/` 하위 그래프: 누적/롤링 성공률, 스텝 분포, 성공/실패 비교, 루트 액션 빈도.

