# Hybrid_test/stage2_mcts

### 목적
- MCTS 탐색 실행 및 통계/시각화 강화 버전

### 주요 파일
- `mcts.py`: 표준 MCTS 구현(동률 처리 개선, 로그 탐색식 log(N+1) 사용)
- `run_mcts.py`: 첫 액션 기록 포함, 에피소드 통계 수집, 그래프 생성

### 실행
```bash
python Hybrid_test/stage2_mcts/run_mcts.py --config Hybrid_test/config/mcts_config.yaml
```

### 산출물
- `outputs/figures/mcts/` 하위 여러 그래프
