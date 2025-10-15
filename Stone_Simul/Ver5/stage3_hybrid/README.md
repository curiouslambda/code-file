## Stage 3 — Hybrid (DP-guided MCTS)

### 개요
- DP 가치/정책 테이블을 MCTS의 Selection/Simulation 단계에 가이던스로 결합한 하이브리드 탐색
- 남은 슬롯이 임계값 이하일 때만 DP 가이던스를 적용하여 탐색과 활용의 균형을 조절

### 주요 파일
- `hybrid_solver.py`
  - `HybridSolver`: DP 가이던스 결합 MCTS (`_select_with_guidance`, `_get_action_probabilities`, 통계 수집/저장)
  - 가이던스 사용/일치율, 시뮬레이션 스텝 등 다양한 지표 집계
- `utils.py`
  - `DPGuidanceUtils`: `outputs/dp_values.npz`에서 `V`/`policy` 로드, 상태→인덱스 변환, DP 권장행동 및 하이브리드 UCB 점수 계산
- `run_hybrid.py`
  - 실험 실행/로깅, 결과 저장(`outputs/hybrid/hybrid_results.npz`), 그래프 생성
  - `--compare` 옵션으로 DP/Random/Hybrid 성능 비교 및 그래프 저장

### 설정
- `config/hybrid_config.yaml`
  - MCTS: `c_puct=3.0`, `max_simulations=500`
  - 하이브리드: `guidance_weight`, `guidance_threshold`, `guidance_temperature`, `exploration_weight`
  - `dp_values_path`, `num_episodes`, `output_dir`, `results_save_path`

### 실행 방법
```bash
# 기본 실험
python stage3_hybrid/run_hybrid.py

# 방법 비교 (DP/Random/Hybrid)
python stage3_hybrid/run_hybrid.py --compare
```

### 출력물(예)
- `outputs/hybrid/figures/` 
  - 행동 빈도/성공률/가이던스 일치율, 가이던스 사용률 비교/상세
  - 에피소드 경로(trajectory), 최적 행동 히트맵, 방법 비교 바차트 등
- 결과 수치: `outputs/hybrid/hybrid_results.npz`



