## Stage 1 — Dynamic Programming (Backward DP)

### 요약
- 돌시뮬레이터터 강화 환경(`LostArkStoneEnv`)에서 7차원 상태공간 `(a,b,c,p_idx,a_try,b_try,c_try)`에 대해 역방향 DP로 가치함수 `V`와 정책 `policy`를 계산
- 메모이제이션 기반의 재귀 `_compute_state_value`와 상태별 유효 액션 집합을 활용하여 기대가치를 계산

### 주요 파일
- `dp_solver.py`: `DPSolver` 구현
  - 상태 인덱싱, `_compute_state_value`, `backward_dp`, `value_iteration`, `save_results`
  - 타이브레이크 시 상태 해시 기반 시드로 일관된 랜덤 선택
- `run_dp.py`: 실행 스크립트
  - `config/dp_config.yaml` 로드, 시드/로깅 설정, `DPSolver` 실행 및 결과 저장

### 설정
- `config/dp_config.yaml`
  - `gamma=0.99`, `theta=1e-6`, `max_iterations=1000`
  - 출력 경로: `logs/dp.log`, 그래프(`outputs/figures/*.png`), 가치/정책(`outputs/dp_values*.npz`)

### 실행 방법
```bash
python stage1_dp/run_dp.py
```

### 출력물
- 값/정책: `outputs/dp_values.npz` 또는 `outputs/dp_values_c10.npz`
- 그래프(일부):
  - `dp_value_curve.png` (벨만 잔차 수렴곡선)
  - `dp_policy_decision_map*.png`, `dp_policy_margin_heatmap*.png`
  - `dp_value_heatmap*.png`, `dp_value_vs_probability*.png`, `dp_value_vs_slots*.png`
  - `dp_multi_c_value_heatmap.png`, `dp_state_transition.png`



