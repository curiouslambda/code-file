# Stage1_dp

### 목적
- DP(Value Iteration) 기반 정책/가치 함수 계산 및 시각화

### 주요 파일
- `dp_solver.py`: 5차원 상태공간에서 Value Iteration 수행, 터미널 상태 보상 고정, 성공/실패 분기 기대가치 계산, 결과 저장 및 그래프 생성
- `run_dp.py`: 설정 로드 → 로깅 → 환경 생성 → DP 실행 → 결과 저장 파이프라인

### 실행
```bash
python stage1_dp/run_dp.py
```

### 산출물
- `outputs/dp_values.npz`(V, policy), 여러 PNG 그래프(수렴곡선, 정책/가치 히트맵, 확률/슬롯 변화, 상태 전이)

