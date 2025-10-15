# Hybrid_test/stage1_dp

### 목적
- DP(Value Iteration) 실행 및 결과 저장

### 주요 파일
- `run_dp.py`: 설정 로드, 환경 생성, DP 실행, 결과 저장
- `dp_solver.py`: Value Iteration 로직(터미널 보상 고정, 성공/실패 분기 기대값), 그래프 생성

### 실행
```bash
python Hybrid_test/stage1_dp/run_dp.py --config Hybrid_test/config/dp_config.yaml
```

### 산출물
- `outputs/dp_values.npz`, `outputs/figures/*`(DP 그래프)
