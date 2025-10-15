# Hybrid_test/stage3_hybrid

### 목적
- DP 가이던스를 결합한 하이브리드 MCTS 실험 및 분석

### 주요 파일
- `hybrid_solver.py`: DP 가치/정책을 활용한 selection·simulation 가이드, 통계 수집/저장, 분석 출력
- `utils.py`: DP 결과(.npz) 로딩, 상태 인덱싱, DP 기반 우선순위/행동 선택, 하이브리드 UCB 점수 계산, 가이던스 적용 판정
- `run_hybrid.py`: 설정/로깅, 실험 실행, 결과 저장(npz), 다양한 그래프 생성, 방법 비교 모드 지원

### 실행
```bash
python Hybrid_test/stage3_hybrid/run_hybrid.py --config Hybrid_test/config/hybrid_config.yaml
```

### 산출물
- `outputs/hybrid/hybrid_results.npz`
- `outputs/hybrid/figures/` 하이브리드 관련 그래프 모음
