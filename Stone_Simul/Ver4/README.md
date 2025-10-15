# Stone Simulator

돌깎이 게임 시뮬레이션을 위한 강화학습 기반 솔루션

## 프로젝트 구조

```
lostark_stone_opt/
├── README.md
├── requirements.txt
├── config/
│   ├── dp_config.yaml
│   ├── mcts_config.yaml
│   └── hybrid_config.yaml
├── common/
│   ├── __init__.py
│   ├── env.py              # MDP 환경 정의
│   └── utils.py            # 로깅·메트릭·공용 함수
├── stage1_dp/
│   ├── __init__.py
│   ├── dp_solver.py        # Value Iteration 구현
│   └── run_dp.py           # dp_solver 실행 스크립트
├── stage2_mcts/
│   ├── __init__.py
│   ├── mcts.py             # 순수 MCTS 구현
│   └── run_mcts.py         # mcts 실행 스크립트
├── stage3_hybrid/
│   ├── __init__.py
│   ├── hybrid_agent.py     # Prior + Leaf 절충 MCTS
│   └── run_hybrid.py       # hybrid 실행 스크립트
├── logs/                   # 단계별 .log 파일 자동 저장
└── outputs/                # 결과물 저장
    └── figures/            # 시각화 이미지
```


## 환경 설정

- `config/dp_config.yaml`: DP 관련 설정
  - gamma: 할인율 (기본값: 0.99)
  - theta: 수렴 기준 (기본값: 1e-6)
  - max_iterations: 최대 반복 횟수 (기본값: 1000)

## 결과물

- `outputs/dp_values.npz`: 학습된 가치 함수와 정책
- `outputs/figures/dp_value_curve.png`: Value Iteration 수렴 곡선
- `outputs/figures/dp_value_heatmap.png`: 가치 함수 히트맵
- `logs/dp.log`: 학습 과정 로그 