## BayesianReinforcement_2 개요

### 목적
카테고리 공용(alpha/beta) 파라미터로 Thompson Sampling 시뮬레이션을 수행하고, 에피소드 결과를 JSON으로 저장

### 주요 스크립트
- `agent.py`: `BayesianAgent`가 행동 선택(Thompson Sampling), 에피소드 데이터 저장 수행
- `environment.py`: 성공/실패에 따른 파라미터 업데이트. C 성공 시 beta+, A/B 성공 시 alpha+ 실패 시 비C는 beta+
- `config.py`: 초기 파라미터(alpha=1, beta=1), 성공확률 범위, 승리조건 정의
- `main.py`: 에피소드 실행 루프(기본 100회)

### 데이터 구조

  - `episode_*.json`: 에피소드별 로그


### 실행 방법
```bash
python main.py
```


### 참고
- 공용 alpha/beta를 사용하며, C에 대한 업데이트 규칙이 A/B와 다름름



