## BeysianReinforcement 개요

### 목적
초기 버전의 Thompson Sampling 기반 시뮬레이션. 단일 alpha/beta 파라미터를 사용하고 에피소드 데이터를 저장

### 주요 스크립트
- `agent.py`: `BayesianAgent`가 행동 선택, 에피소드 데이터 저장. 저장 경로 접미사 `_8` 사용
- `environment.py`: 성공 시 alpha+, 실패 시 beta+ 등 기본 규칙을 적용
- `config.py`: 초기 파라미터(alpha=7, beta=3), 성공확률 범위, 승리조건
- `main.py`: 에피소드 실행 루프(기본 100회)

### 데이터 구조

  - `episode_*.json`: 에피소드별 로그


### 실행 방법
```bash
python main.py
```


### 참고
- 본 폴더는 후속 실험(예: `BayesianReinforcement_2`, `Alpha_Beta_Separate`)의 초기 형태



