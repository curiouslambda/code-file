## Alpha_Beta_Separate 개요

### 목적
Thompson Sampling을 기반으로 A/B와 C 카테고리의 베타분포 파라미터를 분리하여 시뮬레이션하고, 에피소드 데이터를 저장 및 시각화

### 주요 스크립트
- `agent.py`: `BayesianAgent`가 행동 선택(Thompson Sampling), 에피소드 데이터(JSON) 저장, 전체 데이터 시각화를 수행
- `environment.py`: 성공/실패 규칙 및 파라미터 업데이트. A/B 성공 시 alpha++, C 성공 시 beta_c++ 등 카테고리별 분리된 파라미터 적용
- `config.py`: 초기 베타분포 파라미터(`initial_alpha`, `initial_beta`, `initial_alpha_c`, `initial_beta_c`), 성공확률 상/하한, 승리조건 정의
- `graphs.py`: 알파/베타 변화, 처음 스텝 샘플 확률, 성공 추이, 베타분포 변화 그래프 저장
- `main.py`: 에피소드 실행 루프(기본 300회) 및 시각화 호출

### 데이터 구조

  - `episode_*.json`: 에피소드별 로그
  - `total_data.json`: 전체 에피소드 누적 데이터
  - `total/*.png`: 통계 시각화 이미지



### 실행 방법
```bash
python main.py
```


### 참고
- 성공 시 성공확률 감소, 실패 시 증가하는 규칙으로 난이도를 조정


