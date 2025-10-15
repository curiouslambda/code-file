## PPO_TS 개요

### 목적
PPO(Actor-Critic)와 Thompson Sampling을 결합한 하이브리드 실험. 환경 상태를 텍스트로 로깅

### 주요 스크립트
- `agent.py`: `ActorCritic` 모델 정의 및 PPO 업데이트 루틴, 행동 선택 시 베타분포에서 성공확률을 샘플링
- `environment.py`: 상태는 10차원 `[A남은기회,B남은기회,C남은기회, A성공,B성공,C성공, A실패,B실패,C실패, 성공확률]`
- `config.py`: PPO 하이퍼파라미터, 상태/행동 공간 크기, 최대 라운드
- `main.py`: 10 에피소드 훈련 루프, 매 에피소드 후 메모리 기반 업데이트

### 데이터 구조
- 경로: `data/`
  - `game_state_log_episode_*.txt`: 에피소드별 상태 로그
  - `game_state_log.txt`: 통합 로그

### 실행 방법
```bash
python main.py
```

필요 패키지(예시): torch, numpy, scipy

### 참고
- PPO 업데이트 주기(`update_timestep`)에 맞춰 에피소드 후 업데이트를 수행



