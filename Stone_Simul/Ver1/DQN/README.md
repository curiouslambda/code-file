## DQN

### 개요
- A/B/C 3개 카테고리 중에서 30회 선택하며, 성공하면 전체 성공확률이 0.1 하락, 실패하면 0.1 상승하는 환경에서 DQN을 학습
- 에피소드 종료 시 (A,B) 성공 합과 C 성공 횟수 조건에 따라 추가 보상이 부여

### 구성 파일
- agent.py: DQN 에이전트. 타깃 네트워크 동기화, ε-탐욕 행동 선택, 리플레이로 MSE 손실 최적화(Adam)
- environment.py: 게임 환경. 불가 액션 시 대안 선택 유도, 확률 업데이트, 최종 보상 계산
- models.py: 3층 MLP + BatchNorm(학습 모드이면서 배치>1일 때만 BN 적용)
- train.py: 에피소드 루프, 상태 시퀀스 JSON 저장(`data/episode*_data.json`), 타깃 주기 업데이트
- config.py: 하이퍼파라미터(γ, ε, lr, 배치, 메모리 등) 및 환경 설정
- main.py: `train_dqn` 실행 진입점



### 산출물
- 모델 가중치: `dqn_model.pth`
- 로그/데이터: `data/episode*_data.json`


