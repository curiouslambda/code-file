## DQN_ver2

### 개요
- DQN 기반이지만 환경 보상 설계와 불가 액션 처리 로직이 강화된 버전
- C 카테고리의 성공/실패 보상을 차등화하고, (A,B) 7/9/9+ 조합과 C≤4 조건에 큰 보상을 부여

### 구성 파일
- agent.py: ε-탐욕 행동 선택, 리플레이 학습. `initial_steps` 기반 초기 휴리스틱 시도했으나나 현재는 비활성.
- environment.py: 
  - 남은 시도 0인 카테고리 선택 시 유효 액션 중 현재 Q값이 큰 액션을 강제 선택
  - 스텝 보상: A/B 성공 보상, C 실패 보상 등 세밀화
  - 에피소드 보상: (A,B) 7·7, 9·7, 9·9 등 조합 + C≤4에 큰 보상
- models.py: 3층 MLP + BatchNorm(학습 모드·배치>1일 때 BN 적용)
- train.py: 30스텝 루프, 상태 시퀀스 저장(`data/episode*_data.json`), 주기적 타깃 동기화 및 콘솔 로그
- config.py: 환경/학습 파라미터, `total_episodes`, `initial_steps`
- main.py: `Config.total_episodes`로 `train_dqn` 실행

### 실행
```bash
python main.py
```

### 산출물
- 모델 가중치: `dqn_model.pth`
- 로그/데이터: `data/episode*_data.json`

