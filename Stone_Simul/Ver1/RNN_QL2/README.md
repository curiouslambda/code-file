## RNN_QL2

### 개요
- RNN 분류와 QAgent 테이블 학습을 모두 포함하며, 두 가지 변형 환경(`environment.py`, `environment_2.py`)을 제공

### 구성 파일
- agent.py: `QAgent`(상태를 3차원 성공횟수 인덱스로 변환하여 테이블 색인, ε-greedy)
- environment.py / environment_2.py: 시도/성공/확률 상태를 갖는 환경. 성공/실패 시 확률을 0.05~0.10 단위로 조정
- models.py: 단순 RNN 기반 분류기(`RNNModel`)
- train.py: 랜덤 시퀀스로 RNN을 학습, QAgent로 탭 Q-학습 진행(로그 포함)
- train_2.py: `environment_2` 버전용 RNN/Q-러닝 학습 루틴
- config.py: 입력/은닉/출력 크기와 에피소드·에폭·LR
- main.py: `train_rnn`, `train_q_learning`를 순차 실행

### 실행
```bash
python main.py
```

### 참고
- 보류

