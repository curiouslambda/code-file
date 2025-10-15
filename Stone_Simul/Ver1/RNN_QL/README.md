## RNN_QL

### 개요
- Q-러닝으로 정책을 학습하면서, 별도의 RNN 모델을 시퀀스→액션 분류로 학습하여 정책을 추정/보조

### 구성 파일
- agent.py: `QLearningAgent`(상태 문자열 키 기반 Q-테이블, ε-greedy)
- environment.py: 3카테고리, 성공 시 확률 하락/실패 시 상승; A/B는 양보상, C는 음보상(반대 경우 반대 보상)
- models.py: `RNNModel` 입력(시퀀스)→카테고리/확률 두 가지 출력을 예측
- train.py: Q-러닝 학습 루프와, RNN 분류 모델 학습(시퀀스 전개 후 CE 손실 최적화)
- main.py: Q-러닝 학습 후 RNN 예측 실행 및 결과 출력
- config.py: 성공확률/에피소드/ε/α/γ, RNN 입력/은닉/출력 크기

### 실행
```bash
python main.py
```

### 참고
- 보류


