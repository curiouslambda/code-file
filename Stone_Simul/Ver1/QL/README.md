## QL

### 개요
- 전통적인 Q-러닝(테이블) 접근으로 A/B/C 환경에서 30스텝을 진행
- A/B 성공에는 양의 보상, C 성공에는 음의 보상(실패 시 반대)을 주며, 에피소드 종료 시 (A,B) 합·C≤4 조건에 추가 보상/패널티를 부여

### 구성 파일
- agent.py: `QLearningAgent`(Q-테이블, ε-greedy 액션 선택, 탭 업데이트)
- environment.py: 전 카테고리에 동일 성공확률을 적용하는 변형 환경, 스텝 보상과 최종 보상 합산
- train.py: 대규모 에피소드 학습, 주기적 JSON 저장, 통계/플롯 생성
- utils.py: 학습 통계 저장/로드, 학습 곡선과 돌 14/16 카운트 플롯 생성
- models.py: Q-테이블 저장/로드(pickle)
- main.py: 학습→저장→로드→간단 테스트
- config.py: α, γ, ε, 에피소드 수, 성능 점검 주기 등

### 실행
```bash
python main.py
```

### 산출물
- 모델: `q_learning_agent.pkl`, `q_learning_model.pkl`
- 로그/데이터: `data/` 내 주기적 JSON, `data/statistics/` 통계 JSON 3개

