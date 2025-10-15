## PPO

### 개요
- 정책(Policy)·가치(Value) 네트워크를 분리한 Proximal Policy Optimization(PPO) 구현
- 30스텝 궤적을 수집해 에피소드 단위로 업데이트하며, 클리핑 손실과 그래디언트 클리핑을 사용

### 구성 파일
- agent.py: PPOAgent. 메모리(deque)에 궤적 저장, 클립 파라미터, PPO 에폭/배치 크기, GAE(λ) 기반 이점 추정
- environment.py: DQN과 유사한 A/B/C 환경, 성공/실패에 따른 확률 조정과 최종 보상
- models.py: `PolicyNetwork`(softmax 확률), `ValueNetwork`(스칼라 가치) 정의
- train.py: 궤적 수집→JSON 저장→`agent.learn()`으로 업데이트, 에피소드 점수 기록
- config.py: PPO 관련 하이퍼파라미터(clip_param, ppo_epochs, ppo_batch_size, max_grad_norm 등) 및 디바이스 설정
- main.py: `train_ppo` 실행 

### 실행
```bash
python main.py
```

### 산출물
- 모델 가중치: `ppo_policy_model.pth`, `ppo_value_model.pth`
- 로그/데이터: `data/state_data_episode_*.json`

### 참고
- 진행 원활하지 않아 보류

