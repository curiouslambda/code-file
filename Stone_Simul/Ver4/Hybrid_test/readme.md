
# DP-Guided MCTS 기반 강화학습 프로젝트

## 📌 1. 프로젝트 개요

- **프로젝트명**: DP-guided MCTS 기반 강화학습 최적화
- **역할**: 알고리즘 설계 및 구현, 시각화 도구 개발, 실험 수행
- **사용 기술**: Python, Numpy, Matplotlib, YAML, 강화학습 (DP, MCTS), Hybrid RL

---

## 📌 2. 문제 정의 및 환경 설명

- **환경**: `LostArkStoneEnv` - 로스트아크 돌깎이 미니게임임 시스템 모사
- **상태 공간**: (a, b, c, p, remaining_slots)
- **전이**: 성공 시 확률 감소, 실패 시 증가 (0.25 ~ 0.75)
- **승리 조건**:
  - A 성공 횟수 ≥ 7
  - B 성공 횟수 ≥ 7
  - C 성공 횟수 ≤ 4
  - 최대 시도 횟수 30번

---

## 📘 3. 알고리즘 설계

### 🔹 Stage 1: DP (Dynamic Programming)
- 5차원 상태공간에 대한 Value Iteration 수행
- 수렴 조건 (θ), 최대 반복 수, 할인율 설정
- 정책 및 가치 함수 저장 및 시각화

### 🔹 Stage 2: MCTS (Monte Carlo Tree Search)
- UCB 기반 MCTS 구현 (Selection → Expansion → Simulation → Backpropagation)
- Root에서 N회 시뮬레이션 후 최다 방문 행동 선택
- 성능 평가 지표 수집 (성공률, 스텝 수 등)

### 🔹 Stage 3: Hybrid (DP + MCTS)
- DP 정책/가치 테이블을 기반으로 MCTS에 가이던스 제공
- 조건부 적용: 남은 슬롯 ≤ threshold일 경우만 DP-guided
- Hybrid UCB 점수 = (1-α)·UCB + α·DP score

---

## 📊 4. 실험 및 결과

- **비교 대상**: DP-only, MCTS-only, Hybrid
- **측정 지표**:
  - 성공률
  - 평균 시도 횟수
  - 가이던스 적용률 및 일치율
- **시각화**:
  - 정책 히트맵, 가치 변화 곡선, 에피소드 경로 (Trajectory)
  - 행동 선택 분포, 최적 행동 히트맵

---

## 🧠 5. 기술 스택 및 기여도

- Python 기반 강화학습 알고리즘 구현
- DP 학습 결과를 MCTS에 주입하는 Hybrid 구조 설계
- 실험 설정 및 자동화 (YAML 기반)
- 결과 시각화 및 통계 분석 도구 구축

---


## 📁 6. 코드 구조 요약

```
Hybrid_test/
├── stage1_dp/       # DP 계산 및 시각화
├── stage2_mcts/     # MCTS 탐색 기반 학습
├── stage3_hybrid/   # DP + MCTS 하이브리드 강화학습
├── common/          # 환경, 유틸리티, 시각화 도구
├── config/          # YAML 설정 파일
├── outputs/         # 로그 및 그래프 결과 저장
```



