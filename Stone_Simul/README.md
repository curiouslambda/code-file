# Stone_Simul

돌깎이 시뮬레이션 환경을 대상으로 강화학습과 탐색 알고리즘을 단계적으로 발전시켜온 프로젝트입니다.

## Project Overview
이 프로젝트는 하나의 알고리즘만 구현한 결과물이 아니라,  
같은 문제를 여러 방식으로 접근하고 버전별로 개선해온 과정을 담고 있습니다.

초기에는 Q-Learning, DQN, PPO 같은 강화학습 중심 접근을 시도했고,  
이후에는 MCTS와 DP를 결합하는 방향으로 확장하면서 더 정교한 탐색 전략을 설계했습니다.

## Problem Setting
로스트아크 돌깎이 시스템을 모사한 환경에서,  
제한된 시도 횟수와 확률 변화 조건 안에서 목표 상태를 최대한 효율적으로 달성하는 문제를 다뤘습니다.

이 문제는 단순 확률 게임처럼 보이지만,  
상태 전이와 선택 전략이 누적되기 때문에 강화학습과 탐색 알고리즘을 적용해보기 좋은 구조라고 판단했습니다.

## Version History
- `Ver1/`
  - Q-Learning, DQN, PPO 등 초기 강화학습 실험
- `Ver2/`
  - 알고리즘 확장 및 구조 개선
- `Ver3/`
  - MCTS 중심 접근 실험
- `Ver4/`
  - 공통 구조 정리 및 단계별 실행 구조 정돈
- `Ver5/`
  - DP-guided MCTS 기반 하이브리드 강화학습 고도화

## Core Ideas
- 강화학습 기반 정책 학습
- Monte Carlo Tree Search를 활용한 탐색
- Dynamic Programming 기반 가치 정보 활용
- DP와 MCTS를 결합한 Hybrid 구조 설계
- 실험 결과 시각화 및 비교

## Main Work
- 환경 정의 및 상태 공간 설계
- 알고리즘별 실험 코드 구현
- 버전별 비교와 개선
- DP, MCTS, Hybrid RL 구조 설계
- 로그 및 시각화 결과 정리

## Tech Stack
- Python
- Numpy
- Matplotlib
- YAML
- Reinforcement Learning
- MCTS
- Dynamic Programming

## Why This Project Matters
이 프로젝트의 가장 큰 의미는 최종 성능 하나보다도,  
문제를 풀기 위해 어떤 알고리즘을 시도했고 왜 다음 단계로 확장했는지가 코드 구조 안에 남아 있다는 점입니다.

즉, 단순 구현이 아니라  
문제 정의 -> 실험 -> 한계 확인 -> 개선된 접근 설계라는 흐름을 보여주는 대표 프로젝트입니다.
