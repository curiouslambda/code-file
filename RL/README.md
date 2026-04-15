# RL

GridWorld 환경을 기반으로 강화학습 알고리즘을 직접 구현하고 비교한 프로젝트입니다.

## Project Overview
이 폴더는 강화학습의 기본 개념을 직접 코드로 구현하며 이해하기 위해 만든 작업물입니다.  
환경 정의, 에이전트 설계, 학습 루프, 결과 시각화까지 포함하여 강화학습의 전형적인 구성 요소를 단계적으로 다뤘습니다.

## Goal
- GridWorld 환경 직접 구현
- Q-Learning 방식 학습 실험
- DQN 방식으로 확장
- 결과를 시각화하며 학습 흐름 비교

## Directory Structure
- `GridWorld/`
  - 환경 정의
  - 기본 에이전트 코드
  - 실행 스크립트
- `GridWorld/DQN/`
  - DQN 기반 실험 코드
- `GridWorld/plot/`
  - 학습 결과 그래프
- `GridWorld/test/`
  - 테스트용 예제 환경 및 스크립트

## Main Work
- GridWorld 상태/보상 구조 정의
- Q-Learning 에이전트 구현
- DQN 기반 확장 실험
- 하이퍼파라미터 변화에 따른 결과 비교
- 그래프 기반 학습 결과 확인

## Tech Stack
- Python
- Reinforcement Learning
- Q-Learning
- DQN
- Matplotlib

## What I Learned
이 프로젝트를 통해 강화학습 알고리즘을 라이브러리 없이 직접 구성해보면서,  
환경 설계가 학습 결과에 얼마나 큰 영향을 미치는지 확인할 수 있었습니다.

또한 같은 문제라도  
테이블 기반 접근과 신경망 기반 접근이 어떻게 다른지 비교하며 이해할 수 있었습니다.
