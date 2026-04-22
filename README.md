# Portfolio

그동안 진행한 프로젝트, 실험, 학습 기록을 모아둔 포트폴리오 리포지토리입니다.  
각 폴더는 하나의 독립적인 프로젝트이며, 데이터 처리, 딥러닝, 강화학습, 컴퓨터 비전, 웹 구현 관련 작업을 주제별로 정리했습니다.

## Repo Overview



- 데이터 수집 및 전처리
- 딥러닝 모델 구현 및 실험
- 강화학습 알고리즘 학습 및 확장
- 컴퓨터 비전 / 자연어처리 기반 프로젝트
- 웹 서비스 형태의 구현 시도

## Projects

### [Car_Repair](./Car_Repair)
차량 파손 이미지를 분석하고, 손상 부위 처리와 수리비 예측까지 연결해본 프로젝트입니다.  
데이터 전처리, U-Net 기반 이미지 처리, 예측 모델 실험, Django 웹 구성을 함께 다루고 있습니다.

### [Dashboard](./Dashboard)
Django 기반 대시보드 구조를 활용해 인증, 화면 구성, 관리 기능 등 웹 애플리케이션 형태를 구성해본 프로젝트입니다.

### [Deep_Numpy](./Deep_Numpy)
딥러닝 프레임워크 없이 수치 계산만으로 모델을 구현하고, PyTorch 방식과 비교하며 학습한 기록입니다.

### [Model_Study](./Model_Study)
GAN, Transformer, U-Net 등 주요 딥러닝 모델 구조를 직접 구현하거나 따라가며 학습한 스터디 폴더입니다.

### [News_llm](./News_llm)
뉴스 데이터를 수집하고 전처리한 뒤, LLM을 활용해 태그를 추출하고 구조화하는 프로젝트입니다.  
크롤링, 태깅, 저장 구조 설계, RAG 확장 아이디어까지 포함되어 있습니다.

### [NIA_Corpus](./NIA_Corpus)
'NIA 말뭉치 사업' 작업 폴더로 코퍼스 데이터 전처리와 NER 학습 및 파인튜닝 작업을 정리한 프로젝트입니다.

### [RAG_LangGraph](./RAG_LangGraph)
한국은행 보도자료를 수집·정제하고 벡터DB에 인덱싱한 뒤, 질의에 대해 Claim-Evidence 구조의 근거 기반 리포트를 생성하는 RAG 프로젝트입니다.  
LangGraph 기반 워크플로우로 수집, 인덱싱, 질의 응답, 그래프 시각화까지 연결해 결과를 리포트와 HTML 형태로 확인할 수 있습니다.

### [RL](./RL)
GridWorld 환경을 기반으로 Q-Learning, DQN 등 강화학습 알고리즘을 구현하고 비교한 프로젝트입니다.

### [Spaceship](./Spaceship)
데이터 분석과 전처리 중심으로 진행한 프로젝트로, 변수 탐색과 결측치 처리 과정을 정리했습니다.

### [Stone_Simul](./Stone_Simul)
강화학습과 탐색 알고리즘을 버전별로 발전시켜온 프로젝트입니다.  
Q-Learning, DQN, PPO에서 시작해 MCTS, DP-guided MCTS까지 확장하며 개선 과정을 정리했습니다.

### [Video](./Video)
Optical Flow를 중심으로 영상 처리와 움직임 분석을 실험한 프로젝트입니다.

### [YOLO_SAM2](./YOLO_SAM2)
YOLO 기반 객체 탐지와 SAM2 기반 인스턴스 세그멘테이션을 결합해 도로 장면을 해석하고 overlay 결과를 시각화하는 PoC 프로젝트입니다.  
프레임 정리, 객체 검출, 마스크 전파, semantic map 변환, 최종 렌더링까지의 파이프라인을 코드로 구성했습니다.

## Tech Stack

- `Python`
- `Django`
- `PyTorch`
- `Numpy`
- `Pandas`
- `OpenCV`
- `LLM`
- `RAG`
- `Q-Learning`, `DQN`, `PPO`, `MCTS`

## Portfolio Site

프로젝트를 한눈에 볼 수 있는 포트폴리오 사이트입니다.  
소개, 프로젝트, 연구 및 학습 기록을 웹 형태로 정리했습니다.

- [SimHyuna Portfolio](https://hyuna.simpo.pro/)
