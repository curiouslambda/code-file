# NIA_Corpus

코퍼스 데이터 수집, 전처리, 토크나이징, NER 학습 관련 작업을 정리한 프로젝트입니다.

## Project Overview
이 프로젝트는 텍스트 데이터를 수집하고, 모델 학습에 적합한 형태로 가공한 뒤,  
NER 태스크에 활용할 수 있도록 정리한 작업물입니다.

뉴스 및 소셜 데이터 수집 실험을 바탕으로 말뭉치를 구축하고,  
전처리와 토크나이징, 학습 데이터 구성 과정을 거쳐 NER 학습 및 파인튜닝 흐름까지 함께 살펴본 프로젝트입니다.

## Goal
- 원시 텍스트 데이터를 수집하고 정리
- 학습 가능한 형태의 코퍼스 구축
- 토크나이징 및 데이터 포맷 정리
- NER 학습용 데이터셋 구성
- 파인튜닝 흐름 이해 및 관련 작업 참여
- 추론 흐름 확인 및 결과 활용 가능성 검토

## Directory Structure
- `ner-training-main/`
  - 학습 코드 및 데이터 관련 파일
- `Lexis_scraping_crawler.ipynb`
  - LexisNexis 기반 말뭉치 수집 실험
- `twitter_Get_20231012.ipynb`
  - API 기반 트위터 데이터 수집 실험
- `twitter_only_Get.ipynb`
  - 트위터 원시 데이터 수집 실험
- `tokenizing.ipynb`
  - 토크나이징 실험
- `ratio_calulate_1123.ipynb`
  - 데이터 비율 관련 분석
- `NER Fine-tuning.pdf`
  - 관련 정리 문서

## Main Work
- 뉴스 및 소셜 데이터 수집 실험
- 수집 데이터 정리 및 코퍼스화
- 텍스트 데이터 전처리
- JSON / JSONL 기반 데이터 다루기
- 토크나이징 및 포맷 정리
- NER 학습 데이터 구성
- 파인튜닝 과정 참여 및 흐름 정리


## Tech Stack
- Python
- NLP
- NER
- Fine-tuning
- Tokenization
- Crawling
- API
- Selenium

## What I Learned
이 프로젝트를 통해 NLP 작업에서는 모델 학습 자체만큼이나 어떤 데이터를 수집하고 어떤 기준으로 정리하느냐가 결과에 큰 영향을 준다는 점을 체감했습니다.

또한 데이터 수집 단계에서는 소스마다 형식과 품질이 다르기 때문에, 이후 전처리와 학습 단계까지 고려한 데이터 관리가 중요하다는 점을 배울 수 있었습니다.

파인튜닝 관련 작업과 관련해서는 학습 코드 자체보다도 데이터 품질, 라벨 구조, 전처리 방식이 성능에 큰 영향을 준다는 점을 다시 확인할 수 있었습니다.
