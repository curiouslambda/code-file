## preprocessing 프로젝트 개요

### 파일 트리
```text
root\
  - ISSUE.md
  - README.md
  - PROJECT_OVERVIEW.md  ← 이 파일
  - RAG\
    - RAG.md
    - rag_ex.ipynb
    - requirements.txt
  - module\
    - crawler\
      - run.py
      - utils\
        - core.py
    - deepseek\
      - data\
        - log\
          - 20250212123438_log.log
          - 20250219215818_log.log
        - prompt\
          - prompt.json
      - readme.md
      - rag_issue.md
      - rag_deepseek.py
      - rag_deepseek.ipynb
      - run.py
      - tag_extractor.py
      - test.py
      - deepseek_chat_history.csv
      - model_list.txt
      - reference\
        - RAG_with_FAISS.md
      - test\
        - prompt_test.csv
        - prompt_test.py
        - view_test.ipynb
      - translator.py
      - utils\
        - LLM_client.py
        - logger_manager.py
        - prompt_manager.py
  - temp\
    - crawler_tmp.ipynb
```

### 요약
- **목표**: 뉴스 데이터 크롤링 → 전처리 → LLM 기반 태그 추출 → 구조적 저장 및 RAG 활용. 주가와 뉴스의 연계 분석 확장 계획 포함.
- **진행 상태**: 수집/전처리/프롬프트 관리 구조를 정리 중이며, NAS 저장 및 RAG 적용 방향 논의됨. DeepSeek 기반 질의응답으로 전환, 번역 모듈 도입.

### 핵심 문서
- `README.md`: 전체 목표와 단계(크롤링, 전처리, 태그 추출, 저장, 주가 연계) 및 회의 메모.
- `ISSUE.md`: 우선순위(수집, 전처리, 프롬프트 관리, NAS, RAG)와 진행 중 항목.
- `RAG/RAG.md`: RAG 개념 정리(검색/생성, 장단점, 아키텍처).
- `module/deepseek/readme.md`: 태그 데이터 수집 파이프라인, RAG 단계(FAISS), 태그-주식 연결 흐름.
- `module/deepseek/rag_issue.md`: DeepSeek/Translator 도입 배경, 한국어/영어 혼용 이슈, 크롤링→레퍼런스 문서화 필요.

### 주요 스크립트 개요
- `module/crawler/run.py`
  - 지정 URL에서 HTML을 가져와 `<p>` 본문을 섹션별(`h2`)로 매핑하여 텍스트 추출.
- `module/deepseek/rag_deepseek.py`
  - `crawler/run.py` 출력 텍스트를 문서로 사용 → BERT 임베딩 → FAISS 인덱스 검색 → DeepSeek(`LLM_client`) + `Translator` 조합으로 답변 생성.
- `module/deepseek/utils/LLM_client.py`
  - `ollama run <model>`을 호출해 로컬 LLM(예: `deepseek-r1:14b`)과 상호작용.
- `module/deepseek/translator.py`
  - 간단한 한국어 판별 후 `bllossom-llama3.2`로 번역(한↔영). 크롤러 출력 연계 함수 포함.
- `module/deepseek/tag_extractor.py`
  - 태그 추출기 스켈레톤(구현 준비됨).

### 데이터/리소스
- `module/deepseek/data/`
  - `log/`: 실행 로그
  - `prompt/prompt.json`: 프롬프트 템플릿
- `module/deepseek/reference/RAG_with_FAISS.md`: 참고 자료
- `module/deepseek/model_list.txt`: 번역 전용 모델 등 모델 리스트
- `module/deepseek/deepseek_chat_history.csv`: 대화 로그

### 실행 흐름(개요)
1. `module/crawler/run.py`로 웹에서 뉴스/본문 텍스트 수집.
2. `module/deepseek/rag_deepseek.py`에서 임베딩/검색(FAISS) 후 DeepSeek로 질의응답 생성. 한국어 질의 시 번역 모듈 사용.

### 의존성(핵심)
- Python, `requests`, `beautifulsoup4`, `transformers`, `faiss-cpu`, (PyTorch), `ollama` 환경 및 모델(`deepseek-r1:14b`, `bllossom-llama3.2`).
- 추가: Jupyter, 데이터/프롬프트 파일들.


