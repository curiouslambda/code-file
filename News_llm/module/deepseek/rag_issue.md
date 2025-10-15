## 2월 26일 수정
### Deepseek 모델 사용
- 기존 `rag_ex.ipynb`(`RAG -rag_ex.ipynb`) 파일은 gpt2를 사용하여 질의 응답
- 현재는 `deepseek - LLM_client.py` 파일의 클래스 활용 -> deepseek로 질의 응답 하는 것으로 변경
- `deepseek` 폴더로 파일 위치 변경
- `rag_deepseek.py` 파일로 변경함
  
<br>

### Translator 생성
- 기존 test 시, 영어로 질의 응답
- 실제로 사용하게 될 경우, 한국어로 query를 입력할 가능성이 높음
- deepseek의 경우 한국어에 대한 성능은 영어 보다 다소 떨어짐
- 성능이 더 나은 영어로 질의 응답 하는 것이 더 좋은 결과를 낼 것으로 판단
- 따라서 query를 입력 받았을 때 한국어일 경우 영어로 번역 후 deepseek에 질의를 전달하도록 변경

<br>

### RAG의 document
- 기존 test 시, 영어로 참고 자료(document) 작성
- 실제로 사용 시 참고 자료(document)는 crawling data(한국어)가 될 것
- RAG의 참고 자료(document) 유사도 측정 및 판단의 경우 한국어를 사용해도 문제가 되지는 않음
- 대신, 한국어에 대한 성능이 준수한 embedding model(LLM model) 선택이 중요함
- 따라서 이 경우, document에 대한 번역은 추가로 고려할 필요 없을 듯

<br>

### 추가할 기능
- crawling data를 가져와서 Reference document로 만드는 기능 필요
- deepseek로 대답 출력 시, think 부분 안나오게 출력 필요

<br>

### 추가 고려 사항
- 파일 관리와 가시성을 고려 했을 때 RAG 폴더를 분리하는게 나아보임
- 언어 충돌
  - deepseek 모델에 질의 시, `input_text`로 들어가는 3가지 요인은 Role Prompt, Reference, Query 임
  - Reference에 들어가는 값은 RAG에서 유사도 측정 결과 가장 유사도가 큰 문서(str)
  - 즉, Reference는 한국어
  - 하지만 query는 영어로 번역하여 들어감
  - 참고는 한국어로 하고 질문은 영어로 받는 이상 현상이 일어남
  - 해결을 위해서 참고 자료도 영어로 바꾸는 경우의 수도 생각 중
- 번역 충돌
  - 위와 같이 참고 자료(document)도 영어로 번역해서 넣는다고 할 때 생각해야 할 부분
  - 회사 이름, 물품이나 기술의 이름 과 같은 고유 명사는 영어로 그대로 표현 되어 있는 경우가 많음 (ex. KKR, Fountainvest Partners, PAG)
  - 영어와 한국어가 혼재되어 작성된 경우 번역이 애매모호할 것으로 예상
  <br> (ex. 로건 총재는 현재의 ample reserves 방식이 준비금 수요 변동에 대응하기 더 쉽고, 금리 통제를 단순화한다고 전함)
  - 이 경우는 deepseek에서 어떻게 출력되는지 확인이 필요
- sample data가 있다면 test 해 보는 것이 좋을 것 같다

### 토의 사항 (2.26)
- 대명사나 자주 사용하는 단어, 말들을 위한 RAG를 만들어도 괜찮을 것 같다
- 뉴스 데이터 상업적으로 사용 가능한 무료 라이센스 없음
- 그냥 크롤링 가능한 사이트를 크롤링 해야할 듯
- 번역 할 때 deepseek 사용 시 시간이 걸려서 번역을 위한 모델 따로 준비 했음(model_list.txt)
- 디버깅 코드 여러개 준비 해놓고 코드를 만드는게 좋을 듯


### 토의 사항 (3.5)
- 크롤링 데이터 저장 폴더 필요
- 크롤링 데이터, 정제 데이터, 번역 데이터 3가지로 나눠진 폴더 있으면 좋을 듯
- 저장할 때는 csv(or dataframe) 형태로 저장
- rag에 들어가는 크롤링 데이터는 줄글 형식이어야 함
- 줄글 형식이지만 Title/Body 형태의 구조화 된 줄글 형식으로 rag 문서 전달할 예정
- prompt_manager.py(main branch) 에 있는 프롬프트 사용하여 각 상황별 필요한 prompt 활용
- (카더라) 에 관한 결정 아직 보류 - 높은 신뢰성 보장 되지만 공식적인 내용으로 판단할지 보류