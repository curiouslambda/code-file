import subprocess
import faiss
from transformers import BertTokenizer, BertModel
import numpy as np
from utils.LLM_client import LLMClient
from translator import Translator

'''
번역 모델
Bllossom/llama3.2-Korean-Bllossom-3B-gguf-Q4_K_m

저장되어 있는 모델명
bllossom-llama3.2
'''


# run.py 실행 후 출력 가져오기
def get_run_output():
    try:
        result = subprocess.run(
            ["python", "../crawler/run.py"], 
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split("\n")  # 여러 줄 출력일 경우 리스트로 저장
    except subprocess.CalledProcessError as e:
        print("Error executing run.py:", e)
        return []

documents = get_run_output()

'''
# 문서 리스트
documents = [
    "The economy of France is based on a highly developed market-oriented economy.",
    "Inflation in the United States has been a key issue for policymakers in 2023.",
    "The stock market showed a strong recovery after a weak quarter.",
    "Cryptocurrency regulations are being discussed in major financial hubs."
]
'''

# 문서 임베딩 생성 함수
def get_embeddings(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

# BERT 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 문서 임베딩 생성
doc_embeddings = get_embeddings(documents, model, tokenizer)

# FAISS 인덱스 생성
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# 검색 함수
def search(query, top_k=1):
    query_embedding = get_embeddings([query], model, tokenizer)
    D, I = index.search(query_embedding, top_k)
    return [documents[i] for i in I[0]]


# DeepSeekClient 인스턴스 생성
deepseek_client = LLMClient(model="deepseek-r1:14b")

# Translator 인스턴스 생성
translator = Translator()

# DeepSeekClient를 사용한 답변 생성 함수
def generate_answer(query, context):
    # 한국어일 경우 영어로 번역
    if translator.is_korean(query):
        query = translator.translate_to_english(query)


    # 역할 및 지침 추가
    role_prompt = (
        "You are a chatbot that delivers and analyzes economic news.\n"
        "Your role is to answer users' questions based on the given reference.\n"
        "If the user's question is not related to the reference, you must only say, 'I cannot answer with the information I have.'\n\n"
    )
    
    input_text = f"{role_prompt}Reference: {context}\n\nQuestion: {query}\n\nAnswer:"

    return deepseek_client.ask(input_text)  # DeepSeek 모델에 질의하여 응답 반환

# 사용자 질문
query = "What is the inflation rate in 2023?"

# 관련 문서 검색
retrieved_docs = search(query, top_k=1)
context = retrieved_docs[0]

# 답변 생성
answer = generate_answer(query, context)
print(f"Question: {query}")
print(f"Answer: {answer}")
