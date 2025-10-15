import subprocess
import faiss
from transformers import BertTokenizer, BertModel
import numpy as np
from utils.LLM_client import LLMClient
from translator import Translator


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

print(documents)