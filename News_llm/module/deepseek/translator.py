# translater.py
import subprocess
from utils.LLM_client import LLMClient

class Translator:
    def __init__(self):
        # DeepSeek 모델 인스턴스 생성
        self.client = LLMClient(model="bllossom-llama3.2")
    
    def translate_to_english(self, korean_text: str) -> str:
        """
        한국어를 영어로 번역하는 함수
        """
        prompt = f"Translate the following Korean text to English:\n{ korean_text }"
        return self.client.ask(prompt)
    
    def translate_to_korean(self, english_text: str) -> str:
        """
        영어를 한국어로 번역하는 함수
        """
        prompt = f"Translate the following English text to Korean:\n{ english_text }"
        return self.client.ask(prompt)
    
    def translate(self, text: str) -> str:
        """
        입력된 텍스트가 한국어인지 영어인지 판단하고, 해당 언어로 번역
        """
        # 텍스트가 한국어로 시작하면 영어로 번역, 그렇지 않으면 한국어로 번역
        if self.is_korean(text):
            print("한국어")
            return self.translate_to_english(text)
        else:
            return self.translate_to_korean(text)
    
    def is_korean(self, text: str) -> bool:
        """
        텍스트가 한국어인지 판단하는 함수
        """
        return any('가' <= char <= '힣' for char in text)
    
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



# 사용 예시
if __name__ == "__main__":
    translator = Translator()
    documents = get_run_output()
    # print(documents)

    # 예시: 한국어에서 영어로 번역
    korean_text = documents
    translated_to_english = translator.translate(korean_text)
    # print(f"Translated to English: {translated_to_english}")

    # # 예시: 영어에서 한국어로 번역
    # english_text = "Hello, nice to meet you."
    # translated_to_korean = translator.translate(english_text)
    # print(f"Translated to Korean: {translated_to_korean}")
