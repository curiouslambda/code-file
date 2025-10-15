import json
import os

class PromptManager:
    def __init__(self, file_path="prompts.json"):
        """
        JSON 파일에서 프롬프트 데이터를 로드하는 클래스.
        """
        self.file_path = file_path
        self.prompts = self.load_prompts()

    def load_prompts(self):
        """
        JSON 파일을 로드하여 프롬프트 딕셔너리로 반환.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Prompt file {self.file_path} not found.")
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_prompt(self, key, **kwargs):
        """
        프롬프트 키를 사용하여 프롬프트 텍스트를 가져오고, 필요한 변수들을 채운다.
        """
        prompt_template = self.prompts.get(key, "")
        if not prompt_template:
            raise KeyError(f"Prompt key '{key}' not found in {self.file_path}")
        
        return prompt_template.format(**kwargs)

# 사용 예시
if __name__ == "__main__":
    prompt_manager = PromptManager()
    
    context_text = "Amazon reported a 15% increase in quarterly earnings."
    
    try:
        print(prompt_manager.get_prompt("tag_extraction", context=context_text))
    except KeyError as e:
        print(e)
