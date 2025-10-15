import subprocess

class LLMClient:
    def __init__(self, model: str = "deepseek-r1:14b"):
        """
        model: Ollama에서 사용할 모델 이름 (예: 'deepseek-r1:14b', 'gguf-deepseek-r1-14b.bin' 등)
        """
        self.model = model
    
    def ask(self, prompt_text: str) -> str:
        """
        Ollama를 사용해 DeepSeek 모델에게 질의하고 응답을 반환.
        """
        cmd = ["ollama", "run", self.model, prompt_text]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=30)
            
            if result.returncode == 0:
                output = result.stdout.strip() if result.stdout else "No response from model"
                return output
            else:
                return f"Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Error: Request timed out. The model took too long to respond."
        except Exception as e:
            return f"Exception: {str(e)}"
