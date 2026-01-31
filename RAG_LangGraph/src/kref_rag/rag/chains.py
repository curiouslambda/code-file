# src/kref_rag/rag/chains.py
from __future__ import annotations
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel


def get_llm(model: str = None, temperature: float = None) -> BaseChatModel:
    """
    LLM 인스턴스 생성 (OpenAI 또는 Ollama 자동 선택)
    
    config.py의 LLM_PROVIDER 설정에 따라:
    - "openai": OpenAI API 사용 (API Key 필요)
    - "ollama": 로컬 Ollama 사용 (무료, API Key 불필요)
    """
    from kref_rag.config import LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE
    
    # 기본값 설정
    model = model or LLM_MODEL
    temperature = temperature if temperature is not None else LLM_TEMPERATURE
    provider = LLM_PROVIDER.lower()
    
    if provider == "openai":
        # OpenAI 사용
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY가 환경변수에 없습니다. "
                ".env 파일에 추가하거나 config.py에서 LLM_PROVIDER를 'ollama'로 변경하세요."
            )
        
        print(f"[LLM] OpenAI 사용: {model}")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_retries=2
        )
    
    elif provider == "ollama":
        # Ollama 사용 (로컬, 무료)
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama가 설치되지 않았습니다. "
                "터미널에서 'pip install langchain-ollama'를 실행하세요."
            )
        
        print(f"[LLM] Ollama 사용: {model}")
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url="http://localhost:11434"  # Ollama 기본 주소
        )
    
    else:
        raise ValueError(
            f"지원하지 않는 LLM_PROVIDER: {provider}\n"
            f"config.py에서 LLM_PROVIDER를 'openai' 또는 'ollama'로 설정하세요."
        )


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = None,
    temperature: float = None
) -> str:
    """
    LLM 호출 헬퍼
    
    Args:
        system_prompt: 시스템 프롬프트
        user_prompt: 사용자 프롬프트
        model: 모델명 (None이면 config.py의 LLM_MODEL 사용)
        temperature: 온도 (None이면 config.py의 LLM_TEMPERATURE 사용)
    
    Returns:
        LLM 응답 텍스트
    """
    llm = get_llm(model=model, temperature=temperature)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"[ERROR] LLM 호출 실패: {e}")
        return ""

