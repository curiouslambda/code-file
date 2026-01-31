# src/kref_rag/config.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
import torch

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "runs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

BOK_RSS_URL = "https://www.bok.or.kr/portal/bbs/P0000559/news.rss?menuNo=200690"

DEFAULT_LIMIT = 20
DEFAULT_TOP_K = 6

CHUNK_SIZE = 900
CHUNK_OVERLAP = 200  # 맥락 유지를 위해 증가 (150 → 200)

EMBED_MODEL = "intfloat/multilingual-e5-base"   # 가볍고 한국어도 무난

# LLM 설정
LLM_PROVIDER = "ollama"  # "openai" 또는 "ollama" --> ollama사용
LLM_MODEL = "qwen2.5:7b"  # Ollama: "qwen2.5:7b", "llama3.1:8b", etc.
LLM_TEMPERATURE = 0.1


def make_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def ensure_run_dir(run_id: str) -> Path:
    run_dir = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Mac 전용(MPS). Windows에선 보통 False
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

EMBED_DEVICE = get_device()

def device_summary() -> str:
    device = EMBED_DEVICE
    if device == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if device == "mps":
        return "mps"
    return "cpu"


