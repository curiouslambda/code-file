# src/kref_rag/rag/state.py
from __future__ import annotations
from typing import TypedDict, List, Dict, Any

class GraphState(TypedDict, total=False):
    # 기존 필드
    query: str
    top_k: int
    retrieved: List[Dict[str, Any]]
    report_md: str
    
    # Claim 추출
    claims: List[Dict[str, Any]]  
    
    # GoT 구조
    edges: List[Dict[str, Any]]   
    unsupported_claims: List[str]  
    
    # 제어
    retry_count: int  # 재검색 횟수
