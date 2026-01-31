# src/kref_rag/rag/prompts/templates.py
from __future__ import annotations
from typing import List, Dict, Any


EXTRACT_CLAIMS_SYSTEM = """당신은 한국은행 경제 정책 분석 전문가입니다.

사용자의 질의에 대해 **3~5개의 핵심 주장(Claim)**을 추출하세요.

## Claim 작성 규칙:
1. **명확하고 구체적**: "금리가 변했다" ❌ → "2026년 1월 기준금리는 2.50%로 유지되었다" ✅
2. **검증 가능**: 문서에서 근거를 찾을 수 있어야 함
3. **질의와 직접 관련**: 질문에 답하는 내용만
4. **사실 기반**: 추측이나 의견 배제

## 출력 형식:
반드시 아래 JSON 형식으로만 출력하세요. 다른 설명은 쓰지 마세요.

{
  "claims": [
    {"id": "C1", "text": "주장 내용 1"},
    {"id": "C2", "text": "주장 내용 2"},
    {"id": "C3", "text": "주장 내용 3"}
  ]
}
"""

EXTRACT_CLAIMS_USER = """## 질의
{query}

## 검색된 문서 요약 (참고용)
{context_summary}

---
위 질의에 대한 3~5개의 핵심 주장을 JSON 형식으로 추출하세요.
"""


def format_extract_claims_prompt(
    query: str,
    retrieved: List[Dict[str, Any]]
) -> tuple[str, str]:
    """Extract Claims 프롬프트 생성"""
    
    # 검색 결과 요약 (최대 5개)
    summaries = []
    for i, r in enumerate(retrieved[:5], 1):
        meta = r.get("meta", {})
        title = meta.get("title", "제목 없음")
        text = r.get("text", "")[:150]  # 150자만
        summaries.append(f"{i}. {title}\n   {text}...")
    
    context_summary = "\n".join(summaries) if summaries else "(검색 결과 없음)"
    
    system = EXTRACT_CLAIMS_SYSTEM
    user = EXTRACT_CLAIMS_USER.format(
        query=query,
        context_summary=context_summary
    )
    
    return system, user


# Day 3: Match Evidence용 프롬프트 (선택적)
MATCH_EVIDENCE_SYSTEM = """당신은 Claim과 Evidence의 관련성을 판단하는 전문가입니다.

주어진 Claim과 Evidence가 관련 있는지 판단하세요.

출력 형식:
{
  "is_relevant": true/false,
  "confidence": 0.0~1.0,
  "reason": "이유 설명"
}
"""

