# src/kref_rag/indexing/chunker.py
from __future__ import annotations

import hashlib
from typing import Any, Dict, List


def is_meaningful_chunk(text: str, min_length: int = 150) -> bool:
    """
    의미있는 청크인지 판단 (노이즈 청크 필터링)
    
    Args:
        text: 청크 텍스트
        min_length: 최소 길이
    
    Returns:
        True if 청크가 의미있음, False if 필터링해야 함
    """
    text = text.strip()
    
    # 1. 너무 짧으면 제외
    if len(text) < min_length:
        return False
    
    # 2. UI/메타데이터 키워드가 많으면 제외
    junk_keywords = [
        "목록", "메뉴", "연관자료", "만족도", "담당부서",
        "전화번호", "확인", "본문내용 바로가기", "주메뉴",
        "다운로드", "뷰어", "통계보기", "내가 본 콘텐츠",
        "페이지 위로", "이전", "다음"
    ]
    keyword_count = sum(text.count(kw) for kw in junk_keywords)
    
    # junk 키워드가 5개 이상이면 제외
    if keyword_count >= 5:
        return False
    
    # 3. 실제 문장이 거의 없으면 제외 (마침표 개수로 판단)
    sentence_count = text.count('.') + text.count('다.') + text.count('。')
    if sentence_count < 2:
        return False
    
    # 4. 핵심 정책 키워드가 하나도 없으면 관련성 낮음
    policy_keywords = [
        "기준금리", "금융통화", "정책", "경제", "물가", "성장",
        "금융안정", "통화", "한국은행", "위원회", "결정", "운용"
    ]
    has_policy_keyword = any(kw in text for kw in policy_keywords)
    
    # 정책 키워드가 없고 문장이 짧으면 제외
    if not has_policy_keyword and sentence_count < 4:
        return False
    
    return True


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    chunks: List[str] = []
    n = len(text)
    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(0, end - overlap)

    return chunks


def docs_to_chunks(
    docs: List[Dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for doc in docs:
        doc_id = str(doc.get("doc_id", "")).strip()
        url = str(doc.get("url", "")).strip()
        published_at = doc.get("published_at")
        source = str(doc.get("source", "")).strip()
        title = str(doc.get("title", "")).strip()

        content = str(doc.get("content_text", "") or "").strip()
        if not content:
            continue

        parts = chunk_text(content, chunk_size=chunk_size, overlap=overlap)

        for i, p in enumerate(parts):
            # 의미없는 청크 필터링
            if not is_meaningful_chunk(p):
                continue
            
            chunk_id = _make_chunk_id(doc_id=doc_id or url, idx=i, text=p)
            out.append({
                "chunk_id": chunk_id,
                "text": p,
                "doc_id": doc_id,
                "url": url,
                "published_at": published_at,
                "source": source,
                "title": title,
                "chunk_index": i,
            })

    return out


def _make_chunk_id(doc_id: str, idx: int, text: str) -> str:
    """
    안정적인 chunk id 생성:
    - doc_id/url + idx + text hash -> 재실행 시 동일 chunk면 동일 id
    """
    h = hashlib.sha1()
    h.update((doc_id or "").encode("utf-8"))
    h.update(b"|")
    h.update(str(idx).encode("utf-8"))
    h.update(b"|")

    h.update(hashlib.sha1(text.encode("utf-8")).digest())
    return h.hexdigest()
