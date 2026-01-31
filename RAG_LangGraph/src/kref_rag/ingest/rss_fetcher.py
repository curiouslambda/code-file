# src/kref_rag/ingest/rss_fetcher.py
from __future__ import annotations
import feedparser
from datetime import datetime, timezone
from typing import List, Dict, Any
import hashlib

def _to_iso_date(entry: dict) -> str:
    t = entry.get("published_parsed") or entry.get("updated_parsed")
    if not t:
        return ""
    dt = datetime(*t[:6], tzinfo=timezone.utc)
    return dt.date().isoformat()

def _hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def fetch_rss_entries(rss_url: str, limit: int = 20) -> List[Dict[str, Any]]:
    feed = feedparser.parse(rss_url)
    entries = []
    
    # 제외할 카테고리 키워드 (보도자료만 가져오기)
    exclude_keywords = [
        "의사록",
        "의결사항",
        "주간보도계획",
        "언론보도해명"
    ]
    
    for e in feed.entries:
        title = e.get("title", "").strip()
        
        # 제외 키워드가 포함된 항목은 스킵
        if any(keyword in title for keyword in exclude_keywords):
            continue
        
        link = e.get("link", "")
        entries.append({
            "doc_id": _hash_id(link),
            "title": title,
            "url": link,
            "published_at": _to_iso_date(e),
            "source": "BOK_P0000559",
        })
        
        # limit 적용 (필터링 후)
        if len(entries) >= limit:
            break
    
    return entries
