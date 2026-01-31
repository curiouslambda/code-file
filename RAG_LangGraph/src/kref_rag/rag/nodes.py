from __future__ import annotations

from typing import Dict, Any, List


def _squash_ws(s: str) -> str:
    return " ".join((s or "").split())


def node_retrieve(store, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    state: {"query": str, "top_k": int}
    return: {"retrieved": List[...]}
    """
    query = state.get("query", "")
    top_k = int(state.get("top_k", 6))

    # 더 많이 뽑아두고(중복 제거/우선순위 적용)
    raw = store.query(query, top_k=max(top_k * 4, 20))

    # 통화정책 관련 글 우선 수집
    prefer_keywords = ["통화정책방향", "통화신용정책 운영방향", "기준금리"]
    prefer, others = [], []
    for r in raw:
        title = (r.get("meta") or {}).get("title", "") or ""
        (prefer if any(k in title for k in prefer_keywords) else others).append(r)

    merged = prefer + others

    # doc_id 기준 중복 제거(같은 문서 chunk 여러 개 뜨는 현상 완화)
    dedup: List[Dict[str, Any]] = []
    seen = set()
    for r in merged:
        doc_id = (r.get("meta") or {}).get("doc_id") or r.get("chunk_id")
        if doc_id in seen:
            continue
        seen.add(doc_id)
        dedup.append(r)
        if len(dedup) >= top_k:
            break

    return {"retrieved": dedup}


def node_generate_report(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    retrieved를 보기 좋게 report_md로 만드는 단계(LLM 없이 evidence 나열).
    Day 2/3: claims와 edges도 포함
    """
    retrieved = state.get("retrieved", []) or []
    claims = state.get("claims", []) or []
    edges = state.get("edges", []) or []
    unsupported = state.get("unsupported_claims", []) or []

    lines = []
    lines.append(f"# Query\n- {_squash_ws(state.get('query', ''))}\n")
    
    # Claims 섹션
    if claims:
        lines.append("## Claims\n")
        for claim in claims:
            claim_id = claim.get("id", "")
            claim_text = claim.get("text", "")
            
            # 해당 Claim의 Evidence 찾기
            claim_edges = [e for e in edges if e.get("from") == claim_id]
            evidence_count = len(claim_edges)
            
            unsupported_mark = " ⚠️ (근거 없음)" if claim_id in unsupported else ""
            lines.append(f"### {claim_id}: {claim_text}{unsupported_mark}")
            lines.append(f"- Evidence 개수: {evidence_count}")
            
            if claim_edges:
                lines.append("- Evidence:")
                for e in claim_edges[:3]:  # 최대 3개만
                    chunk_id = e.get("to", "")
                    similarity = e.get("similarity", 0)
                    lines.append(f"  - {chunk_id} (유사도: {similarity:.3f})")
            lines.append("")
    
    lines.append("## Top Evidence\n")

    if not retrieved:
        lines.append("- (no results)\n")
        return {"report_md": "\n".join(lines)}

    for i, r in enumerate(retrieved, 1):
        meta = r.get("meta") or {}
        title = meta.get("title", "")
        url = meta.get("url", "")
        published_at = meta.get("published_at", "")
        dist = r.get("distance", None)

        snippet = _squash_ws(r.get("text", ""))[:320]

        lines.append(f"### E{i}. {title}")
        if published_at:
            lines.append(f"- date: {published_at}")
        if url:
            lines.append(f"- url: {url}")
        if dist is not None:
            lines.append(f"- distance: {dist}")
        lines.append(f"- snippet: {snippet}\n")

    return {"report_md": "\n".join(lines)}


# ===== Extract Claims =====

def node_extract_claims(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Claim 추출 노드 (LLM 사용)
    state: {"query": str, "retrieved": List[...]}
    return: {"claims": List[{"id": "C1", "text": "..."}, ...]}
    """
    import json
    from kref_rag.rag.chains import call_llm
    from kref_rag.rag.prompts.templates import format_extract_claims_prompt
    
    query = state.get("query", "")
    retrieved = state.get("retrieved", [])
    
    if not query:
        return {"claims": []}
    
    # 프롬프트 생성
    system, user = format_extract_claims_prompt(query, retrieved)
    
    # LLM 호출 (config.py의 설정 자동 사용)
    print("[node_extract_claims] LLM 호출 중...")
    response = call_llm(system, user)
    
    # JSON 파싱
    try:
        # ```json ... ``` 형식도 대응
        response_clean = response.strip()
        if response_clean.startswith("```"):
            # 코드 블록 제거
            lines = response_clean.split("\n")
            response_clean = "\n".join(lines[1:-1])
        
        result = json.loads(response_clean)
        claims = result.get("claims", [])
        
        print(f"[node_extract_claims] 추출된 Claim 개수: {len(claims)}")
        for c in claims:
            print(f"  - {c.get('id')}: {c.get('text')[:60]}...")
        
        return {"claims": claims}
    
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 파싱 실패: {e}")
        print(f"응답: {response[:200]}")
        return {"claims": []}


# ===== Match Evidence =====

def node_match_evidence(state: Dict[str, Any], store) -> Dict[str, Any]:
    """
    Claim-Evidence 매칭 노드
    각 Claim을 쿼리로 재검색하여 Evidence와 연결
    """
    claims = state.get("claims", [])
    
    if not claims:
        return {"edges": []}
    
    edges = []
    
    print(f"[node_match_evidence] {len(claims)}개 Claim에 대해 Evidence 매칭 중...")
    
    for claim in claims:
        claim_id = claim.get("id", "")
        claim_text = claim.get("text", "")
        
        # Claim 텍스트로 재검색
        results = store.query(claim_text, top_k=3)
        
        print(f"  - {claim_id}: {len(results)}개 Evidence 발견")
        
        # Edge 생성
        for r in results:
            chunk_id = r.get("chunk_id", "")
            distance = r.get("distance", 1.0)
            similarity = 1 - distance  # distance → similarity 변환
            
            # Threshold: 유사도 0.5 이상만
            if similarity >= 0.5:
                edge = {
                    "from": claim_id,
                    "to": chunk_id,
                    "similarity": similarity
                }
                edges.append(edge)
    
    print(f"[node_match_evidence] 총 {len(edges)}개 Edge 생성")
    
    return {"edges": edges}


# ===== Validate =====

def node_validate(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    검증 노드: 근거 없는 Claim 찾기
    """
    claims = state.get("claims", [])
    edges = state.get("edges", [])
    
    if not claims:
        return {"unsupported_claims": []}
    
    # Claim별 Edge 개수 세기
    claim_evidence_count = {}
    for claim in claims:
        claim_id = claim.get("id", "")
        count = len([e for e in edges if e.get("from") == claim_id])
        claim_evidence_count[claim_id] = count
    
    # 근거 없는 Claim (Edge가 0개)
    unsupported = [
        cid for cid, count in claim_evidence_count.items()
        if count == 0
    ]
    
    print(f"[node_validate] Claim 검증 결과:")
    for cid, count in claim_evidence_count.items():
        status = "OK" if count > 0 else "WARN"
        print(f"  [{status}] {cid}: Evidence {count}개")
    
    if unsupported:
        print(f"[node_validate] [WARN] 근거 없는 Claim: {unsupported}")
    
    return {"unsupported_claims": unsupported}
