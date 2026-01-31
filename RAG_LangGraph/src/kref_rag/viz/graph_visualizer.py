# src/kref_rag/viz/graph_visualizer.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False


def visualize_graph(
    run_dir: Path,
    output_filename: str = "graph.html"
) -> Path:
    """
    Graph of Thoughts 시각화 (Pyvis HTML)
    
    Args:
        run_dir: 실행 결과 디렉토리 (report.md, retrieved_chunks.json 있는 곳)
        output_filename: 출력 HTML 파일명
    
    Returns:
        생성된 HTML 파일 경로
    """
    if not PYVIS_AVAILABLE:
        raise ImportError(
            "pyvis가 설치되지 않았습니다. "
            "터미널에서 'pip install pyvis'를 실행하세요."
        )
    
    print(f"[viz] 시각화 시작: {run_dir}")
    
    # report.md 파싱 (Claims, Edges 추출)
    report_path = run_dir / "report.md"
    if not report_path.exists():
        raise FileNotFoundError(f"report.md를 찾을 수 없습니다: {report_path}")
    
    claims, edges = _parse_report(report_path)
    
    if not claims:
        print("[viz] ⚠️ Claims가 없습니다. Day 2, 3 코드가 실행되지 않았을 수 있습니다.")
        return None
    
    print(f"[viz] Claims: {len(claims)}개, Edges: {len(edges)}개")
    
    # retrieved_chunks.json 로드 (Evidence 상세 정보)
    chunks_path = run_dir / "retrieved_chunks.json"
    evidence_details = {}
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
            for chunk in chunks_data:
                chunk_id = chunk.get("chunk_id", "")
                evidence_details[chunk_id] = {
                    "title": chunk.get("meta", {}).get("title", "제목 없음"),
                    "text": chunk.get("text", "")[:100] + "..."
                }
    
    # Pyvis 그래프 생성
    net = Network(
        height="800px",
        width="100%",
        directed=True,
        bgcolor="#f8f9fa",
        font_color="#333333"
    )
    
    # 물리 엔진 설정 (노드 간격, 중력 등)
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "springLength": 200,
                "springConstant": 0.04
            }
        },
        "nodes": {
            "font": {"size": 14, "face": "맑은 고딕, Arial"}
        },
        "edges": {
            "arrows": {"to": {"enabled": true}},
            "smooth": {"type": "continuous"}
        }
    }
    """)
    
    # Claim 노드 추가 (빨간색, 큰 크기)
    for claim in claims:
        claim_id = claim["id"]
        claim_text = claim["text"]
        
        # 노드 라벨 (최대 60자)
        label = f"{claim_id}\n{claim_text[:60]}..."
        
        # 툴팁 (전체 텍스트)
        title = f"<b>{claim_id}</b><br>{claim_text}"
        
        net.add_node(
            claim_id,
            label=label,
            title=title,
            color="#ff6b6b",  # 빨간색
            size=30,
            shape="box",
            font={"size": 16, "bold": True}
        )
    
    # Evidence 노드 및 엣지 추가
    evidence_nodes_added = set()
    
    for edge in edges:
        claim_id = edge["from"]
        evidence_id = edge["to"]
        similarity = edge["similarity"]
        
        # Evidence 노드 추가 (중복 방지)
        if evidence_id not in evidence_nodes_added:
            # Evidence 상세 정보
            details = evidence_details.get(evidence_id, {})
            title_text = details.get("title", "제목 없음")
            snippet = details.get("text", "내용 없음")
            
            # 노드 라벨
            label = f"E\n{title_text[:30]}..."
            
            # 툴팁
            title_html = f"<b>Evidence</b><br>{title_text}<br><br>{snippet}"
            
            net.add_node(
                evidence_id,
                label=label,
                title=title_html,
                color="#4dabf7",  # 파란색
                size=20,
                shape="ellipse"
            )
            evidence_nodes_added.add(evidence_id)
        
        # 엣지 추가 (Claim → Evidence)
        net.add_edge(
            claim_id,
            evidence_id,
            label=f"{similarity:.3f}",
            title=f"유사도: {similarity:.3f}",
            width=similarity * 5,  # 유사도에 따라 선 굵기 조정
            color=_get_edge_color(similarity)
        )
    
    # HTML 저장
    output_path = run_dir / output_filename
    net.save_graph(str(output_path))
    
    print(f"[viz] OK 시각화 완료: {output_path}")
    print(f"[viz] 브라우저로 열기: file:///{output_path}")
    
    return output_path


def _parse_report(report_path: Path) -> tuple[List[Dict], List[Dict]]:
    """
    report.md 파싱하여 Claims와 Edges 추출
    
    Returns:
        (claims, edges)
    """
    claims = []
    edges = []
    
    with open(report_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    current_claim = None
    in_claims_section = False
    
    for line in lines:
        line = line.strip()
        
        # Claims 섹션 시작
        if line == "## Claims":
            in_claims_section = True
            continue
        
        # Claims 섹션 종료
        if in_claims_section and line.startswith("## ") and line != "## Claims":
            in_claims_section = False
            continue
        
        if not in_claims_section:
            continue
        
        # Claim 제목 (### C1: ...)
        if line.startswith("### C"):
            parts = line[4:].split(":", 1)
            if len(parts) == 2:
                claim_id = parts[0].strip()
                claim_text = parts[1].strip()
                
                # ⚠️ 제거
                if " ⚠️" in claim_text:
                    claim_text = claim_text.replace(" ⚠️ (근거 없음)", "")
                
                current_claim = {
                    "id": claim_id,
                    "text": claim_text
                }
                claims.append(current_claim)
        
        # Evidence 엣지 (- chunk_id (유사도: 0.xxx))
        if current_claim and line.startswith("- ") and "(유사도:" in line:
            # 예: "- 91b54b53dff6df10705fa6cb128aff02d158105a (유사도: 0.754)"
            parts = line[2:].split(" (유사도: ")
            if len(parts) == 2:
                chunk_id = parts[0].strip()
                similarity_str = parts[1].rstrip(")")
                try:
                    similarity = float(similarity_str)
                    edges.append({
                        "from": current_claim["id"],
                        "to": chunk_id,
                        "similarity": similarity
                    })
                except ValueError:
                    pass
    
    return claims, edges


def _get_edge_color(similarity: float) -> str:
    """유사도에 따른 엣지 색상"""
    if similarity >= 0.8:
        return "#2ecc71"  # 초록 (매우 높음)
    elif similarity >= 0.7:
        return "#3498db"  # 파랑 (높음)
    elif similarity >= 0.6:
        return "#f39c12"  # 주황 (보통)
    else:
        return "#e74c3c"  # 빨강 (낮음)

