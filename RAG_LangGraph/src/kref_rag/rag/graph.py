# src/kref_rag/rag/graph.py
from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from .state import GraphState
from .nodes import (
    node_retrieve,
    node_extract_claims,
    node_match_evidence,
    node_validate,
    node_generate_report
)


def build_graph(store):
    """LangGraph 구성 (Day 2 & 3: GoT 워크플로우)"""
    g = StateGraph(GraphState)
    
    # 노드 추가
    g.add_node("retrieve", lambda s: node_retrieve(store, s))
    g.add_node("extract_claims", node_extract_claims)  # Day 2
    g.add_node("match_evidence", lambda s: node_match_evidence(s, store))  # Day 3
    g.add_node("validate", node_validate)  # Day 3
    g.add_node("generate", node_generate_report)
    
    # 엣지 연결
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "extract_claims")
    g.add_edge("extract_claims", "match_evidence")
    g.add_edge("match_evidence", "validate")
    g.add_edge("validate", "generate")
    g.add_edge("generate", END)
    
    return g.compile()
