# src/kref_rag/indexing/vector_store.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class ChromaStore:
    def __init__(self, persist_dir: str, collection_name: str, embedder: SentenceTransformer):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder = embedder

    def upsert(self, chunks):
        if not chunks:
            logger.warning("No chunks to upsert.")
            return
        ids = [c["chunk_id"] for c in chunks]
        docs = [c["text"] for c in chunks]
        metas = [{
            "doc_id": c["doc_id"],
            "title": c.get("title", ""),
            "url": c.get("url", ""),
            "published_at": c.get("published_at", ""),
            "source": c.get("source", ""),
        } for c in chunks]

        embs = self.embedder.encode(
        docs, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs.tolist())

    def query(self, query: str, top_k: int = 6, where: Optional[dict] = None) -> List[Dict[str, Any]]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        res = self.collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=top_k,
            where=where
        )
        out = []
        for i in range(len(res["ids"][0])):
            out.append({
                "chunk_id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "meta": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            })
        return out
