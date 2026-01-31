# src/kref_rag/main.py
from __future__ import annotations
import argparse
import json
import requests
from pathlib import Path

from sentence_transformers import SentenceTransformer

from kref_rag import config
from kref_rag.utils.logger import setup_logger
from kref_rag.ingest.rss_fetcher import fetch_rss_entries
from kref_rag.ingest.sources.bok import fetch_bok_article
from kref_rag.indexing.chunker import docs_to_chunks
from kref_rag.indexing.vector_store import ChromaStore



def cmd_ingest(args):
    run_id = config.make_run_id()
    run_dir = config.ensure_run_dir(run_id)
    logger = setup_logger(run_dir / "run.log")

    entries = fetch_rss_entries(config.BOK_RSS_URL, limit=args.limit)

    docs = []
    seen = set()

    session = requests.Session()  
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; kref-rag/1.0; +https://example.local)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })

    for e in entries:
        url = e.get("url") or ""
        if not url or url in seen:
            continue
        seen.add(url)

        try:
            art = fetch_bok_article(url, session=session)

            # fetch 실패면 스킵
            if not art.get("success", False):
                logger.warning(f"skipped (fetch failed): {e.get('title','')} | {url} | {art.get('reason','')}")
                continue

            content = (art.get("content_text") or "").strip()

            # 본문 너무 짧으면 스킵 (UI/빈 페이지 방지)
            if len(content) < 300:
                logger.warning(
                    f"skipped (too short): {e.get('title','')} | {url} | len={len(content)} | src={art.get('source_type','')}"
                )
                continue

            docs.append({
                **e,
                "title": (e.get("title") or art.get("page_title") or "").strip(),
                "content_text": content,
                "attachments": art.get("attachments", []),
                "source_type": art.get("source_type", ""),
                "success": True,
            })
            logger.info(f"ingested: {e.get('title','')} | src={art.get('source_type','')}")

        except Exception as ex:
            logger.exception(f"failed: {url} - {ex}")

    out = run_dir / "ingest_raw.json"
    out.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"saved: {out}")
    print(run_id)

def cmd_index(args):
    run_dir = config.ARTIFACTS_DIR / args.run_id
    logger = setup_logger(run_dir / "index.log")
    logger.info(f"Embed device: {config.device_summary()} | model: {config.EMBED_MODEL}")

    docs = json.loads((run_dir / "ingest_raw.json").read_text(encoding="utf-8"))
    chunks = docs_to_chunks(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    logger.info(f"chunks: {len(chunks)}")

    embedder = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    collection = f"kref_bok_{args.run_id}"
    store = ChromaStore(str(config.CHROMA_DIR), collection, embedder)
    store.upsert(chunks)

    (run_dir / "chunks_count.txt").write_text(str(len(chunks)), encoding="utf-8")
    logger.info("index complete")

def cmd_run(args):
    from kref_rag.rag.graph import build_graph
    run_dir = config.ARTIFACTS_DIR / args.run_id
    logger = setup_logger(run_dir / "run_query.log")
    logger.info(f"Embed device: {config.device_summary()} | model: {config.EMBED_MODEL}")

    embedder = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    collection = f"kref_bok_{args.run_id}"
    store = ChromaStore(str(config.CHROMA_DIR), collection, embedder)

    graph = build_graph(store)
    state = {"query": args.query, "top_k": args.top_k}
    out_state = graph.invoke(state)

    report_md = out_state["report_md"]
    (run_dir / "report.md").write_text(report_md, encoding="utf-8")

    # 디버깅용 저장
    (run_dir / "retrieved_chunks.json").write_text(
        json.dumps(out_state.get("retrieved", []), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    logger.info("report generated")
    print(f"saved report: {run_dir / 'report.md'}")

def cmd_viz(args):
    """시각화 명령어"""
    from kref_rag.viz import visualize_graph
    
    run_dir = config.ARTIFACTS_DIR / args.run_id
    
    if not run_dir.exists():
        print(f"[ERROR] run_id를 찾을 수 없습니다: {args.run_id}")
        print(f"경로: {run_dir}")
        return
    
    try:
        output_path = visualize_graph(run_dir, output_filename=args.output)
        if output_path:
            print(f"\n[OK] 시각화 완료!")
            print(f"[FILE] {output_path}")
            print(f"[BROWSER] file:///{output_path}")
    except Exception as e:
        print(f"[ERROR] 시각화 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("ingest")
    p1.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT)
    p1.set_defaults(func=cmd_ingest)

    p2 = sub.add_parser("index")
    p2.add_argument("--run-id", required=True)
    p2.set_defaults(func=cmd_index)

    p3 = sub.add_parser("run")
    p3.add_argument("--run-id", required=True)
    p3.add_argument("--query", required=True)
    p3.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    p3.set_defaults(func=cmd_run)

    p4 = sub.add_parser("viz", help="Graph of Thoughts 시각화")
    p4.add_argument("--run-id", required=True, help="시각화할 run-id")
    p4.add_argument("--output", default="graph.html", help="출력 HTML 파일명")
    p4.set_defaults(func=cmd_viz)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
