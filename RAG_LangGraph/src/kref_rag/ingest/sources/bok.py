# src/kref_rag/ingest/sources/bok.py
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# -----------------------------
# Public API (keep signature)
# -----------------------------
def fetch_bok_article(url: str, session: Optional[requests.Session] = None, timeout: int = 15):
    session = session or _get_session()
    source_type = "html"
    try:
        html = _fetch_html(session, url, timeout=timeout)
        soup = BeautifulSoup(html, "html.parser")
        page_title = soup.title.get_text(strip=True) if soup.title else ""

        html_text = _extract_main_text(soup)
        attachments = _extract_attachments(soup, base_url=url)

        content_text = html_text
        logger.debug(f"HTML text extracted: len={len(content_text)} chars")
        
        if _looks_like_empty_body(content_text):
            logger.info(f"HTML body looks empty/incomplete, trying PDF extraction ({len(attachments)} attachments)")
            pdf_text = _try_extract_text_from_attachments(
                session=session,
                attachment_urls=attachments,
                timeout=timeout,
            )
            if pdf_text:
                content_text = pdf_text
                source_type = "pdf"
                logger.info(f"✓ Using PDF content: len={len(pdf_text)} chars")
            else:
                logger.warning("✗ PDF extraction failed, keeping HTML content")

        if not (content_text or "").strip():
            logger.warning("Content is empty, using fallback (full page text)")
            content_text = _clean_text(soup.get_text("\n", strip=True))
            source_type = "fallback"

        return {
            "page_title": page_title,
            "content_text": content_text,
            "attachments": attachments,
            "success": True,
            "source_type": source_type,
        }

    except Exception as e:
        logger.exception("fetch_bok_article failed: %s", url)
        err = str(e)
        return {
            "page_title": "",
            "content_text": "",
            "attachments": [],
            "success": False,
            "error": err,
            "reason": err,          
            "source_type": "error",
        }



def fetch_bok_rss(*args, **kwargs):
    raise RuntimeError(
        "fetch_bok_rss() is deprecated. Use kref_rag.ingest.rss_fetcher.fetch_rss_entries() instead."
    )


# -----------------------------
# Session / Fetch
# -----------------------------
_SESSION: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )
        _SESSION = s
    return _SESSION


def _fetch_html(session: requests.Session, url: str, timeout: int) -> str:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    # 인코딩 깨짐 방지
    if not r.encoding:
        r.encoding = r.apparent_encoding or "utf-8"
    return r.text


# -----------------------------
# Main text extraction (HTML)
# -----------------------------
_MAIN_SELECTORS: Sequence[str] = (
    "article",
    "main",
    "div#content",
    "div.contents",
    "div.cont",
    "div.view",
    "div.view-con",
    "div.board-view",
    "div.bbs-view",
    "div.tbl", 
)


def _extract_main_text(soup: BeautifulSoup) -> str:
    # script/style 제거
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    candidates: list[str] = []

    for sel in _MAIN_SELECTORS:
        node = soup.select_one(sel)
        if not node:
            continue
        txt = _clean_text(node.get_text("\n", strip=True))
        if txt:
            candidates.append(txt)

    # 후보가 여러 개면 “가장 긴 텍스트”를 본문으로 채택
    if candidates:
        best = max(candidates, key=len)
        return best

    # fallback
    return _clean_text(soup.get_text("\n", strip=True))


def _remove_header_metadata(text: str) -> str:
    """페이지 상단 메타데이터 제거"""
    lines = text.split('\n')
    
    # 제거할 메타데이터 키워드
    metadata_keywords = ["구분", "등록일", "조회수", "키워드", "담당부서", "첨부파일", "통계보기", "다운로드", "뷰어"]
    
    # 실제 본문 시작점 찾기
    start_idx = 0
    metadata_count = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # 메타데이터 라인 카운트
        if any(kw in line for kw in metadata_keywords):
            metadata_count += 1
            continue
        
        # 메타데이터가 2개 이상 나온 후 첫 번째 실제 내용 (20자 이상)
        if metadata_count >= 2 and line and len(line) > 20:
            start_idx = i
            break
    
    return '\n'.join(lines[start_idx:])


def _truncate_footer(text: str) -> str:
    """페이지 하단 UI 텍스트 제거"""
    cutoff_markers = [
        "\n목록",
        "\n연관자료",
        "\n관련 뉴스/자료",
        "\n관련 미디어 자료",
        "\n유용한 정보가 되었나요?",
        "\n내가 본 콘텐츠",
        "\n메뉴",
    ]
    
    cut = len(text)
    for marker in cutoff_markers:
        pos = text.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    
    return text[:cut].strip()


def _clean_text(text: str) -> str:
    t = (text or "").replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.strip()
    t = _remove_header_metadata(t)
    t = _truncate_footer(t)
    return t


def _looks_like_empty_body(text: str) -> bool:
    """본문이 비어있거나 의미없는 경우 True 반환 → PDF 파싱 시도"""
    t = (text or "").strip()
    
    # 1. 텍스트가 너무 짧으면 비어있음
    if len(t) < 300:
        return True
    
    # 2. "첨부파일 참조/참고" 문구가 있으면 본문 없음
    fallback_phrases = [
        "자세한 내용은 첨부파일을 참조",
        "자세한 내용은 첨부파일을 참고",
        "첨부파일을 참조",
        "첨부파일을 참고",
        "첨부된 파일을 확인",
        "붙임과 같이",
    ]
    if any(p in t for p in fallback_phrases):
        return True
    
    # 3. 네비/메뉴/버튼 문구가 많은 경우 노이즈로 판단
    junk_signals = [
        "메뉴", "본문", "다운로드", "이전", "다음", "목록", "바로가기",
        "유용한 정보가 되었나요", "매우만족", "담당부서", "전화번호",
        "내가 본 콘텐츠", "페이지 위로", "확인", "연관자료"
    ]
    hit = sum(1 for k in junk_signals if k in t)
    
    # threshold 강화: 3개 이상이면 비어있다고 판단
    if hit >= 3:
        return True
    
    # 4. 실제 문장이 거의 없는 경우 (마침표 개수로 판단)
    sentence_count = t.count('.') + t.count('다.') + t.count('다 .')
    if sentence_count < 3 and len(t) < 1000:
        return True
    
    return False


# -----------------------------
# Attachments discovery
# -----------------------------
_FILE_EXTS = (".pdf", ".hwp", ".hwpx")


def _extract_attachments(soup: BeautifulSoup, base_url: str) -> list[str]:
    """
    기존 endswith(.pdf)만으로는 fileDown.do 같은 링크가 누락될 수 있어
    - 확장자 기반
    - filedown/download 패턴 기반
    둘 다 포함.
    """
    links: list[str] = []

    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue

        full = urljoin(base_url, href)
        low = full.lower()

        # 1) 확장자 기반
        if any(low.endswith(ext) for ext in _FILE_EXTS):
            links.append(full)
            continue

        # 2) 확장자 없어도 다운로드 엔드포인트면 후보
        if ("filedown" in low) or ("download" in low):
            links.append(full)
            continue

    # 중복 제거(순서 유지)
    return list(dict.fromkeys(links))


# -----------------------------
# PDF text extraction
# -----------------------------
@dataclass(frozen=True)
class _PdfExtractConfig:
    min_text_len: int = 100  # 100자 이상이면 유효한 PDF 본문으로 간주
    max_bytes: int = 20 * 1024 * 1024  # 20MB 안전장치


def _try_extract_text_from_attachments(
    session: requests.Session,
    attachment_urls: Sequence[str],
    timeout: int,
    cfg: _PdfExtractConfig = _PdfExtractConfig(),
) -> Optional[str]:
    if not attachment_urls:
        logger.debug("No attachments found")
        return None

    for u in attachment_urls:
        # PDF만 우선
        if not _looks_like_pdf(u):
            continue

        logger.info(f"Trying to extract PDF: {u[:80]}...")
        text = _download_and_extract_pdf_text(
            session=session,
            pdf_url=u,
            timeout=timeout,
            cfg=cfg,
        )
        text_len = len(text) if text else 0
        logger.info(f"PDF extracted: len={text_len} chars")
        
        if text and text_len >= cfg.min_text_len:
            logger.info(f"✓ PDF text accepted (>= {cfg.min_text_len} chars)")
            return text
        elif text:
            logger.warning(f"✗ PDF text too short: {text_len} < {cfg.min_text_len}")
        else:
            logger.warning("✗ PDF extraction returned empty text")

    
    logger.debug("Trying fallback: non-PDF URLs with content-type check")
    for u in attachment_urls:
        if _looks_like_pdf(u):
            continue
        logger.debug(f"Fallback trying: {u[:80]}...")
        text = _download_and_extract_pdf_text(
            session=session,
            pdf_url=u,
            timeout=timeout,
            cfg=cfg,
            allow_non_pdf_url=True,
        )
        text_len = len(text) if text else 0
        if text and text_len >= cfg.min_text_len:
            logger.info(f"✓ Fallback PDF text accepted: len={text_len}")
            return text

    logger.warning("No valid PDF text found in attachments")
    return None


def _looks_like_pdf(url: str) -> bool:
    return (url or "").lower().endswith(".pdf")


def _download_and_extract_pdf_text(
    session: requests.Session,
    pdf_url: str,
    timeout: int,
    cfg: _PdfExtractConfig,
    allow_non_pdf_url: bool = False,
) -> Optional[str]:
    try:
        r = session.get(pdf_url, timeout=timeout, stream=True)
        r.raise_for_status()

        ctype = (r.headers.get("Content-Type") or "").lower()

        
        if (not allow_non_pdf_url) and ("pdf" not in ctype) and (not pdf_url.lower().endswith(".pdf")):
            return None

        data = r.content
        if len(data) > cfg.max_bytes:
            logger.warning("PDF too large (%d bytes): %s", len(data), pdf_url)
            return None

    
        text = _extract_with_pdfplumber(data)
        if text and text.strip():
            return _clean_text(text)

        
        text = _extract_with_pypdf(data)
        if text and text.strip():
            return _clean_text(text)

        return None

    except Exception:
        logger.debug("PDF extraction failed: %s", pdf_url, exc_info=True)
        return None


def _extract_with_pdfplumber(pdf_bytes: bytes) -> Optional[str]:
    try:
        import pdfplumber  # type: ignore
    except Exception:
        return None

    try:
        out: list[str] = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                if txt.strip():
                    out.append(txt)
        merged = "\n\n".join(out).strip()
        return merged or None
    except Exception:
        return None


def _extract_with_pypdf(pdf_bytes: bytes) -> Optional[str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return None

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        out: list[str] = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                out.append(txt)
        merged = "\n\n".join(out).strip()
        return merged or None
    except Exception:
        return None
