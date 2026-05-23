"""
retriever.py — Enhanced Dual-Branch Evidence Retrieval
=======================================================
Cải tiến so với phiên bản cũ:
  1. Multi-query: 2-3 truy vấn mục tiêu thay vì 1 query chung
  2. Claim-aware snippet: tách câu hoàn chỉnh, ưu tiên câu có số liệu/claim
  3. Domain diversity: đảm bảo kết quả từ nhiều domain khác nhau
  4. TF-IDF re-rank: score lại toàn bộ kết quả web bằng cosine similarity
  5. Result merging: gộp kết quả từ nhiều query, dedup theo URL
  6. Crawl fallback: nếu crawl thất bại dùng DDG snippet

Nhánh 1 — BM25 (BKAI NewsCategory hoặc ViFN fallback)
Nhánh 2 — Web RAG (DuckDuckGo → báo Việt uy tín)
"""

import re
import time
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import pandas as pd

from query_builder import (
    build_search_queries,
    build_best_snippet_sentence_aware,
    _STOPWORDS as _QB_STOPWORDS,
)

logger = logging.getLogger(__name__)

# ─── Thư viện tuỳ chọn ───────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

try:
    from ddgs import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    _DDGS_AVAILABLE = False

try:
    from datasets import load_dataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


# ════════════════════════════════════════════════════════════
# CẤU HÌNH
# ════════════════════════════════════════════════════════════
BKAI_DATASET_ID  = "bkai-foundation-models/NewsCategory"
BKAI_MAX_SAMPLES = 30_000

_VIFN_CORPUS_PATH = (
    Path(__file__).parent.parent
    / "ViFN-Vietnamese_Fake_New_Datasets_Ver3"
    / "processed"
    / "train_cleaned.csv"
)

VI_NEWS_DOMAINS = [
    "tuoitre.vn",
    "thanhnien.vn",
    "vnexpress.net",
    "dantri.com.vn",
    "baochinhphu.vn",
    "vietnamnet.vn",
    "baomoi.com",
    "nhandan.vn",
    "tienphong.vn",
    "laodong.vn",
]

WEB_SEARCH_MAX_RESULTS = 6   # Per query
WEB_CRAWL_TIMEOUT      = 8
WEB_CRAWL_MAX_WORKERS  = 4
WEB_SNIPPET_MAX_WORDS  = 100


# ════════════════════════════════════════════════════════════
# TOKENIZER
# ════════════════════════════════════════════════════════════
_VI_STOPWORDS = _QB_STOPWORDS | {
    "ông", "bà", "anh", "chị", "người", "việt", "nam", "quốc",
}

def _tokenize_vi(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if len(t) >= 2 and t not in _VI_STOPWORDS]


def _highlight_keywords(snippet: str, query_tokens: List[str]) -> str:
    query_set = set(t.lower() for t in query_tokens)
    words = snippet.split()
    out = []
    for w in words:
        clean = re.sub(r"[^\w]", "", w.lower())
        out.append(f"**{w}**" if clean in query_set and clean else w)
    return " ".join(out)


# ════════════════════════════════════════════════════════════
# TF-IDF COSINE RE-RANKER (nhẹ, không cần sklearn)
# ════════════════════════════════════════════════════════════

def _tfidf_cosine(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """
    Cosine similarity đơn giản dựa trên TF (term frequency overlap).
    Không cần thư viện ngoài.
    """
    if not query_tokens or not doc_tokens:
        return 0.0
    q_set = set(query_tokens)
    d_set = set(doc_tokens)
    intersection = q_set & d_set
    if not intersection:
        return 0.0
    # TF-IDF đơn giản: tf overlap / sqrt(|q| * |d|)
    import math
    return len(intersection) / math.sqrt(len(q_set) * len(d_set))


# ════════════════════════════════════════════════════════════
# PHẦN 1: BM25 LOCAL CORPUS
# ════════════════════════════════════════════════════════════
_corpus_cache: Dict[str, Any] = {}


def _load_bkai_corpus_hf(max_samples: int = BKAI_MAX_SAMPLES) -> List[Dict]:
    if not _HF_AVAILABLE:
        return []
    try:
        logger.info(f"Loading BKAI corpus (streaming {max_samples} samples)...")
        ds = load_dataset(BKAI_DATASET_ID, split="train", streaming=True, trust_remote_code=True)
        records = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            title   = str(row.get("title") or "").strip()
            sapo    = str(row.get("sapo")  or "").strip()
            category = str(row.get("label") or "").strip()
            combined = f"{title}. {sapo}".strip(". ")
            if combined and len(combined.split()) >= 5:
                records.append({"text": combined, "title": title,
                                 "category": category, "source": "bkai_vnexpress"})
        logger.info(f"BKAI loaded: {len(records)} records")
        return records
    except Exception as e:
        logger.error(f"BKAI load failed: {e}")
        return []


def _load_vifn_fallback() -> List[Dict]:
    if not _VIFN_CORPUS_PATH.exists():
        return []
    try:
        df = pd.read_csv(str(_VIFN_CORPUS_PATH), encoding="utf-8-sig")
        # Hỗ trợ cả tên cột cũ (clean_text) và mới (text)
        text_col = "text" if "text" in df.columns else "clean_text"
        if text_col not in df.columns:
            return []
        df = df.dropna(subset=[text_col])
        return [
            {"text": str(r[text_col]), "title": str(r.get("title", "")),
             "category": "ViFN", "source": "vifn_local", "label": int(r.get("label", 0))}
            for _, r in df.iterrows()
        ]
    except Exception as e:
        logger.error(f"ViFN load failed: {e}")
        return []


def load_corpus() -> Dict[str, Any]:
    key = "bkai_primary"
    if key in _corpus_cache:
        return _corpus_cache[key]

    records = _load_bkai_corpus_hf() or _load_vifn_fallback()
    if not records:
        result = {"bm25": None, "records": [], "corpus_name": "empty", "total": 0}
        _corpus_cache[key] = result
        return result

    tokenized = [_tokenize_vi(r["text"]) for r in records]
    bm25 = BM25Okapi(tokenized) if _BM25_AVAILABLE and tokenized else None
    corpus_name = "bkai_vnexpress" if records[0].get("source") == "bkai_vnexpress" else "vifn_local"
    result = {"bm25": bm25, "records": records, "corpus_name": corpus_name, "total": len(records)}
    _corpus_cache[key] = result
    return result


def retrieve_bm25(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """BM25 search trên BKAI/ViFN corpus. Dùng full query tokens."""
    corpus = load_corpus()
    if not corpus["bm25"] or not corpus["records"]:
        return []

    query_tokens = _tokenize_vi(query)
    if not query_tokens:
        return []

    scores = corpus["bm25"].get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results = []
    for rank_i, (idx, score) in enumerate(ranked[:top_k], start=1):
        if score <= 0:
            break
        rec = corpus["records"][idx]
        snippet = build_best_snippet_sentence_aware(rec["text"], query_tokens, max_words=80)
        item = {
            "rank": rank_i,
            "score": float(score),
            "title": rec.get("title", "").strip(),
            "category": rec.get("category", ""),
            "snippet": snippet,
            "snippet_highlighted": _highlight_keywords(snippet, query_tokens),
            "source": rec.get("source", "bkai_vnexpress"),
            "url": None,
            "retrieval_type": "bm25_local",
            "label": rec.get("label"),
            "label_str": ("FAKE" if rec.get("label") == 1 else "REAL") if "label" in rec else "N/A",
        }
        results.append(item)

    return results


# ════════════════════════════════════════════════════════════
# PHẦN 2: WEB RAG
# ════════════════════════════════════════════════════════════

def _crawl_page(url: str, query_tokens: List[str]) -> Optional[str]:
    """Crawl URL, trả về best-sentence-aware snippet."""
    if not (_REQUESTS_AVAILABLE and _BS4_AVAILABLE):
        return None
    headers = {
        "User-Agent": random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15",
        ]),
        "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    }
    try:
        resp = _requests.get(url, headers=headers, timeout=WEB_CRAWL_TIMEOUT)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        paras = [
            p.get_text(" ", strip=True) for p in soup.find_all("p")
            if len(p.get_text().split()) > 8
        ]
        if not paras:
            return None
        full = re.sub(r"http\S+|www\.\S+|\s+", " ", " ".join(paras)).strip()
        snip = build_best_snippet_sentence_aware(full, query_tokens, max_words=WEB_SNIPPET_MAX_WORDS)
        return snip if len(snip.split()) >= 10 else None
    except Exception:
        return None


def _ddg_search_single(query: str, domains: List[str], max_results: int) -> List[Dict]:
    """1 DDG call với site: operator."""
    if not _DDGS_AVAILABLE:
        return []
    site_op = "(" + " OR ".join(f"site:{d}" for d in domains) + ")"
    full_q = f"{query} {site_op}"
    results = []
    try:
        with DDGS(timeout=15) as ddgs:
            for r in ddgs.text(full_q, max_results=max_results):
                url = str(r.get("href", r.get("url", ""))).strip()
                if url:
                    results.append({
                        "title": str(r.get("title", "")).strip(),
                        "url": url,
                        "snippet": str(r.get("body", "")).strip(),
                    })
    except Exception as e:
        logger.warning(f"DDG failed for '{query[:40]}': {e}")
    return results


def _multi_query_ddg_search(
    queries: List[Tuple[str, str]],
    domains: List[str],
    max_per_query: int = 5,
) -> List[Dict]:
    """
    Gọi DDG cho từng query, gộp kết quả, dedup theo URL.
    Đảm bảo domain diversity: không quá 2 kết quả cùng domain.
    """
    all_raw: Dict[str, Dict] = {}  # url → result (dedup)
    domain_count: Dict[str, int] = {}

    for query, qtype in queries:
        raw = _ddg_search_single(query, domains, max_results=max_per_query)
        for r in raw:
            url = r["url"]
            if url in all_raw:
                continue
            # Domain diversity: tối đa 2 kết quả / domain
            try:
                dom = urlparse(url).netloc.replace("www.", "")
            except Exception:
                dom = ""
            if domain_count.get(dom, 0) >= 2:
                continue
            domain_count[dom] = domain_count.get(dom, 0) + 1
            all_raw[url] = {**r, "query_type": qtype, "domain": dom}

    return list(all_raw.values())


def search_web_evidence(
    query: str,
    top_k: int = 4,
    domains: List[str] = None,
    crawl: bool = True,
) -> List[Dict[str, Any]]:
    """
    Web RAG nâng cao:
    1. Tạo 2-3 queries mục tiêu (entity, claim, broad)
    2. DDG multi-query với domain diversity
    3. Crawl song song (optional)
    4. TF-IDF cosine re-rank
    5. Trả về top_k kết quả tốt nhất
    """
    if not _DDGS_AVAILABLE:
        return []

    _domains = domains or VI_NEWS_DOMAINS

    # Bước 1: Build targeted queries
    targeted_queries = build_search_queries(query)
    if not targeted_queries:
        return []

    # Bước 2: Multi-query DDG search
    raw_results = _multi_query_ddg_search(targeted_queries, _domains, max_per_query=5)
    if not raw_results:
        return []

    query_tokens = _tokenize_vi(query)

    # Bước 3: Crawl song song (optional)
    processed: List[Dict] = []
    if crawl and _REQUESTS_AVAILABLE and _BS4_AVAILABLE:
        with ThreadPoolExecutor(max_workers=WEB_CRAWL_MAX_WORKERS) as executor:
            future_map = {
                executor.submit(_crawl_page, r["url"], query_tokens): r
                for r in raw_results
            }
            for future in as_completed(future_map):
                r = future_map[future]
                try:
                    crawled = future.result()
                except Exception:
                    crawled = None
                snippet = crawled or r["snippet"]
                if snippet and len(snippet.split()) >= 8:
                    processed.append({**r, "_final_snippet": snippet})
    else:
        for r in raw_results:
            if r.get("snippet"):
                processed.append({**r, "_final_snippet": r["snippet"]})

    if not processed:
        return []

    # Bước 4: TF-IDF cosine re-rank
    for p in processed:
        doc_tokens = _tokenize_vi(p["_final_snippet"])
        p["_score"] = _tfidf_cosine(query_tokens, doc_tokens)

    processed.sort(key=lambda x: x["_score"], reverse=True)

    # Bước 5: Format output
    results = []
    for rank_i, p in enumerate(processed[:top_k], start=1):
        snippet = p["_final_snippet"]
        highlighted = _highlight_keywords(snippet, query_tokens)
        results.append({
            "rank": rank_i,
            "score": round(p["_score"], 4),
            "title": p["title"],
            "url": p["url"],
            "snippet": snippet,
            "snippet_highlighted": highlighted,
            "source": "web_rag",
            "domain": p.get("domain", ""),
            "query_type": p.get("query_type", ""),
            "label": None,
            "label_str": "WEB",
            "retrieval_type": "web_rag",
        })

    return results


# ════════════════════════════════════════════════════════════
# PHẦN 3: HÀM CHÍNH
# ════════════════════════════════════════════════════════════

def retrieve_evidence(
    query: str,
    top_k_bm25: int = 4,
    top_k_web: int = 3,
    enable_web: bool = True,
    web_domains: List[str] = None,
    crawl_web: bool = True,
) -> Dict[str, Any]:
    """
    Kết hợp BM25 local + Web RAG → trả về dict đầy đủ.
    """
    bm25_results = retrieve_bm25(query, top_k=top_k_bm25)

    web_results = []
    if enable_web:
        web_results = search_web_evidence(
            query, top_k=top_k_web, domains=web_domains, crawl=crawl_web
        )

    all_results = [
        {**r, "global_rank": i} for i, r in enumerate(bm25_results, 1)
    ] + [
        {**r, "global_rank": len(bm25_results) + i} for i, r in enumerate(web_results, 1)
    ]

    corpus = load_corpus()
    return {
        "bm25": bm25_results,
        "web": web_results,
        "all": all_results,
        "corpus": corpus.get("corpus_name", "unknown"),
        "corpus_total": corpus.get("total", 0),
        "queries_used": build_search_queries(query) if enable_web else [],
    }


def analyze_retrieved_distribution(evidences_or_result) -> Dict[str, Any]:
    """Phân tích phân phối nhãn (backward compat)."""
    if isinstance(evidences_or_result, dict):
        items = evidences_or_result.get("all", [])
    else:
        items = evidences_or_result

    labeled   = [e for e in items if e.get("label") is not None]
    real_count = sum(1 for e in labeled if e["label"] == 0)
    fake_count = sum(1 for e in labeled if e["label"] == 1)
    web_count  = sum(1 for e in items if e.get("retrieval_type") == "web_rag")
    bkai_count = sum(1 for e in items if e.get("source") == "bkai_vnexpress")
    total = len(items)

    if labeled:
        dominant  = "REAL" if real_count >= fake_count else "FAKE"
        agreement = max(real_count, fake_count) / len(labeled)
    else:
        dominant  = "UNKNOWN"
        agreement = 0.0

    return {
        "real_count": real_count,
        "fake_count": fake_count,
        "labeled_count": len(labeled),
        "web_count": web_count,
        "bkai_count": bkai_count,
        "total": total,
        "dominant_label": dominant,
        "agreement_ratio": agreement,
    }
