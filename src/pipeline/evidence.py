"""
Evidence bundle construction and dual-model inference.
Handles round-aware demonstration retrieval and LLM+SLM assessment.
"""

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import (
    TOP_K_DEMOS,
    FACT_TOP_K,
    LLM_MAX_OUTPUT_TOKENS_CLASSIFICATION,
    KNOWLEDGE_MODE,
)
from src.utils import clean_text_transformer
from src.labels import parse_llm_label, to_clean_demo_label
from src.prompts import build_classification_prompt
from src.retrieval.demo_retrieval import search_news, retrieve_demonstrations
from src.retrieval.knowledge_agent import (
    build_knowledge_bundle,
    get_cached_knowledge_bundle_local,
)


# ================================================================
# Semantic Encoder Singleton (multilingual-e5-small)
# ================================================================
_demo_encoder = None


def _get_demo_encoder():
    """Lazy-load sentence encoder dùng cho hybrid retrieval."""
    global _demo_encoder
    if _demo_encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
            # multilingual-e5-small đã có trong requirements (dùng bởi knowledge_retrieval)
            _demo_encoder = SentenceTransformer("intfloat/multilingual-e5-small")
        except Exception:
            _demo_encoder = None  # Fallback: sẽ dùng BM25 thuần
    return _demo_encoder


def retrieve_from_clean_pool(query: str, clean_pool: list, k: int = TOP_K_DEMOS) -> list:
    """
    Hybrid retrieval từ D_clean: BM25 (50%) + Semantic Embedding (50%).

    Sử dụng multilingual-e5-small để tìm kiếm theo ngữ nghĩa, kết hợp với
    BM25 để tăng độ chính xác. Nếu encoder không sẵn có, fallback về BM25 thuần.

    Flow:
    1. Kiểm tra nếu clean_pool trống thì trả về danh sách rỗng.
    2. Tính BM25 scores và normalize về [0, 1].
    3. Tính semantic similarity dùng sentence encoder (cosine).
    4. Hybrid score = 0.5 * BM25 + 0.5 * Semantic.
    5. Chọn top-k theo hybrid score.
    6. Trả về demos kèm source="D_clean_hybrid" để prompt biết đây là đã xác nhận.
    """
    if not clean_pool:
        return []

    cleaned_query = clean_text_transformer(query)
    corpus_items = [clean_text_transformer(item["text"]) for item in clean_pool]

    # === BM25 Scores ===
    tokenized_corpus = [doc.lower().split() for doc in corpus_items]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(cleaned_query.lower().split())
    bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
    if bm25_max > bm25_min:
        bm25_scores_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
    else:
        bm25_scores_norm = np.zeros_like(bm25_scores)

    # === Semantic Scores (fallback về BM25 thuần nếu encoder không có) ===
    encoder = _get_demo_encoder()
    if encoder is not None:
        try:
            query_emb = encoder.encode([cleaned_query], normalize_embeddings=True)
            corpus_embs = encoder.encode(corpus_items, normalize_embeddings=True, batch_size=32)
            semantic_scores = (query_emb @ corpus_embs.T)[0]  # cosine similarity
            # Hybrid: 50% BM25 + 50% Semantic
            hybrid_scores = 0.5 * bm25_scores_norm + 0.5 * semantic_scores
            source_tag = "D_clean_hybrid"
        except Exception:
            hybrid_scores = bm25_scores_norm
            source_tag = "D_clean"
    else:
        hybrid_scores = bm25_scores_norm
        source_tag = "D_clean"

    top_k_indices = np.argsort(hybrid_scores)[::-1][:k]

    demos = []
    for idx in top_k_indices:
        item = clean_pool[int(idx)]
        clean_label = item.get("label", item.get("label_slm", 1))
        demos.append(
            {
                "text": clean_text_transformer(item["text"]),
                "label": to_clean_demo_label(clean_label),
                "source": source_tag,
                "score": float(hybrid_scores[idx]),
            }
        )
    return demos



def prefetch_query_context(
    text: str,
    demo_k: int = TOP_K_DEMOS,
    fact_top_k: int = FACT_TOP_K,
    reuse_knowledge_cache: bool = True,
    knowledge_cache_local: dict = None,
    knowledge_mode: str = None,
    wiki_fetch_full: bool = False,
) -> dict:
    """
    Khởi tạo ngữ cảnh truy xuất (bootstrap context) trước khi chạy vòng lặp MRCD.
    Truy xuất kiến thức (văn bản bổ trợ) và các kết quả mồi (seed) từ Bing.
    
     
    1. Tiền xử lý văn bản đầu vào.
    2. Xác định chế độ kiến thức (wiki_only hoặc full).
    3. Lấy gói kiến thức (knowledge bundle):
       - Nếu reuse_knowledge_cache=True: Thử lấy từ cache cục bộ.
       - Nếu không: Xây dựng mới bằng `build_knowledge_bundle`.
    4. Tìm kiếm tin tức mồi (seed news) từ Bing qua hàm `search_news`.
    5. Trả về một dictionary chứa văn bản sạch, gói kiến thức và tin tức mồi.
    """
    cleaned_text = clean_text_transformer(text)
    mode = knowledge_mode or KNOWLEDGE_MODE

    if reuse_knowledge_cache:
        knowledge_bundle = get_cached_knowledge_bundle_local(
            cleaned_text,
            knowledge_cache_local,
            fact_top_k=fact_top_k,
            mode=mode,
            wiki_fetch_full=wiki_fetch_full,
        )
    else:
        knowledge_bundle = build_knowledge_bundle(
            cleaned_text, fact_top_k=fact_top_k, mode=mode, wiki_fetch_full=wiki_fetch_full
        )

    bing_seed_news = search_news(cleaned_text, max_results=demo_k)

    return {
        "text": cleaned_text,
        "knowledge_bundle": knowledge_bundle,
        "knowledge_text": knowledge_bundle.get("combined_text", "No info."),
        "knowledge_mode": knowledge_bundle.get("mode", mode),
        "bing_seed_news": bing_seed_news,
    }


def build_evidence_bundle(
    text: str,
    static_corpus: list,
    clean_pool: list,
    round_id: int,
    query_context: dict,
    demo_k: int = TOP_K_DEMOS,
) -> tuple:
    """
    Xây dựng gói bằng chứng (evidence bundle) có sự phân hoá theo vòng (round-aware).
    
     
    1. Tiền xử lý văn bản và lấy thông tin từ query_context.
    2. Nếu Round 1:
       - Kết hợp corpus tĩnh và tin tức mồi (Bing seed).
       - Sử dụng `retrieve_demonstrations` để lấy ví dụ với nhãn ĐỒNG NGHĨA ngẫu nhiên.
       - Nguồn truy xuất: "external_prefetched".
    3. Nếu Round 2 trở đi:
       - Ưu tiên sử dụng `retrieve_from_clean_pool` để lấy ví ví dụ từ pool sạch (D_clean).
       - Tại đây, nhãn sẽ được gán TRỰC TIẾP là "Real" hoặc "Fake" (không dùng từ đồng nghĩa).
       - Nếu pool sạch không có kết quả: Quay lại fallback dùng corpus tĩnh + tin tức mồi.
       - Khi fallback, hệ thống vẫn dùng nhãn ĐỒNG NGHĨA ngẫu nhiên như cũ.
       - Xác định nguồn truy xuất tương ứng ("d_clean" hoặc "fallback_external_prefetched").
    4. Trả về tuple gồm: danh sách demos, nội dung kiến thức, và tên nguồn truy xuất.
    """
    cleaned_text = clean_text_transformer(text)
    knowledge_k = query_context.get("knowledge_text", "No info.")
    bing_seed_news = query_context.get("bing_seed_news", [])

    if round_id == 1:
        combined_corpus = static_corpus + bing_seed_news
        demos = retrieve_demonstrations(cleaned_text, combined_corpus, k=demo_k)
        retrieval_source = "external_prefetched"
    else:
        demos = retrieve_from_clean_pool(cleaned_text, clean_pool, k=demo_k)
        if demos:
            retrieval_source = "d_clean"
        else:
            combined_corpus = static_corpus + bing_seed_news
            demos = retrieve_demonstrations(cleaned_text, combined_corpus, k=demo_k)
            retrieval_source = "fallback_external_prefetched"

    return demos, knowledge_k, retrieval_source


def assess_with_llm(text: str, demos: list, knowledge_k: str, llm, round_id: int = 1) -> dict:
    """
    Đánh giá tin tức bằng mô hình LLM (Round-Aware).
    
    
    1. Tiền xử lý văn bản đầu vào.
    2. Xây dựng prompt phân loại với round_id (round 2+ nhấn mạnh demo đã được xác nhận).
    3. Gọi LLM để sinh văn bản phản hồi.
    4. Phân tích phản hồi của LLM bằng `parse_llm_label` để lấy nhãn Real/Fake (0/1).
    5. Trả về dictionary chứa nhãn của LLM và phản hồi thô.
    """
    cleaned_text = clean_text_transformer(text)

    prompt = build_classification_prompt(
        text=cleaned_text,
        knowledge_k=knowledge_k,
        demos=demos,
        round_id=round_id,
    )
    llm_resp = llm.generate_text(
        prompt, max_output_tokens=LLM_MAX_OUTPUT_TOKENS_CLASSIFICATION
    )
    y_llm, matched_label = parse_llm_label(
        llm_resp,
        default_fake=1,
        return_matched_label=True,
    )

    return {
        "y_llm": y_llm,
        "llm_raw": llm_resp,
        "llm_label_matched": matched_label,
        "prompt": prompt,
    }


def assess_with_llm_and_slm(
    text: str,
    demos: list,
    knowledge_k: str,
    llm,
    slm,
    round_id: int = 1,
) -> dict:
    """
    Đánh giá tin tức bằng cả LLM và SLM (Round-Aware).
    
    1. Gọi assess_with_llm để đánh giá bằng LLM.
    2. Gọi slm.inference để đánh giá bằng SLM.
    3. Trả về kết quả kết hợp.
    """
    llm_res = assess_with_llm(
        text=text,
        demos=demos,
        knowledge_k=knowledge_k,
        llm=llm,
        round_id=round_id,
    )
    
    y_slm, conf_slm, probs_slm = slm.inference(text)
    
    return {
        "y_llm": llm_res["y_llm"],
        "llm_raw": llm_res["llm_raw"],
        "llm_label_matched": llm_res["llm_label_matched"],
        "prompt": llm_res["prompt"],
        "y_slm": y_slm,
        "conf_slm": conf_slm,
        "probs_slm": probs_slm,
    }
