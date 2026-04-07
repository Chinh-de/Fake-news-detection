"""
Evidence bundle construction and dual-model inference.
Handles round-aware demonstration retrieval and LLM+SLM assessment.
"""

from rank_bm25 import BM25Okapi

from src.config import (
    TOP_K_DEMOS,
    FACT_TOP_K,
    LLM_MAX_OUTPUT_TOKENS_CLASSIFICATION,
    KNOWLEDGE_MODE,
)
from src.utils import preprocess_text
from src.labels import parse_llm_label, to_clean_demo_label
from src.prompts import build_classification_prompt
from src.retrieval.demo_retrieval import search_news, retrieve_demonstrations
from src.retrieval.knowledge_agent import (
    build_knowledge_bundle,
    get_cached_knowledge_bundle_local,
)


def retrieve_from_clean_pool(query: str, clean_pool: list, k: int = TOP_K_DEMOS) -> list:
    """
    Truy xuất các ví dụ minh họa gần nhất từ tập D_clean (pool sạch) sử dụng BM25.
    Các nhãn được gán trực tiếp ("Real" hoặc "Fake") theo yêu cầu tuyệt đối từ vòng 2.
    
     
    1. Kiểm tra nếu clean_pool trống thì trả về danh sách rỗng.
    2. Tiền xử lý truy vấn và các văn bản trong clean_pool.
    3. Tokenize corpus và khởi tạo mô hình BM25Okapi.
    4. Tính điểm BM25 cho truy vấn so với các tài liệu trong pool.
    5. Chọn ra top-k kết quả có điểm số cao nhất.
    6. Với mỗi kết quả:
       - Lấy nhãn sạch (clean label) từ item.
       - Chuyển đổi nhãn sạch sang chuỗi trực tiếp "Real"/"Fake" bằng `to_clean_demo_label`.
       - Đóng gói thông tin văn bản, nhãn và nguồn ("D_clean").
    7. Trả về danh sách các ví dụ (demos).
    """
    if not clean_pool:
        return []

    cleaned_query = preprocess_text(query)
    corpus_items = [preprocess_text(item["text"]) for item in clean_pool]
    tokenized_corpus = [doc.lower().split() for doc in corpus_items]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(cleaned_query.lower().split())
    scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_k = [idx for idx, _ in scored_indices[:k]]

    demos = []
    for idx in top_k:
        item = clean_pool[idx]
        clean_label = item.get("label", item.get("label_slm", 1))
        demos.append(
            {
                "text": preprocess_text(item["text"]),
                "label": to_clean_demo_label(clean_label),
                "source": "D_clean",
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
    cleaned_text = preprocess_text(text)
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
    cleaned_text = preprocess_text(text)
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


def assess_with_llm(text: str, demos: list, knowledge_k: str, llm) -> dict:
    """
    Đánh giá tin tức bằng mô hình LLM.
    
    
    1. Tiền xử lý văn bản đầu vào.
    2. Xây dựng prompt phân loại bằng `build_classification_prompt`.
    3. Gọi LLM để sinh văn bản phản hồi.
    4. Phân tích phản hồi của LLM bằng `parse_llm_label` để lấy nhãn Real/Fake (0/1).
    5. Trả về dictionary chứa nhãn của LLM và phản hồi thô.
    """
    cleaned_text = preprocess_text(text)

    prompt = build_classification_prompt(
        text=cleaned_text,
        knowledge_k=knowledge_k,
        demos=demos,
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
