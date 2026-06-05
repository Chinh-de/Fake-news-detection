"""
MRCD Pipeline Runner - Main orchestrator.
Implements the multi-round collaborative detection flow:

1. Bootstrap: Prefetch knowledge + Bing seed for all events
2. Round 1: External retrieval + LLM/SLM assessment + clean/noisy split
3. Rounds 2-N: D_clean retrieval + SLM fine-tune + re-assessment
4. Final Judgment: SLM force-labeling for remaining noisy samples
"""

from typing import List

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

from src.config import (
    NUM_LOOP,
    CONFIDENCE_THRESHOLD,
    TOP_K_DEMOS,
    FACT_TOP_K,
    KNOWLEDGE_MODE,
    BOOTSTRAP_ENABLE_PARALLEL,
    BOOTSTRAP_MAX_WORKERS,
    ENABLE_SLM_FINETUNE,
    SLM_FINETUNE_EPOCHS,
    SLM_FINETUNE_BATCH_SIZE,
    SLM_FINETUNE_LR,
    SLM_FINETUNE_WEIGHT_DECAY,
    SLM_FINETUNE_MIN_SAMPLES,
    WIKI_FETCH_FULL,
)
from src.utils import (
    clean_text_transformer,
    log_prediction_to_csv,
    log_round_trace_to_csv,
)
from src.llm.handler import get_llm
from src.retrieval.demo_retrieval import load_news_corpus
from src.pipeline.evidence import (
    prefetch_query_context,
    build_evidence_bundle,
    assess_with_llm,
)
from src.pipeline.selection import split_clean_noisy, finalize_remaining_noisy_with_slm
from src.pipeline.finetune import maybe_finetune_slm_on_clean


def run_mrcd_pipeline(
    events: List[str],
    slm,
    max_rounds: int = NUM_LOOP,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    knowledge_mode: str = None,
    reuse_knowledge_cache: bool = True,
    bootstrap_parallel: bool = BOOTSTRAP_ENABLE_PARALLEL,
    bootstrap_max_workers: int = BOOTSTRAP_MAX_WORKERS,
    enable_slm_finetune: bool = ENABLE_SLM_FINETUNE,
    slm_finetune_epochs: int = SLM_FINETUNE_EPOCHS,
    slm_finetune_batch_size: int = SLM_FINETUNE_BATCH_SIZE,
    slm_finetune_lr: float = SLM_FINETUNE_LR,
    slm_finetune_weight_decay: float = SLM_FINETUNE_WEIGHT_DECAY,
    slm_finetune_min_samples: int = SLM_FINETUNE_MIN_SAMPLES,
    ground_truth: List[int] = None,
    
    # Caching parameters
    cached_llm_labels: List[int] = None,
    cached_llm_raws: List[str] = None,
    cached_knowledge_texts: List[str] = None,
    cached_final_prompts: List[str] = None,
    cached_wiki_evidences: List[list] = None,
    cached_rag_evidences: List[list] = None,
    cached_fewshot_examples_list: List[list] = None,
):
    """
    Run the full MRCD multi-round pipeline with database caching optimization.
    
    Args:
        events: List of news texts to classify
        slm: IntegratedSLM instance (injected dependency)
        max_rounds: Maximum number of rounds (default: 3)
        confidence_threshold: SLM confidence threshold for clean selection
        knowledge_mode: "wiki_only" or "full" (default from config)
        reuse_knowledge_cache: Whether to cache knowledge bundles
        bootstrap_parallel: Enable parallel bootstrap context fetching
        enable_slm_finetune: Whether to fine-tune SLM each round
        
    Returns:
        dict with keys: results, clean, noisy, finalized_noisy,
                       history, finetune_history, knowledge_cache_size
    """
    mode = knowledge_mode or KNOWLEDGE_MODE

    print(f"Starting MRCD pipeline for {len(events)} events...")
    print(f"Knowledge mode: {mode}")
    llm = get_llm()

    print("Loading news corpus (fact-checking base)...")
    static_corpus = load_news_corpus()
    print(f"Corpus loaded: {len(static_corpus)} documents")

    cleaned_events = [clean_text_transformer(e) for e in events]

    event_states = [
        {
            "event_id": idx,
            "text": text,
            "round": 0,
            "status": "unprocessed",
            "label": None,
            "label_llm": None,
            "label_slm": None,
            "conf_slm": None,
            "llm_raw": None,
            "llm_label_matched": None,
            "retrieval_source": None,
            "knowledge": None,
            "query_context": None,
            "ground_truth": ground_truth[idx] if ground_truth is not None else None,
            
            # Caching fields
            "cached_llm_label": cached_llm_labels[idx] if (cached_llm_labels is not None and idx < len(cached_llm_labels)) else -1,
            "cached_llm_raw": cached_llm_raws[idx] if (cached_llm_raws is not None and idx < len(cached_llm_raws)) else None,
            "cached_knowledge_text": cached_knowledge_texts[idx] if (cached_knowledge_texts is not None and idx < len(cached_knowledge_texts)) else None,
            "cached_final_prompt": cached_final_prompts[idx] if (cached_final_prompts is not None and idx < len(cached_final_prompts)) else None,
            "cached_wiki_evidence": cached_wiki_evidences[idx] if (cached_wiki_evidences is not None and idx < len(cached_wiki_evidences)) else None,
            "cached_rag_evidence": cached_rag_evidences[idx] if (cached_rag_evidences is not None and idx < len(cached_rag_evidences)) else None,
            "cached_fewshot_examples": cached_fewshot_examples_list[idx] if (cached_fewshot_examples_list is not None and idx < len(cached_fewshot_examples_list)) else None,
        }
        for idx, text in enumerate(cleaned_events)
    ]

    d_clean = []
    d_noisy = []
    round_logs = []
    round_history = []
    finetune_history = []
    knowledge_cache_local = {}

    # ================================================================
    # Bootstrap: Prefetch knowledge + Bing seed
    # ================================================================
    print("\n=== Bootstrap Retrieval Context ===")
    
    # Chỉ prefetch ngữ cảnh cho những bài viết chưa được lưu kết quả LLM trong cache
    unique_texts = []
    for s in event_states:
        if s["cached_llm_label"] == -1 or s["cached_llm_label"] is None:
            if s["text"] not in unique_texts:
                unique_texts.append(s["text"])

    context_map = {}

    if unique_texts:
        if bootstrap_parallel and len(unique_texts) > 1:
            workers = max(1, int(bootstrap_max_workers))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_text = {
                    executor.submit(
                        prefetch_query_context,
                        text,
                        TOP_K_DEMOS,
                        FACT_TOP_K,
                        reuse_knowledge_cache,
                        knowledge_cache_local,
                        mode,
                        WIKI_FETCH_FULL,
                    ): text
                    for text in unique_texts
                }

                for future in tqdm(
                    as_completed(future_to_text),
                    total=len(future_to_text),
                    desc="Bootstrap Context",
                ):
                    text = future_to_text[future]
                    try:
                        context_map[text] = future.result()
                    except Exception:
                        context_map[text] = {
                            "text": text,
                            "knowledge_bundle": {"combined_text": "No info."},
                            "knowledge_text": "No info.",
                            "knowledge_mode": mode,
                            "bing_seed_news": [],
                        }
        else:
            for text in tqdm(unique_texts, desc="Bootstrap Context"):
                context_map[text] = prefetch_query_context(
                    text=text,
                    demo_k=TOP_K_DEMOS,
                    fact_top_k=FACT_TOP_K,
                    reuse_knowledge_cache=reuse_knowledge_cache,
                    knowledge_cache_local=knowledge_cache_local,
                    knowledge_mode=mode,
                    wiki_fetch_full=WIKI_FETCH_FULL,
                )

    for state in event_states:
        if state["cached_llm_label"] != -1 and state["cached_llm_label"] is not None:
            # Tái dựng query context từ các trường đã lưu trong cache
            state["query_context"] = {
                "text": state["text"],
                "knowledge_text": state["cached_knowledge_text"] or "",
                "bing_seed_news": [],
            }
            state["knowledge"] = state["cached_knowledge_text"] or ""
        else:
            qctx = context_map.get(state["text"])
            if qctx is None:
                qctx = {
                    "text": state["text"],
                    "knowledge_bundle": {"combined_text": "No info."},
                    "knowledge_text": "No info.",
                    "knowledge_mode": mode,
                    "bing_seed_news": [],
                }
            state["query_context"] = qctx
            state["knowledge"] = qctx.get("knowledge_text", "No info.")

    # ================================================================
    # Round 1: External Retrieval + Assessment + Selection
    # ================================================================
    round_id = 1
    print("\n=== Round 1: Retrieval + Assessment + Selection ===")
    
    # 1. LLM Assessment (Or Cache Load)
    for state in tqdm(event_states, desc="Round 1 - LLM Processing"):
        # Nếu có cache và hợp lệ, bỏ qua tìm kiếm và gọi LLM, lấy luôn kết quả
        if state["cached_llm_label"] != -1 and state["cached_llm_label"] is not None:
            y_llm = int(state["cached_llm_label"])
            llm_raw = state["cached_llm_raw"] or ""
            matched_label = "Thật" if y_llm == 0 else "Giả"
            prompt = state["cached_final_prompt"] or ""
            retrieval_source = "db_cache"
            knowledge_k = state["cached_knowledge_text"] or ""
            wiki_ev = state["cached_wiki_evidence"]
            rag_ev = state["cached_rag_evidence"]
            fewshot_ev = state["cached_fewshot_examples"]
        else:
            text = clean_text_transformer(state["text"])
            demos, knowledge_k, retrieval_source = build_evidence_bundle(
                text=text,
                static_corpus=static_corpus,
                clean_pool=d_clean,
                round_id=round_id,
                query_context=state["query_context"],
                demo_k=TOP_K_DEMOS,
            )
            assess = assess_with_llm(
                text=text, demos=demos, knowledge_k=knowledge_k,
                llm=llm, round_id=round_id,
            )
            y_llm = assess["y_llm"]
            llm_raw = assess["llm_raw"]
            matched_label = assess["llm_label_matched"]
            prompt = assess["prompt"]
            # Extract structured evidence from query_context and demos
            wiki_ev = state["query_context"].get("knowledge_bundle", {}).get("wiki_definitions", {})
            rag_ev = state["query_context"].get("knowledge_bundle", {}).get("rag_evidence", [])
            fewshot_ev = demos

        state.update(
            {
                "round": round_id,
                "label": y_llm,
                "label_llm": y_llm,
                "llm_raw": llm_raw,
                "llm_label_matched": matched_label,
                "retrieval_source": retrieval_source,
                "knowledge": knowledge_k,
                "prompt": prompt,
                "wiki_evidence": wiki_ev,
                "rag_evidence": rag_ev,
                "fewshot_examples": fewshot_ev,
            }
        )

    # 2. SLM Batch Inference (Luôn dự đoán bằng SLM hiện tại để kiểm tra độ đồng thuận)
    print(f"Round 1 - SLM Batch Inference for {len(event_states)} items")
    slm_texts = [state["text"] for state in event_states]
    slm_results = slm.inference_batch(slm_texts)

    # 3. Merge, Log, and Split
    for state, res in zip(event_states, slm_results):
        pred, conf, probs = res
        state["label_slm"] = pred
        state["y_slm"] = pred
        state["conf_slm"] = conf

        # Ghi log vết (trace) cho từng sự kiện trong vòng này
        log_round_trace_to_csv(
            round_id=round_id,
            event_id=state["event_id"],
            text=state["text"],
            y_slm=state["y_slm"],
            y_llm=state["label_llm"],
            ground_truth=state["ground_truth"],
            conf_slm=state["conf_slm"],
            prompt=state["prompt"],
        )

        if split_clean_noisy(state, confidence_threshold, round_id=round_id):
            state["status"] = "clean"
            d_clean.append(state)
            # Ghi log kết quả ngay lập tức để tiết kiệm RAM
            log_prediction_to_csv(
                event_id=state["event_id"],
                text=state["text"],
                label=state["label"],
                conf=state["conf_slm"],
                round_id=round_id,
                status=state["status"],
            )
        else:
            state["status"] = "noisy"
            d_noisy.append(state)

        round_logs.append({
            "event_id": state["event_id"],
            "round_id": round_id,
            "y_llm": state["label_llm"],
            "y_slm": state["y_slm"],
            "conf_slm": state["conf_slm"],
            "status": state["status"],
            "fewshot_examples": state["fewshot_examples"],
            # RAG và wiki chỉ lưu ở Round 1 (bootstrap). Round 2+ dùng None để tránh trùng lặp.
            "rag_evidence": state["rag_evidence"],
            "wiki_evidence": state["wiki_evidence"],
        })

    round_history.append(
        {
            "round": round_id,
            "clean_count": len(d_clean),
            "noisy_count": len(d_noisy),
        }
    )
    print(f"Round {round_id} summary -> Clean: {len(d_clean)}, Noisy: {len(d_noisy)}")

    # ================================================================
    # Rounds 2-N: D_clean Retrieval + Fine-tune + Re-assessment
    # ================================================================
    round_id = 2
    while d_noisy and round_id <= max_rounds:
        print(f"\n=== Round {round_id}: Re-Assessment + SLM Fine-tune ===")

        # Finetune ở đầu mỗi vòng lặp tiếp theo (dựa trên D_clean đã thu thập)
        ft_stats = maybe_finetune_slm_on_clean(
            slm=slm,
            clean_pool=d_clean,
            round_id=round_id,
            enable_slm_finetune=enable_slm_finetune,
            slm_finetune_epochs=slm_finetune_epochs,
            slm_finetune_batch_size=slm_finetune_batch_size,
            slm_finetune_lr=slm_finetune_lr,
            slm_finetune_weight_decay=slm_finetune_weight_decay,
            slm_finetune_min_samples=slm_finetune_min_samples,
        )
        finetune_history.append({"round": round_id, **ft_stats})

        next_noisy = []
        promoted_clean = 0

        # 1. LLM Assessment (Được chạy lại bình thường ở Round 2 trở đi để cập nhật demos từ D_clean)
        for state in tqdm(d_noisy, desc=f"Round {round_id} - LLM Processing"):
            text = clean_text_transformer(state["text"])
            demos, knowledge_k, retrieval_source = build_evidence_bundle(
                text=text,
                static_corpus=static_corpus,
                clean_pool=d_clean,
                round_id=round_id,
                query_context=state["query_context"],
                demo_k=TOP_K_DEMOS,
            )
            assess = assess_with_llm(
                text=text, demos=demos, knowledge_k=knowledge_k,
                llm=llm, round_id=round_id,
            )

            # Extract structured evidence from query_context and demos for current round
            wiki_ev = state["query_context"].get("knowledge_bundle", {}).get("wiki_definitions", {})
            rag_ev = state["query_context"].get("knowledge_bundle", {}).get("rag_evidence", [])
            fewshot_ev = demos

            state.update(
                {
                    "round": round_id,
                    "label": assess["y_llm"],
                    "label_llm": assess["y_llm"],
                    "llm_raw": assess["llm_raw"],
                    "llm_label_matched": assess["llm_label_matched"],
                    "retrieval_source": retrieval_source,
                    "knowledge": knowledge_k,
                    "prompt": assess["prompt"],
                    "round_wiki_evidence": wiki_ev,
                    "round_rag_evidence": rag_ev,
                    "round_fewshot_examples": fewshot_ev,
                }
            )

        # 2. SLM Batch Inference
        print(f"Round {round_id} - SLM Batch Inference for {len(d_noisy)} items")
        slm_texts = [state["text"] for state in d_noisy]
        slm_results = slm.inference_batch(slm_texts)

        # 3. Merge, Log, and Split
        for state, res in zip(d_noisy, slm_results):
            pred, conf, probs = res
            state["label_slm"] = pred
            state["y_slm"] = pred
            state["conf_slm"] = conf

            # Ghi log vết (trace) cho từng sự kiện trong vòng này
            log_round_trace_to_csv(
                round_id=round_id,
                event_id=state["event_id"],
                text=state["text"],
                y_slm=state["y_slm"],
                y_llm=state["label_llm"],
                ground_truth=state["ground_truth"],
                conf_slm=state["conf_slm"],
                prompt=state["prompt"],
            )

            if split_clean_noisy(state, confidence_threshold, round_id=round_id):
                state["status"] = f"clean@round{round_id}"
                d_clean.append(state)
                promoted_clean += 1
                # Ghi log kết quả ngay lập tức
                log_prediction_to_csv(
                    event_id=state["event_id"],
                    text=state["text"],
                    label=state["label"],
                    conf=state["conf_slm"],
                    round_id=round_id,
                    status=state["status"],
                )
            else:
                state["status"] = f"noisy@round{round_id}"
                next_noisy.append(state)

            round_logs.append({
                "event_id": state["event_id"],
                "round_id": round_id,
                "y_llm": state["label_llm"],
                "y_slm": state["y_slm"],
                "conf_slm": state["conf_slm"],
                "status": state["status"],
                "fewshot_examples": state.get("round_fewshot_examples"),
                # RAG và wiki được bootstrap 1 lần ở Round 1 và tái sử dụng qua query_context cache.
                # Không lưu lại ở round 2+ để tránh trùng lặp dữ liệu lớn trong DB.
                "rag_evidence": None,
                "wiki_evidence": None,
            })

        d_noisy = next_noisy
        round_history.append(
            {
                "round": round_id,
                "promoted_to_clean": promoted_clean,
                "clean_count": len(d_clean),
                "noisy_count": len(d_noisy),
            }
        )
        print(
            f"Round {round_id} summary -> +Clean: {promoted_clean}, "
            f"Total Clean: {len(d_clean)}, Remaining Noisy: {len(d_noisy)}"
        )

        round_id += 1

    # ================================================================
    # Final Judgment: SLM force-labeling remaining noisy
    # ================================================================
    finalized_noisy = []
    if d_noisy:
        print(
            f"\n=== Final Judgment: SLM force-labeling "
            f"{len(d_noisy)} unresolved noisy samples ==="
        )
        finalized_noisy = finalize_remaining_noisy_with_slm(d_noisy, slm)
        for final_sample in finalized_noisy:
            final_sample["status"] = "finalized_by_slm"
            # Ghi log vết cho bước chốt hạ cuối cùng
            log_round_trace_to_csv(
                round_id="final_judgment",
                event_id=final_sample["event_id"],
                text=final_sample["text"],
                y_slm=final_sample["label"],
                y_llm=None,
                ground_truth=final_sample["ground_truth"],
                conf_slm=final_sample["conf_slm"],
                prompt="N/A (Final SLM Judgment)",
            )
            # Ghi log kết quả cuối cùng
            log_prediction_to_csv(
                event_id=final_sample["event_id"],
                text=final_sample["text"],
                label=final_sample["label"],
                conf=final_sample["conf_slm"],
                round_id=max_rounds + 1,  # Final judgment round
                status=final_sample["status"],
            )
        round_history.append(
            {
                "round": "final_judgment",
                "force_labeled_by_slm": len(finalized_noisy),
                "clean_count": len(d_clean),
                "remaining_noisy_after_final": 0,
            }
        )

    ordered_results = sorted(event_states, key=lambda x: x["event_id"])

    return {
        "results": ordered_results,
        "clean": d_clean,
        "noisy": d_noisy,
        "finalized_noisy": finalized_noisy,
        "history": round_history,
        "finetune_history": finetune_history,
        "knowledge_cache_size": len(knowledge_cache_local),
        "round_logs": round_logs,
    }
