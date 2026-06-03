from src.utils import clean_text_transformer


# ================================================================
# Adaptive Threshold Helpers
# ================================================================
def compute_adaptive_threshold(round_id: int, base_threshold: float = 0.8) -> float:
    """
    Tính ngưỡng confidence thích nghi theo vòng lặp (đã cmt phần thay đổi, giữ nguyên ngưỡng).
    """
    # delta = 0.05 - 0.05 * (round_id - 1)
    # return max(0.65, base_threshold + delta)
    return base_threshold


def split_clean_noisy(sample: dict, confidence_threshold: float, round_id: int = None) -> bool:
    """
    Selection rule với Adaptive Threshold + Soft Agreement (Case 2 tạm cmt).

    Trường hợp được xếp vào D_clean:
    1. LLM == SLM đồng thuận VÀ conf_slm >= adaptive_threshold (standard)
    """
    llm_label = sample.get("label_llm")
    slm_label = sample.get("label_slm")
    conf = sample.get("conf_slm", 0.0)

    if round_id is None:
        adaptive_threshold = confidence_threshold
    else:
        adaptive_threshold = compute_adaptive_threshold(round_id, confidence_threshold)

    # Case 1: Standard agreement
    if llm_label == slm_label and conf >= adaptive_threshold:
        return True

    # # Case 2: Soft agreement — SLM rất tự tin từ Round 2 (tạm cmt)
    # # (Chỉ áp dụng khi SLM đã được fine-tune ít nhất một lần)
    # if round_id is not None and round_id >= 2 and conf >= 0.92:
    #     return True

    return False


def finalize_remaining_noisy_with_slm(d_noisy: list, slm) -> list:
    """
    Final judgment: Quyết định cuối của những mẫu nhiễu bằng SLM (không liên quan LLM).
    
    Hàm cũ dùng weighted ensemble (SLM + lịch sử LLM) đã được cmt lại dưới đây.
    """
    # finalized = []
    # for sample in d_noisy:
    #     text = clean_text_transformer(sample["text"])
    #     y_slm, conf_slm, probs_slm = slm.inference(text)
    #     p_fake_slm = float(probs_slm[1])
    #     llm_label = sample.get("label_llm")
    #     if llm_label is not None:
    #         p_fake_llm = float(llm_label)
    #         p_fake_ensemble = 0.6 * p_fake_slm + 0.4 * p_fake_llm
    #     else:
    #         p_fake_ensemble = p_fake_slm
    #     final_label = 1 if p_fake_ensemble >= 0.5 else 0
    #     final_conf = max(p_fake_ensemble, 1 - p_fake_ensemble)
    #     final_sample = dict(sample)
    #     final_sample["label"] = final_label
    #     final_sample["label_final"] = final_label
    #     final_sample["conf_slm_final"] = final_conf
    #     final_sample["p_fake_ensemble"] = p_fake_ensemble
    #     final_sample["status"] = "finalized_by_slm"
    #     finalized.append(final_sample)
    # return finalized

    finalized = []
    for sample in d_noisy:
        text = clean_text_transformer(sample["text"])
        y_slm, conf_slm, probs_slm = slm.inference(text)

        final_sample = dict(sample)
        final_sample["label"] = y_slm
        final_sample["label_final"] = y_slm
        final_sample["conf_slm_final"] = conf_slm
        final_sample["status"] = "finalized_by_slm"
        finalized.append(final_sample)
    return finalized
