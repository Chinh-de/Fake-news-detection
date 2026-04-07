"""
Data selection logic for MRCD pipeline.
Handles clean/noisy splitting and final judgment.
"""

from src.utils import preprocess_text


def split_clean_noisy(sample: dict, confidence_threshold: float) -> bool:
    """
    Selection rule: LLM-SLM agreement + SLM confidence threshold.
    
    Returns True if sample should go to D_clean, False for D_noisy.
    """
    return (
        sample["label_llm"] == sample["label_slm"]
        and sample["conf_slm"] >= confidence_threshold
    )


def finalize_remaining_noisy_with_slm(d_noisy: list, slm) -> list:
    """
    Final judgment: force SLM labels for all unresolved noisy samples.
    Used after all rounds are exhausted.
    
    Args:
        d_noisy: List of noisy sample dicts
        slm: IntegratedSLM instance
        
    Returns:
        List of finalized sample dicts with SLM-assigned labels
    """
    finalized = []
    for sample in d_noisy:
        text = preprocess_text(sample["text"])
        y_slm, conf_slm, _ = slm.inference(text)
        final_sample = dict(sample)
        final_sample["label"] = y_slm
        final_sample["label_final"] = y_slm
        final_sample["conf_slm_final"] = conf_slm
        final_sample["status"] = "finalized_by_slm"
        finalized.append(final_sample)
    return finalized
