"""
ml/predictor.py — Wraps PhoBERT model + Retrieval pipeline
"""
import sys
import re
import os
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Project root (demo_phobert/) — for locating model/ directory
_PROJECT_ROOT = Path(__file__).parent.parent.parent
# be/ directory — retriever, explainer, query_builder live here
_BE_DIR = Path(__file__).parent.parent

_tokenizer = None
_model = None
_load_error = None


def _load_model():
    global _tokenizer, _model, _load_error
    if _model is not None:
        return
    model_path = str(_PROJECT_ROOT / "model")
    try:
        _tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        _model = AutoModelForSequenceClassification.from_pretrained(model_path)
        _model.eval()
    except Exception as e:
        _load_error = str(e)


def light_clean(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_text(text: str, max_len: int = 256):
    """Run PhoBERT inference. Returns (label, confidence, probs)."""
    _load_model()
    if _load_error or _tokenizer is None:
        raise RuntimeError(f"Model not loaded: {_load_error}")

    inputs = _tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).numpy().flatten()

    pred_id = int(np.argmax(probs))
    label_map = {0: "REAL", 1: "FAKE"}
    return label_map[pred_id], float(probs[pred_id]), probs


def run_prediction(
    text: str,
    max_length: int = 256,
    top_k_bm25: int = 4,
    top_k_web: int = 3,
    enable_web: bool = True,
    crawl_web: bool = False,
) -> dict:
    """Full pipeline: predict + retrieve evidence + build explanation."""
    cleaned = light_clean(text)

    label, conf, probs = predict_text(cleaned, max_len=max_length)

    # Retrieval — add be/ to sys.path so retriever/explainer/query_builder are importable
    try:
        be_dir = str(_BE_DIR)
        if be_dir not in sys.path:
            sys.path.insert(0, be_dir)

        from retriever import retrieve_evidence, analyze_retrieved_distribution
        from explainer import build_explanation

        retrieval = retrieve_evidence(
            cleaned,
            top_k_bm25=top_k_bm25,
            top_k_web=top_k_web,
            enable_web=enable_web,
            crawl_web=crawl_web,
        )
        distribution = analyze_retrieved_distribution(retrieval)
        explanation = build_explanation(label, conf, probs, retrieval["all"], distribution)

        bm25_evs = retrieval["bm25"]
        web_evs = retrieval["web"]
        corpus_name = retrieval["corpus"]
        corpus_total = retrieval["corpus_total"]
    except Exception as e:
        # Fallback if retrieval fails
        explanation = {
            "verdict": f"{'🔴 TIN GIẢ' if label == 'FAKE' else '🟢 TIN THẬT'} (độ tin cậy: {conf:.1%})",
            "evidence": "⚠️ Không thể tải retrieval module.",
            "factors": "",
            "disclaimer": str(e),
            "is_consistent": True,
        }
        bm25_evs = []
        web_evs = []
        corpus_name = "unavailable"
        corpus_total = 0

    return {
        "label": label,
        "confidence": conf,
        "prob_real": float(probs[0]),
        "prob_fake": float(probs[1]),
        "explanation": explanation,
        "bm25_evidences": bm25_evs,
        "web_evidences": web_evs,
        "corpus_name": corpus_name,
        "corpus_total": corpus_total,
    }
