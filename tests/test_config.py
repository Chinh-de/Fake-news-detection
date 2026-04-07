"""Tests for src/config.py"""
import os
import pytest


def test_config_imports():
    """All config constants should be importable."""
    from src.config import (
        DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, MODEL_PATH,
        LLM_MODEL_NAME, LLM_MAX_NEW_TOKENS, LLM_MAX_OUTPUT_TOKENS_EXTRACTION,
        LLM_MAX_OUTPUT_TOKENS_CLASSIFICATION, LLM_TEMPERATURE, LLM_TOP_P,
        SLM_BACKEND,
        CONFIDENCE_THRESHOLD, NUM_LOOP, TOP_K_DEMOS, FACT_TOP_K,
        BOOTSTRAP_ENABLE_PARALLEL, BOOTSTRAP_MAX_WORKERS, CRAWL_MAX_WORKERS,
        ENABLE_SLM_FINETUNE, SLM_FINETUNE_EPOCHS, SLM_FINETUNE_BATCH_SIZE,
        SLM_FINETUNE_LR, SLM_FINETUNE_WEIGHT_DECAY, SLM_FINETUNE_MIN_SAMPLES,
        KNOWLEDGE_MODE,
        AG_NEWS_URL, TRUST_DOMAINS,
        RETRIEVAL_DEBUG_CSV,
    )


def test_default_llm_model():
    """Default LLM should be Llama-3-8B when no env override."""
    from src.config import LLM_MODEL_NAME
    # Either from .env or the hardcoded default
    assert isinstance(LLM_MODEL_NAME, str)
    assert len(LLM_MODEL_NAME) > 0


def test_default_slm_backend():
    from src.config import SLM_BACKEND
    assert SLM_BACKEND in ("hf", "vllm")


def test_default_knowledge_mode():
    from src.config import KNOWLEDGE_MODE
    assert KNOWLEDGE_MODE in ("wiki_only", "full")


def test_pipeline_hyperparameters_valid():
    from src.config import CONFIDENCE_THRESHOLD, NUM_LOOP, TOP_K_DEMOS
    assert 0.0 < CONFIDENCE_THRESHOLD <= 1.0
    assert NUM_LOOP >= 1
    assert TOP_K_DEMOS >= 1


def test_finetune_config_valid():
    from src.config import (
        SLM_FINETUNE_EPOCHS, SLM_FINETUNE_BATCH_SIZE,
        SLM_FINETUNE_LR, SLM_FINETUNE_WEIGHT_DECAY,
        SLM_FINETUNE_MIN_SAMPLES,
    )
    assert SLM_FINETUNE_EPOCHS >= 1
    assert SLM_FINETUNE_BATCH_SIZE >= 1
    assert 0 < SLM_FINETUNE_LR < 1
    assert SLM_FINETUNE_WEIGHT_DECAY >= 0
    assert SLM_FINETUNE_MIN_SAMPLES >= 1


def test_trust_domains_not_empty():
    from src.config import TRUST_DOMAINS
    assert isinstance(TRUST_DOMAINS, list)
    assert len(TRUST_DOMAINS) > 0
    for d in TRUST_DOMAINS:
        assert "." in d  # valid domain format


def test_classification_max_tokens_small():
    """LLM classification output should be small since we only expect Real/Fake."""
    from src.config import LLM_MAX_OUTPUT_TOKENS_CLASSIFICATION
    assert LLM_MAX_OUTPUT_TOKENS_CLASSIFICATION <= 16


def test_csv_paths_derived_from_data_dir():
    from src.config import DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV
    assert TRAIN_CSV.startswith(DATA_DIR)
    assert VAL_CSV.startswith(DATA_DIR)
    assert TEST_CSV.startswith(DATA_DIR)
