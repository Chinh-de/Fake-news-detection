"""Tests for src/pipeline/evidence.py"""
import pytest
import numpy as np
from unittest.mock import MagicMock

# Skip if rank_bm25 not installed
pytest.importorskip("rank_bm25", reason="rank_bm25 not installed")


class TestRetrieveFromCleanPool:
    def test_empty_pool(self):
        from src.pipeline.evidence import retrieve_from_clean_pool
        assert retrieve_from_clean_pool("query", [], k=4) == []

    def test_returns_k_demos(self):
        from src.pipeline.evidence import retrieve_from_clean_pool
        pool = [
            {"text": "climate change effects", "label": 0},
            {"text": "financial market news", "label": 1},
            {"text": "global warming rise", "label": 0},
        ]
        demos = retrieve_from_clean_pool("climate warming", pool, k=2)
        assert len(demos) == 2
        assert all(d["source"] == "D_clean" for d in demos)


class TestBuildEvidenceBundle:
    def test_round1_external(self):
        from src.pipeline.evidence import build_evidence_bundle
        ctx = {"knowledge_text": "K", "bing_seed_news": ["news1"]}
        _, _, source = build_evidence_bundle(
            "test", ["static doc"], [], round_id=1, query_context=ctx,
        )
        assert source == "external_prefetched"

    def test_round2_clean_pool(self):
        from src.pipeline.evidence import build_evidence_bundle
        ctx = {"knowledge_text": "K", "bing_seed_news": []}
        pool = [{"text": "clean topic text here", "label": 0}]
        _, _, source = build_evidence_bundle(
            "topic text", ["static"], pool, round_id=2, query_context=ctx,
        )
        assert source == "d_clean"

    def test_round2_fallback(self):
        from src.pipeline.evidence import build_evidence_bundle
        ctx = {"knowledge_text": "K", "bing_seed_news": ["news"]}
        _, _, source = build_evidence_bundle(
            "test", ["static doc"], [], round_id=2, query_context=ctx,
        )
        assert source == "fallback_external_prefetched"


class TestAssessWithLlmAndSlm:
    def test_both_real(self):
        from src.pipeline.evidence import assess_with_llm_and_slm
        mock_llm = MagicMock()
        mock_llm.generate_text.return_value = "Real"
        mock_slm = MagicMock()
        mock_slm.inference.return_value = (0, 0.92, np.array([0.92, 0.08]))
        r = assess_with_llm_and_slm("test", [], "info", mock_llm, mock_slm)
        assert r["y_llm"] == 0
        assert r["y_slm"] == 0

    def test_disagreement(self):
        from src.pipeline.evidence import assess_with_llm_and_slm
        mock_llm = MagicMock()
        mock_llm.generate_text.return_value = "Fake"
        mock_slm = MagicMock()
        mock_slm.inference.return_value = (0, 0.85, np.array([0.85, 0.15]))
        r = assess_with_llm_and_slm("test", [], "info", mock_llm, mock_slm)
        assert r["y_llm"] == 1
        assert r["y_slm"] == 0
