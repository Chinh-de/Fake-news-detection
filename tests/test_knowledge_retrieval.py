"""Tests for src/retrieval/knowledge_retrieval.py (pure logic, no network)"""
import pytest
from unittest.mock import patch, MagicMock

# Skip entire module if heavy deps not installed
pytest.importorskip("ddgs", reason="ddgs not installed")
pytest.importorskip("rank_bm25", reason="rank_bm25 not installed")
pytest.importorskip("sentence_transformers", reason="sentence_transformers not installed")
pytest.importorskip("curl_cffi", reason="curl_cffi not installed")


class TestBuildTrustedDomainQuery:
    def test_with_domains(self):
        from src.retrieval.knowledge_retrieval import build_trusted_domain_query
        result = build_trusted_domain_query(["apnews.com", "reuters.com"])
        assert "site:apnews.com" in result
        assert " OR " in result

    def test_empty_domains(self):
        from src.retrieval.knowledge_retrieval import build_trusted_domain_query
        assert build_trusted_domain_query([]) == ""

    def test_single_domain(self):
        from src.retrieval.knowledge_retrieval import build_trusted_domain_query
        result = build_trusted_domain_query(["example.com"])
        assert result == "(site:example.com)"


class TestChunkTextBySentences:
    def test_empty_text(self):
        from src.retrieval.knowledge_retrieval import chunk_text_by_sentences
        assert chunk_text_by_sentences("") == []

    def test_single_sentence(self):
        from src.retrieval.knowledge_retrieval import chunk_text_by_sentences
        result = chunk_text_by_sentences("Single sentence.", max_words=100)
        assert len(result) == 1

    def test_splits_at_boundary(self):
        from src.retrieval.knowledge_retrieval import chunk_text_by_sentences
        text = "First sentence. Second sentence. Third sentence."
        result = chunk_text_by_sentences(text, max_words=4, overlap_sentences=0)
        assert len(result) >= 2

    def test_very_long_single_sentence(self):
        from src.retrieval.knowledge_retrieval import chunk_text_by_sentences
        long_sentence = " ".join(["word"] * 500)
        result = chunk_text_by_sentences(long_sentence, max_words=100)
        assert len(result) >= 1


class TestAnalyzeClaimEntitiesAndQuery:
    @patch("src.retrieval.knowledge_retrieval.get_llm")
    def test_successful_extraction(self, mock_get_llm):
        from src.retrieval.knowledge_retrieval import analyze_claim_entities_and_query
        mock_llm = MagicMock()
        mock_llm.generate_text.return_value = '{"entities": ["NASA"], "query": "NASA mission"}'
        mock_get_llm.return_value = mock_llm

        result = analyze_claim_entities_and_query("NASA plans mission")
        assert "NASA" in result["entities"]

    @patch("src.retrieval.knowledge_retrieval.get_llm")
    def test_fallback_on_error(self, mock_get_llm):
        from src.retrieval.knowledge_retrieval import analyze_claim_entities_and_query
        mock_llm = MagicMock()
        mock_llm.generate_text.side_effect = Exception("error")
        mock_get_llm.return_value = mock_llm

        result = analyze_claim_entities_and_query("test")
        assert result["entities"] == []


class TestCrawlResultsParallel:
    @patch("src.retrieval.knowledge_retrieval.scrape_full_article")
    def test_parallel_crawl(self, mock_scrape):
        from src.retrieval.knowledge_retrieval import crawl_results_parallel
        mock_scrape.return_value = "Crawled article content with enough words to pass the minimum filter check."
        results = [{"title": "A1", "url": "http://a.com/1", "snippet": "s1"}]
        docs = crawl_results_parallel(results, max_workers=1)
        assert len(docs) == 1

    def test_empty_results(self):
        from src.retrieval.knowledge_retrieval import crawl_results_parallel
        assert crawl_results_parallel([]) == []
