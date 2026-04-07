"""Tests for src/retrieval/demo_retrieval.py"""
import pytest
from unittest.mock import patch, MagicMock

# Skip if rank_bm25 or ddgs not installed
rank_bm25 = pytest.importorskip("rank_bm25", reason="rank_bm25 not installed")
ddgs = pytest.importorskip("ddgs", reason="ddgs not installed")


class TestRetrieveDemonstrations:
    def test_empty_corpus_returns_empty(self):
        from src.retrieval.demo_retrieval import retrieve_demonstrations
        assert retrieve_demonstrations("test query", [], k=4) == []

    def test_returns_k_demos(self):
        from src.retrieval.demo_retrieval import retrieve_demonstrations
        corpus = [
            "climate change global warming effects",
            "stock market financial news today",
            "new vaccine research medical breakthrough",
            "election results political campaign update",
            "sports football championship final game",
        ]
        demos = retrieve_demonstrations("climate research science", corpus, k=3)
        assert len(demos) == 3

    def test_demo_structure(self):
        from src.retrieval.demo_retrieval import retrieve_demonstrations
        corpus = ["test document about news"]
        demos = retrieve_demonstrations("test", corpus, k=1)
        assert len(demos) == 1
        assert "text" in demos[0]
        assert "label" in demos[0]
        assert "source" in demos[0]
        assert demos[0]["source"] == "Bing/Retrieved"

    def test_k_larger_than_corpus(self):
        from src.retrieval.demo_retrieval import retrieve_demonstrations
        corpus = ["only one doc"]
        demos = retrieve_demonstrations("test", corpus, k=4)
        assert len(demos) == 1

    def test_bm25_relevance_ordering(self):
        from src.retrieval.demo_retrieval import retrieve_demonstrations
        corpus = [
            "unrelated sports football game",
            "climate change global warming temperature rise",
            "cooking recipe pasta italian food",
        ]
        demos = retrieve_demonstrations("climate global warming", corpus, k=1)
        assert "climate" in demos[0]["text"].lower()


class TestSearchNews:
    @patch("src.retrieval.demo_retrieval.DDGS")
    def test_search_handles_exception(self, mock_ddgs_cls):
        from src.retrieval.demo_retrieval import search_news
        mock_ddgs_cls.side_effect = Exception("Network error")
        results = search_news("test")
        assert results == []


class TestLoadNewsCorpus:
    @patch("src.retrieval.demo_retrieval.requests.get")
    def test_load_success(self, mock_get):
        from src.retrieval.demo_retrieval import load_news_corpus
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = MagicMock()
        mock_get.return_value.text = '1,"Title A","Desc A"\n2,"Title B","Desc B"'
        result = load_news_corpus("http://fake-url.com/data.csv")
        assert len(result) == 2

    @patch("src.retrieval.demo_retrieval.requests.get")
    def test_load_failure_returns_empty(self, mock_get):
        from src.retrieval.demo_retrieval import load_news_corpus
        mock_get.side_effect = Exception("Connection error")
        result = load_news_corpus("http://fake-url.com/data.csv")
        assert result == []
