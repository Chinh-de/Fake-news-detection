"""Tests for src/retrieval/knowledge_agent.py"""
import pytest
from unittest.mock import patch, MagicMock

# Skip if wikipedia not installed
pytest.importorskip("wikipedia", reason="wikipedia not installed")
pytest.importorskip("ddgs", reason="ddgs not installed")


class TestFormatVerifiedReports:
    def test_empty_chunks(self):
        from src.retrieval.knowledge_agent import format_verified_reports
        result = format_verified_reports([])
        assert "No verified report found" in result

    def test_with_chunks(self):
        from src.retrieval.knowledge_agent import format_verified_reports
        chunks = [{"title": "AP News", "chunk_text": "Important fact"}]
        result = format_verified_reports(chunks)
        assert "AP News" in result


class TestFormatEntityDefinitions:
    def test_empty_dict(self):
        from src.retrieval.knowledge_agent import format_entity_definitions
        assert "N/A" in format_entity_definitions({})

    def test_with_entities(self):
        from src.retrieval.knowledge_agent import format_entity_definitions
        result = format_entity_definitions({"NASA": "Space agency"})
        assert "NASA" in result


class TestQueryWikipedia:
    @patch("src.retrieval.knowledge_agent.wikipedia")
    def test_successful_query(self, mock_wiki):
        from src.retrieval.knowledge_agent import query_wikipedia
        mock_wiki.summary.return_value = "NASA is a space agency."
        assert "NASA" in query_wikipedia("NASA")

    @patch("src.retrieval.knowledge_agent.wikipedia")
    def test_not_found(self, mock_wiki):
        from src.retrieval.knowledge_agent import query_wikipedia
        mock_wiki.summary.side_effect = Exception("Not found")
        assert query_wikipedia("XYZ") == "Not found"


class TestExtractWikiKnowledge:
    @patch("src.retrieval.knowledge_agent.query_wikipedia")
    def test_valid_entities(self, mock_query):
        from src.retrieval.knowledge_agent import extract_wiki_knowledge_from_entities
        mock_query.return_value = "A definition"
        result = extract_wiki_knowledge_from_entities(["Entity1"])
        assert "Entity1" in result

    @patch("src.retrieval.knowledge_agent.query_wikipedia")
    def test_filters_not_found(self, mock_query):
        from src.retrieval.knowledge_agent import extract_wiki_knowledge_from_entities
        mock_query.return_value = "Not found"
        assert len(extract_wiki_knowledge_from_entities(["Missing"])) == 0

    def test_empty_entities(self):
        from src.retrieval.knowledge_agent import extract_wiki_knowledge_from_entities
        assert extract_wiki_knowledge_from_entities([]) == {}


class TestCachedKnowledgeBundle:
    @patch("src.retrieval.knowledge_agent.build_knowledge_bundle")
    def test_caches_result(self, mock_build):
        from src.retrieval.knowledge_agent import get_cached_knowledge_bundle_local
        mock_build.return_value = {"combined_text": "cached", "mode": "full"}
        cache = {}
        get_cached_knowledge_bundle_local("text1", cache, mode="full")
        get_cached_knowledge_bundle_local("text1", cache, mode="full")
        assert mock_build.call_count == 1

    @patch("src.retrieval.knowledge_agent.build_knowledge_bundle")
    def test_different_keys(self, mock_build):
        from src.retrieval.knowledge_agent import get_cached_knowledge_bundle_local
        mock_build.return_value = {"combined_text": "r", "mode": "full"}
        cache = {}
        get_cached_knowledge_bundle_local("text1", cache, mode="full")
        get_cached_knowledge_bundle_local("text2", cache, mode="full")
        assert mock_build.call_count == 2
