"""Tests for src/prompts.py"""
import pytest


class TestBuildDualExtractionPrompt:
    def test_contains_input_text(self):
        from src.prompts import build_dual_extraction_prompt
        prompt = build_dual_extraction_prompt("test news article")
        assert "test news article" in prompt

    def test_contains_json_schema(self):
        from src.prompts import build_dual_extraction_prompt
        prompt = build_dual_extraction_prompt("some text")
        assert "entities" in prompt
        assert "query" in prompt

    def test_mentions_wikipedia(self):
        from src.prompts import build_dual_extraction_prompt
        prompt = build_dual_extraction_prompt("some text")
        assert "Wikipedia" in prompt

    def test_returns_string(self):
        from src.prompts import build_dual_extraction_prompt
        result = build_dual_extraction_prompt("input")
        assert isinstance(result, str)
        assert len(result) > 100


class TestBuildClassificationPromptWikiOnly:
    def test_contains_text_and_knowledge(self):
        from src.prompts import build_classification_prompt_wiki_only
        prompt = build_classification_prompt_wiki_only(
            text="Test article",
            knowledge_k="Entity: Test Org - A major organization",
            demos=[],
        )
        assert "Test article" in prompt
        assert "Entity Definitions" in prompt or "Test Org" in prompt

    def test_only_real_fake_labels(self):
        """Prompt should only mention Real and Fake, no synonym lists."""
        from src.prompts import build_classification_prompt_wiki_only
        prompt = build_classification_prompt_wiki_only(
            text="test", knowledge_k="info", demos=[]
        )
        assert "Real" in prompt
        assert "Fake" in prompt
        # Should NOT contain synonym labels
        assert "Hoax" not in prompt
        assert "Authentic" not in prompt
        assert "Fabricated" not in prompt

    def test_with_demos(self):
        from src.prompts import build_classification_prompt_wiki_only
        demos = [
            {"text": "Demo article 1", "label": "Verified"},
            {"text": "Demo article 2", "label": "Misleading"},
        ]
        prompt = build_classification_prompt_wiki_only(
            text="classify me", knowledge_k="info", demos=demos
        )
        assert "Example 1" in prompt
        assert "Example 2" in prompt
        assert "Demo article 1" in prompt

    def test_no_demos_placeholder(self):
        from src.prompts import build_classification_prompt_wiki_only
        prompt = build_classification_prompt_wiki_only(
            text="test", knowledge_k="info", demos=[]
        )
        assert "No examples provided" in prompt


class TestBuildClassificationPromptFull:
    def test_contains_background_knowledge(self):
        from src.prompts import build_classification_prompt_full
        prompt = build_classification_prompt_full(
            text="Test", knowledge_k="<VERIFIED_REPORTS>Some reports</VERIFIED_REPORTS>",
            demos=[],
        )
        assert "VERIFIED_REPORTS" in prompt or "Some reports" in prompt

    def test_only_real_fake_labels(self):
        from src.prompts import build_classification_prompt_full
        prompt = build_classification_prompt_full(
            text="test", knowledge_k="info", demos=[]
        )
        assert "Real" in prompt
        assert "Fake" in prompt
        assert "Bogus" not in prompt
        assert "Credible" not in prompt


class TestBuildClassificationPromptDispatcher:
    def test_wiki_only_mode(self):
        from src.prompts import build_classification_prompt
        prompt = build_classification_prompt(
            text="test", knowledge_k="info", demos=[], mode="wiki_only"
        )
        assert "Entity Definitions" in prompt or "Real" in prompt

    def test_full_mode(self):
        from src.prompts import build_classification_prompt
        prompt = build_classification_prompt(
            text="test", knowledge_k="info", demos=[], mode="full"
        )
        assert "Real" in prompt

    def test_default_mode_is_full(self):
        from src.prompts import build_classification_prompt
        prompt_default = build_classification_prompt(
            text="test", knowledge_k="info", demos=[]
        )
        prompt_full = build_classification_prompt(
            text="test", knowledge_k="info", demos=[], mode="full"
        )
        assert prompt_default == prompt_full

    def test_demo_text_truncated(self):
        """Demo text longer than 1000 chars should be truncated."""
        from src.prompts import build_classification_prompt
        long_demo_text = "x" * 2000
        demos = [{"text": long_demo_text, "label": "Test"}]
        prompt = build_classification_prompt(
            text="test", knowledge_k="info", demos=demos
        )
        # The demo text in prompt should be <= 1000 chars
        assert long_demo_text not in prompt  # full 2000 chars not present
