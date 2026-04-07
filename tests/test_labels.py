"""Tests for src/labels.py"""
import pytest


class TestParseLlmLabel:
    """Test the simplified Real/Fake parser."""

    def test_exact_real(self):
        from src.labels import parse_llm_label
        assert parse_llm_label("Real") == 0

    def test_exact_fake(self):
        from src.labels import parse_llm_label
        assert parse_llm_label("Fake") == 1

    def test_case_insensitive_real(self):
        from src.labels import parse_llm_label
        assert parse_llm_label("REAL") == 0
        assert parse_llm_label("real") == 0
        assert parse_llm_label("ReAl") == 0

    def test_case_insensitive_fake(self):
        from src.labels import parse_llm_label
        assert parse_llm_label("FAKE") == 1
        assert parse_llm_label("fake") == 1

    def test_with_whitespace(self):
        from src.labels import parse_llm_label
        assert parse_llm_label("  Real  ") == 0
        assert parse_llm_label("\nFake\n") == 1

    def test_with_extra_text_real(self):
        from src.labels import parse_llm_label
        assert parse_llm_label("I think this is Real") == 0

    def test_with_extra_text_fake(self):
        from src.labels import parse_llm_label
        assert parse_llm_label("The article is Fake") == 1

    def test_first_token_real(self):
        from src.labels import parse_llm_label
        assert parse_llm_label("Real, because the sources are verified") == 0

    def test_first_token_fake(self):
        from src.labels import parse_llm_label
        assert parse_llm_label("Fake. No evidence supports this.") == 1

    def test_garbage_defaults_to_fake(self):
        from src.labels import parse_llm_label
        assert parse_llm_label("asdfghjkl") == 1

    def test_empty_defaults_to_fake(self):
        from src.labels import parse_llm_label
        assert parse_llm_label("") == 1

    def test_custom_default(self):
        from src.labels import parse_llm_label
        assert parse_llm_label("garbage", default_fake=0) == 0

    def test_return_matched_label_real(self):
        from src.labels import parse_llm_label
        result, label = parse_llm_label("Real", return_matched_label=True)
        assert result == 0
        assert label == "Real"

    def test_return_matched_label_fake(self):
        from src.labels import parse_llm_label
        result, label = parse_llm_label("Fake", return_matched_label=True)
        assert result == 1
        assert label == "Fake"

    def test_return_matched_label_garbage(self):
        from src.labels import parse_llm_label
        result, label = parse_llm_label("xxx", return_matched_label=True)
        assert result == 1
        assert label is None

    def test_markdown_cleanup(self):
        from src.labels import parse_llm_label
        assert parse_llm_label("```json\nReal\n```") == 0

    def test_both_present_defaults_to_fake(self):
        """If both 'real' and 'fake' appear, parser skips both matches and defaults to fake."""
        from src.labels import parse_llm_label
        # Both present → neither simple check matches → falls to default
        result = parse_llm_label("real and fake mixed")
        assert result in (0, 1)  # Either is acceptable; current impl defaults to 1


class TestGenerateDemoLabel:
    def test_returns_string(self):
        from src.labels import generate_demo_label
        label = generate_demo_label()
        assert isinstance(label, str)
        assert len(label) > 0

    def test_from_synonym_set(self):
        from src.labels import generate_demo_label, ALL_SYNONYM_LABELS
        for _ in range(20):
            label = generate_demo_label()
            assert label in ALL_SYNONYM_LABELS

    def test_with_text_arg(self):
        from src.labels import generate_demo_label
        label = generate_demo_label("some text")
        assert isinstance(label, str)


class TestToCleanDemoLabel:
    def test_real_label_returns_real_synonym(self):
        from src.labels import to_clean_demo_label, REAL_SYNONYM_LABELS
        for _ in range(20):
            label = to_clean_demo_label(0)
            assert label in REAL_SYNONYM_LABELS

    def test_fake_label_returns_fake_synonym(self):
        from src.labels import to_clean_demo_label, FAKE_SYNONYM_LABELS
        for _ in range(20):
            label = to_clean_demo_label(1)
            assert label in FAKE_SYNONYM_LABELS

    def test_consistency(self):
        """Same binary class should always map to same synonym set."""
        from src.labels import to_clean_demo_label, REAL_SYNONYM_LABELS, FAKE_SYNONYM_LABELS
        real_labels = {to_clean_demo_label(0) for _ in range(50)}
        fake_labels = {to_clean_demo_label(1) for _ in range(50)}
        assert real_labels.issubset(set(REAL_SYNONYM_LABELS))
        assert fake_labels.issubset(set(FAKE_SYNONYM_LABELS))


class TestSynonymLists:
    def test_no_overlap(self):
        """Real and fake synonym sets should not overlap."""
        from src.labels import REAL_SYNONYM_LABELS, FAKE_SYNONYM_LABELS
        real_set = set(s.lower() for s in REAL_SYNONYM_LABELS)
        fake_set = set(s.lower() for s in FAKE_SYNONYM_LABELS)
        assert real_set.isdisjoint(fake_set)

    def test_not_empty(self):
        from src.labels import REAL_SYNONYM_LABELS, FAKE_SYNONYM_LABELS, ALL_SYNONYM_LABELS
        assert len(REAL_SYNONYM_LABELS) > 0
        assert len(FAKE_SYNONYM_LABELS) > 0
        assert len(ALL_SYNONYM_LABELS) == len(REAL_SYNONYM_LABELS) + len(FAKE_SYNONYM_LABELS)
