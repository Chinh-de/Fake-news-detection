"""Tests for src/utils.py"""
import os
import csv
import tempfile
import pytest


class TestPreprocessText:
    def test_lowercase(self):
        from src.utils import preprocess_text
        assert preprocess_text("HELLO WORLD") == "hello world"

    def test_remove_urls(self):
        from src.utils import preprocess_text
        result = preprocess_text("Check http://example.com now")
        assert "http" not in result
        assert "example" not in result

    def test_remove_https_urls(self):
        from src.utils import preprocess_text
        result = preprocess_text("Visit https://news.com/article")
        assert "https" not in result

    def test_remove_www_urls(self):
        from src.utils import preprocess_text
        result = preprocess_text("See www.example.com for details")
        assert "www" not in result

    def test_remove_mentions(self):
        from src.utils import preprocess_text
        result = preprocess_text("@user said something")
        assert "@user" not in result
        assert "said something" in result

    def test_keep_hashtags(self):
        from src.utils import preprocess_text
        result = preprocess_text("Trending #news today")
        assert "#news" in result

    def test_remove_special_chars(self):
        from src.utils import preprocess_text
        result = preprocess_text("Hello! World? Yes.")
        assert "!" not in result
        assert "?" not in result

    def test_normalize_whitespace(self):
        from src.utils import preprocess_text
        result = preprocess_text("too   many    spaces")
        assert result == "too many spaces"

    def test_non_string_input(self):
        from src.utils import preprocess_text
        result = preprocess_text(12345)
        assert result == "12345"

    def test_empty_string(self):
        from src.utils import preprocess_text
        assert preprocess_text("") == ""

    def test_complex_tweet(self):
        from src.utils import preprocess_text
        tweet = "BREAKING: @CNN reports http://t.co/abc123 shocking news!!! #fakenews"
        result = preprocess_text(tweet)
        assert "breaking" in result
        assert "@cnn" not in result
        assert "http" not in result
        assert "#fakenews" in result


class TestCleanQuery:
    def test_removes_punctuation(self):
        from src.utils import clean_query
        result = clean_query("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_collapse_whitespace(self):
        from src.utils import clean_query
        result = clean_query("too   many    spaces")
        assert result == "too many spaces"

    def test_unicode_normalization(self):
        from src.utils import clean_query
        # NFKC normalization converts fullwidth chars
        result = clean_query("Ｈｅｌｌｏ")
        assert result == "Hello"

    def test_empty_input(self):
        from src.utils import clean_query
        assert clean_query("") == ""


class TestTruncateText:
    def test_short_text_unchanged(self):
        from src.utils import truncate_text
        assert truncate_text("short", 10) == "short"

    def test_exact_length_unchanged(self):
        from src.utils import truncate_text
        text = "exact"
        assert truncate_text(text, 5) == text

    def test_truncate_at_word_boundary(self):
        from src.utils import truncate_text
        result = truncate_text("hello world foo bar", max_length=12)
        assert result.endswith("...")
        assert len(result.replace("...", "")) <= 12

    def test_truncate_single_long_word(self):
        from src.utils import truncate_text
        result = truncate_text("superlongword", max_length=5)
        assert result == "super..."

    def test_default_max_length(self):
        from src.utils import truncate_text
        long_text = "a " * 100
        result = truncate_text(long_text.strip())
        assert len(result) <= 53  # 50 + "..."


class TestLogRetrievalToCsv:
    def test_disabled_by_default(self):
        """Should do nothing when no filepath provided and env var is empty."""
        from src.utils import log_retrieval_to_csv
        # Should not raise
        log_retrieval_to_csv("test", "q", "t", "u", "s")

    def test_writes_csv_when_filepath_provided(self):
        from src.utils import log_retrieval_to_csv
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_log.csv")
            log_retrieval_to_csv("func1", "query1", "title1", "url1", "snippet1", filepath=filepath)
            log_retrieval_to_csv("func2", "query2", "title2", "url2", "snippet2", filepath=filepath)

            assert os.path.exists(filepath)
            with open(filepath, encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            # Header + 2 data rows
            assert len(rows) == 3
            assert rows[0] == ["source_function", "query", "title", "url", "snippet"]
            assert rows[1][0] == "func1"
            assert rows[2][0] == "func2"


class TestSetSeed:
    def test_set_seed_reproducibility(self):
        import random
        from src.utils import set_seed

        set_seed(123)
        val1 = random.random()
        set_seed(123)
        val2 = random.random()
        assert val1 == val2

    def test_different_seeds_different_results(self):
        import random
        from src.utils import set_seed

        set_seed(1)
        val1 = random.random()
        set_seed(2)
        val2 = random.random()
        assert val1 != val2
