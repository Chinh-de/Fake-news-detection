"""Tests for src/pipeline/ modules (selection, finetune, evidence)"""
import pytest
import numpy as np
from unittest.mock import MagicMock


class TestSplitCleanNoisy:
    def test_clean_when_agree_and_confident(self):
        from src.pipeline.selection import split_clean_noisy
        sample = {"label_llm": 0, "label_slm": 0, "conf_slm": 0.95}
        assert split_clean_noisy(sample, 0.8) is True

    def test_noisy_when_disagree(self):
        from src.pipeline.selection import split_clean_noisy
        sample = {"label_llm": 0, "label_slm": 1, "conf_slm": 0.99}
        assert split_clean_noisy(sample, 0.8) is False

    def test_noisy_when_low_confidence(self):
        from src.pipeline.selection import split_clean_noisy
        sample = {"label_llm": 1, "label_slm": 1, "conf_slm": 0.5}
        assert split_clean_noisy(sample, 0.8) is False

    def test_clean_at_exact_threshold(self):
        from src.pipeline.selection import split_clean_noisy
        sample = {"label_llm": 1, "label_slm": 1, "conf_slm": 0.8}
        assert split_clean_noisy(sample, 0.8) is True

    def test_noisy_just_below_threshold(self):
        from src.pipeline.selection import split_clean_noisy
        sample = {"label_llm": 1, "label_slm": 1, "conf_slm": 0.79}
        assert split_clean_noisy(sample, 0.8) is False


class TestFinalizeRemainingNoisy:
    def test_finalizes_all_samples(self):
        from src.pipeline.selection import finalize_remaining_noisy_with_slm
        mock_slm = MagicMock()
        mock_slm.inference.return_value = (1, 0.85, np.array([0.15, 0.85]))
        d_noisy = [
            {"text": "noisy article 1", "label": None},
            {"text": "noisy article 2", "label": None},
        ]
        result = finalize_remaining_noisy_with_slm(d_noisy, mock_slm)
        assert len(result) == 2
        for r in result:
            assert r["label"] == 1
            assert r["status"] == "finalized_by_slm"

    def test_empty_noisy_returns_empty(self):
        from src.pipeline.selection import finalize_remaining_noisy_with_slm
        assert finalize_remaining_noisy_with_slm([], MagicMock()) == []

    def test_does_not_mutate_original(self):
        from src.pipeline.selection import finalize_remaining_noisy_with_slm
        mock_slm = MagicMock()
        mock_slm.inference.return_value = (0, 0.9, np.array([0.9, 0.1]))
        original = {"text": "test", "label": None}
        finalize_remaining_noisy_with_slm([original], mock_slm)
        assert original["label"] is None


class TestMaybeFinetuneSlmOnClean:
    def test_disabled(self):
        from src.pipeline.finetune import maybe_finetune_slm_on_clean
        mock_slm = MagicMock()
        result = maybe_finetune_slm_on_clean(
            slm=mock_slm, clean_pool=[{"text": "x", "label": 0}] * 20,
            round_id=1, enable_slm_finetune=False,
        )
        assert result["trained"] is False
        assert result["reason"] == "disabled"

    def test_insufficient_samples(self):
        from src.pipeline.finetune import maybe_finetune_slm_on_clean
        result = maybe_finetune_slm_on_clean(
            slm=MagicMock(), clean_pool=[{"text": "x", "label": 0}] * 5,
            round_id=1, slm_finetune_min_samples=16,
        )
        assert result["trained"] is False
        assert result["reason"] == "insufficient_samples"

    def test_calls_finetune(self):
        from src.pipeline.finetune import maybe_finetune_slm_on_clean
        mock_slm = MagicMock()
        mock_slm.finetune_on_clean.return_value = {
            "trained": True, "samples": 20, "epochs": 1,
            "batch_size": 32, "lr": 1e-5, "weight_decay": 1e-4, "avg_loss": 0.5,
        }
        pool = [{"text": f"s{i}", "label": i % 2} for i in range(20)]
        result = maybe_finetune_slm_on_clean(
            slm=mock_slm, clean_pool=pool, round_id=1, slm_finetune_min_samples=16,
        )
        assert result["trained"] is True
        mock_slm.finetune_on_clean.assert_called_once()
