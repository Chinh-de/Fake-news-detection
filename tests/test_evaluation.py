"""Tests for src/evaluation/metrics.py"""
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing


class TestEvaluateAndPlot:
    def test_perfect_predictions(self):
        from src.evaluation.metrics import evaluate_and_plot
        y_true = [0, 0, 1, 1, 0, 1]
        y_pred = [0, 0, 1, 1, 0, 1]
        result = evaluate_and_plot(y_true, y_pred, labels=["Real", "Fake"], model_name="Test")
        assert result["accuracy"] == 1.0
        assert result["auc"] == 1.0
        assert result["confusion_matrix"] is not None
        assert result["classification_report"] is not None

    def test_random_predictions(self):
        from src.evaluation.metrics import evaluate_and_plot
        y_true = [0, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        result = evaluate_and_plot(y_true, y_pred, labels=["R", "F"], model_name="T")
        assert 0 <= result["accuracy"] <= 1.0

    def test_all_same_class(self):
        from src.evaluation.metrics import evaluate_and_plot
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        # Don't pass labels when only 1 class is present (sklearn raises ValueError)
        result = evaluate_and_plot(y_true, y_pred, labels=None, model_name="T")
        assert result["accuracy"] == 1.0

    def test_returns_dataframe_report(self):
        import pandas as pd
        from src.evaluation.metrics import evaluate_and_plot
        result = evaluate_and_plot([0, 1], [0, 1], labels=["R", "F"])
        assert isinstance(result["classification_report"], pd.DataFrame)

    def test_confusion_matrix_shape(self):
        from src.evaluation.metrics import evaluate_and_plot
        result = evaluate_and_plot([0, 0, 1, 1], [0, 1, 0, 1], labels=["R", "F"])
        assert result["confusion_matrix"].shape == (2, 2)


class TestCompareModels:
    def _make_metrics(self, acc):
        import pandas as pd
        report = pd.DataFrame({
            "precision": {"Real": 0.8, "Fake": 0.9, "weighted avg": 0.85},
            "recall": {"Real": 0.7, "Fake": 0.95, "weighted avg": 0.82},
            "f1-score": {"Real": 0.75, "Fake": 0.92, "weighted avg": 0.83},
            "support": {"Real": 50, "Fake": 50, "weighted avg": 100},
        })
        return {"accuracy": acc, "auc": 0.85, "classification_report": report, "confusion_matrix": np.eye(2)}

    def test_compare_two_models(self):
        from src.evaluation.metrics import compare_models
        metrics = {
            "Model A": self._make_metrics(0.80),
            "Model B": self._make_metrics(0.90),
        }
        df = compare_models(metrics)
        assert len(df) == 2
        assert "Model A" in df["Model"].values
        assert "Model B" in df["Model"].values
        assert df.loc[df["Model"] == "Model B", "Accuracy"].values[0] == 0.90

    def test_single_model(self):
        from src.evaluation.metrics import compare_models
        df = compare_models({"Solo": self._make_metrics(0.75)})
        assert len(df) == 1
