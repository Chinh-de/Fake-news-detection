"""
Evaluation metrics and visualization for MRCD Framework.
Classification report, confusion matrix, and model comparison plotting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)


def evaluate_and_plot(
    y_true,
    y_pred,
    labels=None,
    model_name="Model",
) -> dict:
    """
    Compute classification metrics and plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Class names for display
        model_name: Name for plot titles
        
    Returns:
        dict with accuracy, auc, classification_report (DataFrame), confusion_matrix
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Overall Metrics
    acc = accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = None

    # Classification Report
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T

    print(f"\n===== {model_name} =====")
    print(f"Accuracy: {acc:.4f}")
    if auc is not None:
        print(f"AUC     : {auc:.4f}")
    print("\n=== Precision / Recall / F1 per class ===")
    print(report_df.round(4))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )
    plt.figure(figsize=(5, 5))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()

    return {
        "accuracy": acc,
        "auc": auc,
        "classification_report": report_df,
        "confusion_matrix": cm,
    }


def compare_models(metrics_dict: dict, figsize=(10, 5)) -> pd.DataFrame:
    """
    Compare multiple models side-by-side.
    
    Args:
        metrics_dict: Dict of {model_name: metrics_result} from evaluate_and_plot
        figsize: Figure size for bar chart
        
    Returns:
        Comparison DataFrame
    """

    def extract_prf(metrics):
        report_df = metrics.get("classification_report")
        if report_df is None or "weighted avg" not in report_df.index:
            return float("nan"), float("nan"), float("nan")
        return (
            float(report_df.loc["weighted avg", "precision"]),
            float(report_df.loc["weighted avg", "recall"]),
            float(report_df.loc["weighted avg", "f1-score"]),
        )

    rows = []
    for name, metrics in metrics_dict.items():
        p, r, f1 = extract_prf(metrics)
        rows.append(
            {
                "Model": name,
                "Accuracy": float(metrics["accuracy"]),
                "Precision": p,
                "Recall": r,
                "F1": f1,
            }
        )

    comparison_df = pd.DataFrame(rows)
    print(comparison_df.round(4))

    ax = comparison_df.set_index("Model").plot(
        kind="bar", figsize=figsize, ylim=(0, 1)
    )
    ax.set_title("Model Comparison - Metrics")
    ax.set_ylabel("Score")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return comparison_df
