# Evaluation package - lazy imports


def __getattr__(name):
    if name in ("evaluate_and_plot", "compare_models"):
        from src.evaluation import metrics
        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["evaluate_and_plot", "compare_models"]
