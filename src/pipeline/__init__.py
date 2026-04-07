# Pipeline package - lazy imports to avoid pulling in heavy dependencies
# Use: from src.pipeline.runner import run_mrcd_pipeline


def __getattr__(name):
    if name == "run_mrcd_pipeline":
        from src.pipeline.runner import run_mrcd_pipeline
        return run_mrcd_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["run_mrcd_pipeline"]
