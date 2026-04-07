from src.llm.base import BaseLLM

# Lazy import for handler to avoid pulling torch at import time
def __getattr__(name):
    if name in ("LocalLLM", "get_llm"):
        from src.llm import handler
        return getattr(handler, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["BaseLLM", "LocalLLM", "get_llm"]
