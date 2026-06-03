# Retrieval package - lazy imports to avoid pulling in heavy dependencies


def __getattr__(name):
    import importlib
    if name == "demo_retrieval":
        return importlib.import_module("src.retrieval.demo_retrieval")
    if name == "knowledge_retrieval":
        return importlib.import_module("src.retrieval.knowledge_retrieval")
    if name == "knowledge_agent":
        return importlib.import_module("src.retrieval.knowledge_agent")

    if name in ("load_news_corpus", "search_news", "retrieve_demonstrations"):
        m = importlib.import_module("src.retrieval.demo_retrieval")
        return getattr(m, name)
    if name in ("analyze_claim_entities_and_query", "retrieve_fact_evidence", "get_fact_ranker"):
        m = importlib.import_module("src.retrieval.knowledge_retrieval")
        return getattr(m, name)
    if name in ("build_knowledge_bundle", "get_cached_knowledge_bundle_local"):
        m = importlib.import_module("src.retrieval.knowledge_agent")
        return getattr(m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "load_news_corpus", "search_news", "retrieve_demonstrations",
    "analyze_claim_entities_and_query", "retrieve_fact_evidence", "get_fact_ranker",
    "build_knowledge_bundle", "get_cached_knowledge_bundle_local",
]
