# Retrieval package - lazy imports to avoid pulling in heavy dependencies


def __getattr__(name):
    if name in ("load_news_corpus", "search_news", "retrieve_demonstrations"):
        from src.retrieval import demo_retrieval
        return getattr(demo_retrieval, name)
    if name in ("analyze_claim_entities_and_query", "retrieve_fact_evidence", "get_fact_ranker"):
        from src.retrieval import knowledge_retrieval
        return getattr(knowledge_retrieval, name)
    if name in ("build_knowledge_bundle", "get_cached_knowledge_bundle_local"):
        from src.retrieval import knowledge_agent
        return getattr(knowledge_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "load_news_corpus", "search_news", "retrieve_demonstrations",
    "analyze_claim_entities_and_query", "retrieve_fact_evidence", "get_fact_ranker",
    "build_knowledge_bundle", "get_cached_knowledge_bundle_local",
]
