"""Shared hybrid retrieval and lazy embedding refresh for finance long facts."""

from __future__ import annotations

from typing import Any

from .ollama import embed_text, embed_texts


def search_long_facts(
    query_text: str,
    memory_module: Any,
    ollama_url: str,
    embedding_model: str,
    timeout: int = 600,
    limit: int = 5,
    backfill_limit: int = 1000,
    batch_size: int = 64,
) -> tuple[list[dict[str, Any]], str]:
    """Hybrid-search facts; lazily batch missing/mismatched embeddings and fall back to BM25."""
    if not query_text or len(query_text) > 2000:
        raise ValueError("Long-fact query must contain between 1 and 2000 characters")
    limit = min(10, max(1, int(limit)))
    candidate_limit = min(50, max(20, limit * 4))
    vector_results: list[dict[str, Any]] = []
    try:
        query_vector = embed_text(query_text, ollama_url, embedding_model, timeout)
        pending = memory_module.get_long_facts_needing_embedding(embedding_model, limit=backfill_limit)
        for start in range(0, len(pending), max(1, batch_size)):
            batch = pending[start : start + max(1, batch_size)]
            try:
                vectors = embed_texts(
                    [fact["fact_text"] for fact in batch],
                    ollama_url,
                    embedding_model,
                    timeout,
                )
                for fact, vector in zip(batch, vectors):
                    memory_module.set_long_fact_embedding(fact["fact_key"], vector, embedding_model)
            except Exception as exc:
                print(f"Could not backfill a long-fact embedding batch: {exc}")
        vector_results = memory_module.search_long_facts(query_vector, embedding_model, limit=candidate_limit)
    except Exception as exc:
        print(f"Long-fact vector retrieval unavailable; using BM25 fallback: {exc}")

    lexical_results = memory_module.search_long_facts_bm25(query_text, limit=candidate_limit)
    results = memory_module.combine_fact_search_results(vector_results, lexical_results, limit=limit)
    return results, "vector+bm25_rrf" if vector_results else "bm25_fallback"
