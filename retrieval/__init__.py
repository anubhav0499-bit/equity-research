"""
Retrieval subsystem for the Equity Research Platform.

Provides:
  - Per-company vector store (LlamaIndex + ChromaDB) via vector_store module
  - LangGraph multi-agent RAG pipeline via rag_pipeline module
  - LangChain tools (web search, SEC EDGAR, financial snapshot) via tools module

Typical usage from any BaseAgent subclass:
    answer = self.rag_query("What is management's revenue guidance?", state)

Lazy imports
------------
vector_store requires llama_index + chromadb which may not be installed in
lightweight environments (evaluation, testing, CI). Imports are deferred to
first attribute access so the package is importable without the heavy deps.
"""

from __future__ import annotations


def __getattr__(name: str):
    _vs_names = frozenset(
        ("ingest_document", "ingest_texts", "vector_query", "collection_size", "clear_company")
    )
    _rp_names = frozenset(("rag_run", "rag_query"))

    if name in _vs_names:
        from .vector_store import (
            ingest_document,
            ingest_texts,
            collection_size,
            clear_company,
        )
        from .vector_store import query as vector_query

        _map = {
            "ingest_document": ingest_document,
            "ingest_texts":    ingest_texts,
            "vector_query":    vector_query,
            "collection_size": collection_size,
            "clear_company":   clear_company,
        }
        return _map[name]

    if name in _rp_names:
        from .rag_pipeline import run as rag_run, query as rag_query

        _map = {"rag_run": rag_run, "rag_query": rag_query}
        return _map[name]

    raise AttributeError(f"module 'equity_research.retrieval' has no attribute {name!r}")


__all__ = [
    "ingest_document",
    "ingest_texts",
    "vector_query",
    "collection_size",
    "clear_company",
    "rag_run",
    "rag_query",
]
