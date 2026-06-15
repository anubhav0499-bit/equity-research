"""
Retrieval subsystem for the Equity Research Platform.

Provides:
  - Per-company vector store (LlamaIndex + ChromaDB) via vector_store module
  - LangGraph multi-agent RAG pipeline via rag_pipeline module
  - LangChain tools (web search, SEC EDGAR, financial snapshot) via tools module

Typical usage from any BaseAgent subclass:
    answer = self.rag_query("What is management's revenue guidance?", state)
"""

from .vector_store import (
    ingest_document,
    ingest_texts,
    query as vector_query,
    collection_size,
    clear_company,
)
from .rag_pipeline import run as rag_run, query as rag_query

__all__ = [
    "ingest_document",
    "ingest_texts",
    "vector_query",
    "collection_size",
    "clear_company",
    "rag_run",
    "rag_query",
]
