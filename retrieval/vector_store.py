"""
Per-company vector store — LlamaIndex + ChromaDB backend.

Each company research run gets its own ChromaDB collection keyed by ticker,
so queries are always scoped to the company under analysis.

Public API
----------
ingest_document(text, metadata, ticker)  → int   # nodes added
ingest_texts(texts, metadatas, ticker)   → int
query(question, ticker, top_k)           → list[str]
collection_size(ticker)                  → int
clear_company(ticker)                    → None   # drop and recreate
"""

from __future__ import annotations
import threading
from pathlib import Path
from typing import Optional
from loguru import logger

import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from ..core.config import EMBEDDING_CONFIG, DB_CONFIG

# ── Module-level singletons, keyed by ticker ─────────────────────────────

_indices:  dict[str, VectorStoreIndex]         = {}
_clients:  dict[str, chromadb.PersistentClient] = {}
_colls:    dict[str, chromadb.Collection]       = {}
_locks:    dict[str, threading.Lock]            = {}
_settings_initialised = False
_settings_lock = threading.Lock()

_CHROMA_DIR = DB_CONFIG.faiss_dir.parent / "chroma_db"   # data/chroma_db


def _ensure_settings() -> None:
    global _settings_initialised
    if _settings_initialised:
        return
    with _settings_lock:
        if _settings_initialised:
            return
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_CONFIG.model_name,
            trust_remote_code=True,
        )
        Settings.chunk_size    = EMBEDDING_CONFIG.chunk_size
        Settings.chunk_overlap = EMBEDDING_CONFIG.chunk_overlap
        Settings.llm           = None   # retrieval-only; no LlamaIndex LLM needed
        _settings_initialised  = True
        logger.info(f"[retrieval] embeddings: {EMBEDDING_CONFIG.model_name}")


def _lock_for(ticker: str) -> threading.Lock:
    if ticker not in _locks:
        _locks[ticker] = threading.Lock()
    return _locks[ticker]


def _get_index(ticker: str) -> VectorStoreIndex:
    """Return (or create) the per-ticker vector store index."""
    if ticker in _indices:
        return _indices[ticker]

    with _lock_for(ticker):
        if ticker in _indices:
            return _indices[ticker]

        _ensure_settings()
        persist_path = _CHROMA_DIR / ticker.upper()
        persist_path.mkdir(parents=True, exist_ok=True)

        client     = chromadb.PersistentClient(path=str(persist_path))
        collection = client.get_or_create_collection(f"er_{ticker.lower()}")
        vs         = ChromaVectorStore(chroma_collection=collection)
        ctx        = StorageContext.from_defaults(vector_store=vs)
        index      = VectorStoreIndex.from_vector_store(vs, storage_context=ctx)

        _clients[ticker] = client
        _colls[ticker]   = collection
        _indices[ticker] = index
        logger.info(
            f"[retrieval] opened collection er_{ticker.lower()} "
            f"({collection.count()} docs) at {persist_path}"
        )
    return _indices[ticker]


# ── Public API ────────────────────────────────────────────────────────────

def ingest_texts(
    texts: list[str],
    metadatas: Optional[list[dict]] = None,
    ticker: str = "UNKNOWN",
) -> int:
    """Chunk and index raw text strings into the company's collection."""
    if not texts:
        return 0
    index = _get_index(ticker)
    splitter = SentenceSplitter(
        chunk_size=EMBEDDING_CONFIG.chunk_size,
        chunk_overlap=EMBEDDING_CONFIG.chunk_overlap,
    )
    docs = [
        Document(text=t, metadata=(metadatas[i] if metadatas and i < len(metadatas) else {}))
        for i, t in enumerate(texts)
    ]
    nodes = splitter.get_nodes_from_documents(docs)
    index.insert_nodes(nodes)
    logger.info(f"[retrieval:{ticker}] ingested {len(nodes)} nodes from {len(texts)} texts")
    return len(nodes)


def ingest_document(
    text: str,
    metadata: Optional[dict] = None,
    ticker: str = "UNKNOWN",
) -> int:
    """Chunk and index a single document (filing, transcript, report)."""
    return ingest_texts([text], [metadata or {}], ticker)


def query(
    question: str,
    ticker: str = "UNKNOWN",
    top_k: int = 5,
) -> list[str]:
    """Return the top-k most relevant chunks for a question, scoped to the ticker."""
    index = _get_index(ticker)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    nodes  = retriever.retrieve(question)
    chunks = [n.get_content() for n in nodes]
    logger.debug(f"[retrieval:{ticker}] {len(chunks)} chunks for: {question[:80]}")
    return chunks


def collection_size(ticker: str = "UNKNOWN") -> int:
    _get_index(ticker)
    coll = _colls.get(ticker)
    return coll.count() if coll else 0


def clear_company(ticker: str) -> None:
    """Drop and recreate the collection for a ticker (used between test runs)."""
    with _lock_for(ticker):
        client = _clients.get(ticker)
        if client:
            try:
                client.delete_collection(f"er_{ticker.lower()}")
            except Exception:
                pass
        for store in (_indices, _clients, _colls):
            store.pop(ticker, None)
    logger.info(f"[retrieval:{ticker}] collection cleared")
