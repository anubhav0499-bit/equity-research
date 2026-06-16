"""
Per-company vector store — LlamaIndex + ChromaDB backend.

Each company research run gets its own ChromaDB collection keyed by ticker,
so queries are always scoped to the company under analysis.

Retrieval pipeline
------------------
1. Dense retrieval: VectorIndexRetriever (top-k × 4 candidates, min 20)
2. Hybrid re-ranking: BM25-style keyword scoring merged with dense rank via
   Reciprocal Rank Fusion — no extra packages required
3. Cross-encoder reranking: sentence-transformers CrossEncoder, if installed
   (pip install sentence-transformers)
4. Metadata filtering (by doc_type, fiscal_period, etc.) applied post-retrieval

Public API
----------
ingest_document(text, metadata, ticker)                   → int  (nodes added)
ingest_texts(texts, metadatas, ticker)                    → int
query(question, ticker, top_k, metadata_filter)           → list[str]
collection_size(ticker)                                   → int
clear_company(ticker)                                     → None
"""

from __future__ import annotations
import re
import threading
import time
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

_indices:  dict[str, VectorStoreIndex]          = {}
_clients:  dict[str, chromadb.PersistentClient] = {}
_colls:    dict[str, chromadb.Collection]        = {}
_locks:    dict[str, threading.Lock]             = {}
_settings_initialised = False
_settings_lock = threading.Lock()

_CHROMA_DIR = DB_CONFIG.faiss_dir.parent / "chroma_db"   # data/chroma_db

# Candidate multiplier: retrieve this many more chunks than top_k before reranking
_CANDIDATE_MULTIPLIER = 4
_MIN_CANDIDATES       = 20

# ── Concurrency controls ──────────────────────────────────────────────────
# BGE-large on CPU holds the GIL during torch inference. Limiting concurrent
# embedding calls prevents 10-thread contention from multiplying latency 7×.
import os as _os
# Allow up to 8 concurrent BGE encodings — higher than the old cap of 4
# so P95 under 10 concurrent users stays under 2 s on modern multi-core CPUs.
_EMBED_CONCURRENCY = max(1, min(_os.cpu_count() or 2, 8))
_embed_sem  = threading.Semaphore(_EMBED_CONCURRENCY)

# ── Query result cache ────────────────────────────────────────────────────
# Caches retrieval results by (question, ticker, top_k, filter) for 5 min.
# Eliminates redundant BGE encode+ChromaDB round-trips for repeated queries.
_CACHE_TTL   = 300  # seconds
_query_cache: dict[tuple, tuple[list[str], float]] = {}
_cache_lock  = threading.Lock()

# ── Optional cross-encoder reranker ──────────────────────────────────────
# Enable with: pip install sentence-transformers
HAS_RERANKER = False
_reranker    = None

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    _reranker    = _CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    HAS_RERANKER = True
    logger.info("[retrieval] cross-encoder reranker loaded (ms-marco-MiniLM-L-6-v2)")
except Exception:
    pass


# ── Settings initialisation (once, thread-safe) ───────────────────────────

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
        logger.info(
            f"[retrieval] embeddings: {EMBEDDING_CONFIG.model_name} "
            f"chunk={EMBEDDING_CONFIG.chunk_size} overlap={EMBEDDING_CONFIG.chunk_overlap}"
        )


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
            f"({collection.count()} chunks) at {persist_path}"
        )
    return _indices[ticker]


# ── Hybrid re-ranking helpers ─────────────────────────────────────────────

def _keyword_score(query_terms: list[str], text: str) -> float:
    """
    Simplified BM25 term-frequency scoring (no IDF — corpus statistics unavailable).
    Provides keyword-match signal to complement dense semantic retrieval.
    """
    if not query_terms or not text:
        return 0.0
    text_lower = text.lower()
    words      = text_lower.split()
    dl         = len(words)
    if dl == 0:
        return 0.0
    avg_dl = 200.0
    k1, b  = 1.5, 0.75
    score  = 0.0
    for term in query_terms:
        tf = text_lower.count(term)
        if tf > 0:
            tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg_dl))
            score  += tf_norm
    return score


def _reciprocal_rank_fusion(
    dense_order:   list[int],
    keyword_order: list[int],
    k: int = 60,
) -> list[int]:
    """Merge two ranked index lists via Reciprocal Rank Fusion (RRF)."""
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_order):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(keyword_order):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)


# ── Public API ────────────────────────────────────────────────────────────

def ingest_texts(
    texts:     list[str],
    metadatas: Optional[list[dict]] = None,
    ticker:    str = "UNKNOWN",
) -> int:
    """Chunk and index raw text strings into the company's collection."""
    if not texts:
        return 0
    index    = _get_index(ticker)
    splitter = SentenceSplitter(
        chunk_size    = EMBEDDING_CONFIG.chunk_size,
        chunk_overlap = EMBEDDING_CONFIG.chunk_overlap,
    )
    docs = [
        Document(
            text     = t,
            metadata = (metadatas[i] if metadatas and i < len(metadatas) else {}),
        )
        for i, t in enumerate(texts)
    ]
    nodes = splitter.get_nodes_from_documents(docs)
    index.insert_nodes(nodes)
    logger.info(f"[retrieval:{ticker}] ingested {len(nodes)} chunks from {len(texts)} texts")
    return len(nodes)


def ingest_document(
    text:     str,
    metadata: Optional[dict] = None,
    ticker:   str = "UNKNOWN",
) -> int:
    """Chunk and index a single document (filing, transcript, report)."""
    return ingest_texts([text], [metadata or {}], ticker)


def query(
    question:        str,
    ticker:          str = "UNKNOWN",
    top_k:           int = 5,
    metadata_filter: Optional[dict] = None,
) -> list[str]:
    """
    Return the top-k most relevant chunks for a question, scoped to the ticker.

    Pipeline:
      1. Cache check — return cached result if available (5-min TTL)
      2. Dense retrieval — VectorIndexRetriever(top_k × 4, min 20 candidates)
         (gated by semaphore to limit concurrent BGE encoding on CPU)
      3. Metadata filtering — post-retrieval filter by any metadata field
      4. Keyword re-ranking — BM25-style TF scoring merged with dense rank via RRF
      5. Cross-encoder reranking — if sentence-transformers is installed
      6. Return top-k (result is cached for future calls)
    """
    # Stage 0: cache lookup
    _filter_key = tuple(sorted((metadata_filter or {}).items()))
    cache_key   = (question, ticker, top_k, _filter_key)
    with _cache_lock:
        cached = _query_cache.get(cache_key)
        if cached is not None:
            result, ts = cached
            if time.time() - ts < _CACHE_TTL:
                logger.debug(f"[retrieval:{ticker}] cache hit for: {question[:60]}")
                return result

    index       = _get_index(ticker)
    candidate_k = max(top_k * _CANDIDATE_MULTIPLIER, _MIN_CANDIDATES)

    # Stage 1: dense retrieval — semaphore limits concurrent BGE encoding
    with _embed_sem:
        dense_retriever = VectorIndexRetriever(index=index, similarity_top_k=candidate_k)
        nodes = dense_retriever.retrieve(question)

    if not nodes:
        return []

    # Stage 2: metadata filtering
    if metadata_filter:
        nodes = [
            n for n in nodes
            if all(n.metadata.get(k) == v for k, v in metadata_filter.items())
        ]
        if not nodes:
            return []

    # Stage 3: BM25-style keyword re-ranking via RRF
    query_terms   = [t.lower() for t in re.findall(r'\b\w{3,}\b', question.lower())]
    kw_scores     = [_keyword_score(query_terms, n.get_content()) for n in nodes]
    dense_order   = list(range(len(nodes)))                              # already ranked best→worst
    keyword_order = sorted(range(len(nodes)), key=lambda i: kw_scores[i], reverse=True)
    fused_order   = _reciprocal_rank_fusion(dense_order, keyword_order)

    # Keep 2× top_k for the reranker (trim to top_k afterwards)
    pre_rerank_k = min(top_k * 2, len(nodes))
    candidates   = [nodes[i] for i in fused_order[:pre_rerank_k]]

    # Stage 4: cross-encoder reranking (if available)
    if HAS_RERANKER and _reranker is not None and len(candidates) > top_k:
        try:
            pairs     = [(question, n.get_content()[:512]) for n in candidates]
            ce_scores = _reranker.predict(pairs)
            candidates = [
                n for _, n in sorted(zip(ce_scores, candidates),
                                     key=lambda x: x[0], reverse=True)
            ]
        except Exception as e:
            logger.warning(f"[retrieval:{ticker}] cross-encoder reranking failed: {e}")

    chunks = [n.get_content() for n in candidates[:top_k]]
    logger.debug(f"[retrieval:{ticker}] {len(chunks)} chunks for: {question[:80]}")

    # Cache the result
    with _cache_lock:
        _query_cache[cache_key] = (chunks, time.time())
        # Evict entries older than TTL to prevent unbounded growth
        if len(_query_cache) > 500:
            now = time.time()
            expired = [k for k, (_, ts) in _query_cache.items() if now - ts > _CACHE_TTL]
            for k in expired:
                del _query_cache[k]

    return chunks


def collection_size(ticker: str = "UNKNOWN") -> int:
    # Fast path: if the ChromaDB collection is already open, count without re-init
    if ticker in _colls:
        return _colls[ticker].count()
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
