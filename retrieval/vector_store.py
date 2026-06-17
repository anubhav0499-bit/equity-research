"""
Per-company vector store — FAISS backend with multi-vector retrieval.

Architecture
------------
Each ticker gets its own FAISS index directory under data/faiss_db/<TICKER>/:

  child_index.faiss   — IndexFlatIP of child-chunk embeddings (size ~256 tokens)
  parent_index.faiss  — IndexFlatIP of parent-chunk embeddings (size ~1024 tokens)
  child_docs.json     — [{text, metadata, parent_id}] for each child chunk
  parent_docs.json    — [{text, metadata}] for each parent chunk

Multi-vector retrieval
----------------------
1. Search child_index for top-k × MULTIPLIER candidates (high precision, small chunks)
2. Map each retrieved child_id → parent_id (deduplicate)
3. Return parent chunk texts (rich context window)
4. Apply BM25 keyword re-ranking via Reciprocal Rank Fusion
5. Apply cross-encoder reranking if sentence-transformers is available

Embedding
---------
Model:  BAAI/bge-small-en-v1.5 (~130MB, free, local)
        Falls back to all-MiniLM-L6-v2 if bge-small is unavailable.
Vectors are L2-normalised before insertion; IndexFlatIP then gives cosine similarity.

Public API (unchanged from previous version)
--------------------------------------------
ingest_document(text, metadata, ticker)                  → int  (parent chunks added)
ingest_texts(texts, metadatas, ticker)                   → int
query(question, ticker, top_k, metadata_filter)          → list[str]
collection_size(ticker)                                  → int
clear_company(ticker)                                    → None
"""

from __future__ import annotations
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from ..core.config import DB_CONFIG, RAG_CONFIG
from .chunking import SmartChunker

# ── FAISS ─────────────────────────────────────────────────────────────────────

try:
    import faiss as _faiss
    HAS_FAISS = True
except ImportError:
    _faiss    = None   # type: ignore[assignment]
    HAS_FAISS = False
    logger.warning("[retrieval] faiss not installed — vector store disabled. Run: pip install faiss-cpu")

# ── Sentence-Transformers ──────────────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    HAS_ST = True
except ImportError:
    _SentenceTransformer = None   # type: ignore[assignment]
    HAS_ST = False
    logger.warning("[retrieval] sentence-transformers not installed. Run: pip install sentence-transformers")

# ── Optional cross-encoder ────────────────────────────────────────────────────

HAS_RERANKER = False
_reranker    = None
try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    _reranker    = _CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    HAS_RERANKER = True
    logger.info("[retrieval] cross-encoder reranker loaded (ms-marco-MiniLM-L-6-v2)")
except Exception:
    pass

# ── Module-level singletons ───────────────────────────────────────────────────

_model:               Optional[_SentenceTransformer] = None
_settings_initialised = False
_settings_lock        = threading.Lock()
_stores:              dict[str, "_TickerStore"] = {}
_store_locks:         dict[str, threading.Lock] = {}

_FAISS_DIR     = DB_CONFIG.faiss_dir  # data/faiss_index

_EMBED_SEM     = threading.Semaphore(max(1, min(os.cpu_count() or 2, 8)))
_CACHE_TTL     = 300
_query_cache:  dict[tuple, tuple[list[str], float]] = {}
_cache_lock    = threading.Lock()


# ── Embedding initialisation ──────────────────────────────────────────────────

def _ensure_settings() -> None:
    global _model, _settings_initialised
    if _settings_initialised:
        return
    with _settings_lock:
        if _settings_initialised:
            return
        if not HAS_ST or _SentenceTransformer is None:
            _settings_initialised = True
            return
        try:
            _model = _SentenceTransformer(RAG_CONFIG.model_name, trust_remote_code=True)
            logger.info(f"[retrieval] embedding model: {RAG_CONFIG.model_name}")
        except Exception as exc:
            logger.warning(f"[retrieval] {RAG_CONFIG.model_name} load failed ({exc}); trying fallback")
            try:
                _model = _SentenceTransformer(RAG_CONFIG.fallback_model)
                logger.info(f"[retrieval] embedding model (fallback): {RAG_CONFIG.fallback_model}")
            except Exception as exc2:
                logger.error(f"[retrieval] all embedding models failed: {exc2}")
        _settings_initialised = True


def _embed(texts: list[str]) -> np.ndarray:
    """Encode texts with L2-normalisation. Returns float32 array (n, d)."""
    _ensure_settings()
    if _model is None or not texts:
        return np.zeros((len(texts), 384), dtype=np.float32)
    with _EMBED_SEM:
        vecs = _model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
    return np.array(vecs, dtype=np.float32)


def _embed_fn_for_chunker(texts: list[str]) -> list:
    """Adapter for SmartChunker semantic mode (returns list of lists)."""
    return _embed(texts).tolist()


# ── TickerStore ───────────────────────────────────────────────────────────────

@dataclass
class _TickerStore:
    ticker:       str
    path:         Path
    child_index:  Optional[object]          = None   # faiss.Index
    parent_index: Optional[object]          = None   # faiss.Index
    child_docs:   list[dict]               = field(default_factory=list)
    parent_docs:  list[dict]               = field(default_factory=list)
    dim:          int                       = 384

    # ── Index helpers ──────────────────────────────────────────────────────

    def _new_index(self) -> object:
        return _faiss.IndexFlatIP(self.dim)

    def _load_or_create(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        ci_path = self.path / "child_index.faiss"
        pi_path = self.path / "parent_index.faiss"
        cd_path = self.path / "child_docs.json"
        pd_path = self.path / "parent_docs.json"

        if ci_path.exists() and cd_path.exists():
            try:
                self.child_index  = _faiss.read_index(str(ci_path))
                self.parent_index = _faiss.read_index(str(pi_path))
                self.child_docs   = json.loads(cd_path.read_text(encoding="utf-8"))
                self.parent_docs  = json.loads(pd_path.read_text(encoding="utf-8"))
                self.dim          = self.child_index.d
                logger.info(
                    f"[retrieval:{self.ticker}] loaded {len(self.parent_docs)} parent / "
                    f"{len(self.child_docs)} child chunks"
                )
                return
            except Exception as exc:
                logger.warning(f"[retrieval:{self.ticker}] index load failed ({exc}); recreating")

        self.child_index  = self._new_index()
        self.parent_index = self._new_index()
        self.child_docs   = []
        self.parent_docs  = []
        logger.info(f"[retrieval:{self.ticker}] created new FAISS index at {self.path}")

    def _save(self) -> None:
        if not HAS_FAISS:
            return
        try:
            _faiss.write_index(self.child_index,  str(self.path / "child_index.faiss"))
            _faiss.write_index(self.parent_index, str(self.path / "parent_index.faiss"))
            (self.path / "child_docs.json").write_text(
                json.dumps(self.child_docs,  ensure_ascii=False), encoding="utf-8"
            )
            (self.path / "parent_docs.json").write_text(
                json.dumps(self.parent_docs, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning(f"[retrieval:{self.ticker}] save failed: {exc}")

    # ── Ingest ─────────────────────────────────────────────────────────────

    def add_documents(self, texts: list[str], metadatas: list[dict]) -> int:
        if not HAS_FAISS or not texts:
            return 0

        parent_chunker = SmartChunker(
            mode=RAG_CONFIG.chunking_mode,
            chunk_size=RAG_CONFIG.parent_chunk_size,
            chunk_overlap=RAG_CONFIG.parent_chunk_overlap,
            embed_fn=_embed_fn_for_chunker,
            semantic_threshold=RAG_CONFIG.semantic_threshold,
        )
        child_chunker = SmartChunker(
            mode="recursive",   # always recursive for child chunks (precision)
            chunk_size=RAG_CONFIG.child_chunk_size,
            chunk_overlap=RAG_CONFIG.child_chunk_overlap,
        )

        new_parent_texts:   list[str]  = []
        new_parent_metas:   list[dict] = []
        new_child_texts:    list[str]  = []
        new_child_parent_ids: list[int] = []

        base_parent_id = len(self.parent_docs)

        for text, meta in zip(texts, metadatas):
            p_texts, p_metas = parent_chunker.split(text, meta)
            for p_idx, (p_text, p_meta) in enumerate(zip(p_texts, p_metas)):
                parent_id = base_parent_id + len(new_parent_texts)
                new_parent_texts.append(p_text)
                new_parent_metas.append(p_meta)

                c_texts, _ = child_chunker.split(p_text, p_meta)
                for c_text in c_texts:
                    new_child_texts.append(c_text)
                    new_child_parent_ids.append(parent_id)

        if not new_parent_texts:
            return 0

        # Embed and add to indices
        parent_vecs = _embed(new_parent_texts)
        child_vecs  = _embed(new_child_texts)  if new_child_texts else np.zeros((0, parent_vecs.shape[1]), dtype=np.float32)

        # Ensure index dimension matches
        if self.child_index.ntotal == 0 and self.parent_index.ntotal == 0:
            self.dim          = parent_vecs.shape[1]
            self.child_index  = _faiss.IndexFlatIP(self.dim)
            self.parent_index = _faiss.IndexFlatIP(self.dim)

        self.parent_index.add(parent_vecs)
        if child_vecs.shape[0] > 0:
            self.child_index.add(child_vecs)

        # Update docstores
        self.parent_docs.extend({"text": t, **m} for t, m in zip(new_parent_texts, new_parent_metas))
        base_child_id = len(self.child_docs)
        for c_text, p_id in zip(new_child_texts, new_child_parent_ids):
            self.child_docs.append({"text": c_text, "parent_id": p_id})

        self._save()
        logger.info(
            f"[retrieval:{self.ticker}] added {len(new_parent_texts)} parent / "
            f"{len(new_child_texts)} child chunks"
        )
        return len(new_parent_texts)

    def size(self) -> int:
        return len(self.parent_docs)

    def clear(self) -> None:
        self.child_index  = self._new_index()
        self.parent_index = self._new_index()
        self.child_docs   = []
        self.parent_docs  = []
        self._save()
        logger.info(f"[retrieval:{self.ticker}] store cleared")

    # ── Retrieval ──────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_vec:       np.ndarray,
        top_k:           int  = 5,
        metadata_filter: Optional[dict] = None,
    ) -> list[str]:
        """Multi-vector retrieval: search child index, return parent texts."""
        if not HAS_FAISS or self.child_index.ntotal == 0:
            return []

        candidate_k = max(top_k * RAG_CONFIG.candidate_multiplier, RAG_CONFIG.min_candidates)
        candidate_k = min(candidate_k, self.child_index.ntotal)

        # Stage 1: dense search on child index
        D, I = self.child_index.search(query_vec.reshape(1, -1), candidate_k)
        child_ids = [int(i) for i in I[0] if i >= 0 and i < len(self.child_docs)]

        # Map child → parent (deduplicate)
        seen_parents: set[int] = set()
        parent_ids_ordered: list[int] = []
        for cid in child_ids:
            pid = self.child_docs[cid].get("parent_id", cid)
            if pid not in seen_parents:
                seen_parents.add(pid)
                parent_ids_ordered.append(pid)

        # Load parent docs
        parent_candidates = []
        for pid in parent_ids_ordered:
            if pid < len(self.parent_docs):
                doc = self.parent_docs[pid]
                if metadata_filter:
                    if not all(doc.get(k) == v for k, v in metadata_filter.items()):
                        continue
                parent_candidates.append((pid, doc))

        if not parent_candidates:
            return []

        # Stage 2: BM25 re-rank via RRF (dense order already established)
        query_terms = [t.lower() for t in re.findall(r"\b\w{3,}\b",
                                                       _current_query.lower())]
        kw_scores  = [_bm25_tf(query_terms, d["text"]) for _, d in parent_candidates]
        dense_ord  = list(range(len(parent_candidates)))
        kw_ord     = sorted(dense_ord, key=lambda i: kw_scores[i], reverse=True)
        fused      = _rrf(dense_ord, kw_ord)

        pre_rerank_k = min(top_k * 2, len(parent_candidates))
        rerank_cands = [parent_candidates[i] for i in fused[:pre_rerank_k]]

        # Stage 3: cross-encoder reranking
        if HAS_RERANKER and _reranker is not None and len(rerank_cands) > top_k:
            try:
                pairs   = [(_current_query, d["text"][:512]) for _, d in rerank_cands]
                scores  = _reranker.predict(pairs)
                rerank_cands = [
                    c for _, c in sorted(zip(scores, rerank_cands), key=lambda x: x[0], reverse=True)
                ]
            except Exception as exc:
                logger.warning(f"[retrieval:{self.ticker}] cross-encoder failed: {exc}")

        return [d["text"] for _, d in rerank_cands[:top_k]]


# Thread-local to pass query text into the store's BM25 step
_current_query = ""


# ── BM25 + RRF helpers ────────────────────────────────────────────────────────

def _bm25_tf(query_terms: list[str], text: str, k1: float = 1.5, b: float = 0.75) -> float:
    if not query_terms or not text:
        return 0.0
    words   = text.lower().split()
    dl      = len(words)
    avg_dl  = 300.0
    score   = 0.0
    text_l  = text.lower()
    for term in query_terms:
        tf = text_l.count(term)
        if tf > 0:
            score += tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg_dl))
    return score


def _rrf(a: list[int], b: list[int], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for rank, idx in enumerate(a):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(b):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)


# ── Store accessor ────────────────────────────────────────────────────────────

def _get_store(ticker: str) -> _TickerStore:
    if ticker in _stores:
        return _stores[ticker]
    if ticker not in _store_locks:
        _store_locks[ticker] = threading.Lock()
    with _store_locks[ticker]:
        if ticker in _stores:
            return _stores[ticker]
        _ensure_settings()
        store_path = _FAISS_DIR / ticker.upper()
        store      = _TickerStore(ticker=ticker, path=store_path)
        store._load_or_create()
        _stores[ticker] = store
        return store


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_texts(
    texts:     list[str],
    metadatas: Optional[list[dict]] = None,
    ticker:    str = "UNKNOWN",
) -> int:
    if not texts:
        return 0
    metas = metadatas or [{} for _ in texts]
    store = _get_store(ticker)
    return store.add_documents(texts, metas)


def ingest_document(
    text:     str,
    metadata: Optional[dict] = None,
    ticker:   str = "UNKNOWN",
) -> int:
    return ingest_texts([text], [metadata or {}], ticker)


def query(
    question:        str,
    ticker:          str = "UNKNOWN",
    top_k:           int = 5,
    metadata_filter: Optional[dict] = None,
    hyde_vec:        Optional[np.ndarray] = None,
) -> list[str]:
    """
    Return the top-k most relevant parent chunks for a question.

    Pipeline:
      1. Cache check (5-min TTL)
      2. Embed question (or use pre-computed HyDE vector)
      3. Multi-vector retrieval: child index → parent docs
      4. BM25 keyword re-rank via RRF
      5. Cross-encoder reranking
      6. Cache result
    """
    global _current_query
    _filter_key = tuple(sorted((metadata_filter or {}).items()))
    cache_key   = (question, ticker, top_k, _filter_key)

    with _cache_lock:
        cached = _query_cache.get(cache_key)
        if cached is not None:
            result, ts = cached
            if time.time() - ts < _CACHE_TTL:
                logger.debug(f"[retrieval:{ticker}] cache hit: {question[:60]}")
                return result

    _ensure_settings()
    store = _get_store(ticker)

    if store.size() == 0:
        return []

    # Embed query (or use pre-computed HyDE vector)
    if hyde_vec is not None:
        q_vec = hyde_vec
    else:
        q_vec = _embed([question])[0]

    _current_query = question
    chunks = store.retrieve(q_vec, top_k=top_k, metadata_filter=metadata_filter)

    logger.debug(f"[retrieval:{ticker}] {len(chunks)} chunks for: {question[:80]}")

    with _cache_lock:
        _query_cache[cache_key] = (chunks, time.time())
        if len(_query_cache) > 500:
            now     = time.time()
            expired = [k for k, (_, ts) in _query_cache.items() if now - ts > _CACHE_TTL]
            for k in expired:
                del _query_cache[k]

    return chunks


def collection_size(ticker: str = "UNKNOWN") -> int:
    if ticker in _stores:
        return _stores[ticker].size()
    store_path = _FAISS_DIR / ticker.upper() / "parent_docs.json"
    if store_path.exists():
        try:
            return len(json.loads(store_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return 0


def clear_company(ticker: str) -> None:
    if ticker in _stores:
        _stores[ticker].clear()
        del _stores[ticker]
    # Wipe cache entries for this ticker
    with _cache_lock:
        stale = [k for k in _query_cache if k[1] == ticker]
        for k in stale:
            del _query_cache[k]
    logger.info(f"[retrieval:{ticker}] collection cleared")
