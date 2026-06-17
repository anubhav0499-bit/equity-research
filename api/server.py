"""
Equity Research RAG Platform — Production HTTP API

Endpoints
---------
GET  /health          Liveness + dependency check (LLM, vector store, model)
POST /query           Run RAG pipeline for one analyst question (sync)
POST /stream          Stream RAG response token-by-token via SSE
POST /ingest          Ingest a document into the per-ticker vector store
GET  /collection/{t}  Chunk count for a ticker's knowledge base
DELETE /collection/{t} Drop a ticker's knowledge base

SSE streaming format (POST /stream):
    Each event: data: <token>\\n\\n
    Final event: data: [DONE]\\n\\n

Run:
    uvicorn equity_research.api.server:app --host 0.0.0.0 --port 8000 --workers 2
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from ..core.config import validate_llm_config, EMBEDDING_CONFIG

app = FastAPI(
    title="Equity Research RAG API",
    version="1.0.0",
    description="Institutional-grade equity research retrieval-augmented generation platform.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ── Startup validation ────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    """Fail fast if LLM provider or embedding model are misconfigured."""
    try:
        provider = validate_llm_config()
        logger.info(f"[api] LLM provider: {provider}")
    except RuntimeError as e:
        logger.error(f"[api] LLM configuration error: {e}")
        # Allow startup so /health returns 503 with a useful message rather than crashing

    try:
        from ..retrieval.vector_store import _ensure_settings
        _ensure_settings()
        logger.info(f"[api] Embedding model loaded: {EMBEDDING_CONFIG.model_name}")
    except Exception as e:
        logger.error(f"[api] Embedding model init failed: {e}")


# ── Request / Response models ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:     str        = Field(..., min_length=3, max_length=2000)
    ticker:       str        = Field(..., min_length=1, max_length=10)
    company_name: str        = Field("", max_length=200)
    session_id:   str        = Field("", max_length=128,
                                     description="Opaque session ID for conversation memory. "
                                                 "Leave blank for stateless queries.")

class StreamQueryRequest(BaseModel):
    question:     str        = Field(..., min_length=3, max_length=2000)
    ticker:       str        = Field(..., min_length=1, max_length=10)
    company_name: str        = Field("", max_length=200)
    session_id:   str        = Field("", max_length=128)

class QueryResponse(BaseModel):
    question:           str
    ticker:             str
    answer:             str
    sources_used:       list[str]
    relevance_score:    float
    confidence_score:   float  = 0.0
    groundedness_score: float  = 0.0
    latency_ms:         float

class IngestRequest(BaseModel):
    ticker:   str            = Field(..., min_length=1, max_length=10)
    text:     str            = Field(..., min_length=10)
    metadata: dict           = Field(default_factory=dict)

class IngestResponse(BaseModel):
    ticker:     str
    chunks_added: int
    total_chunks: int


# ── Rate limiting (simple in-memory token bucket per IP) ─────────────────

_rate_buckets: dict[str, tuple[float, int]] = {}   # ip → (last_refill, tokens)
_RATE_LIMIT_RPS  = 10
_RATE_LIMIT_BURST = 20

def _check_rate_limit(ip: str) -> bool:
    now = time.time()
    last, tokens = _rate_buckets.get(ip, (now, _RATE_LIMIT_BURST))
    elapsed = now - last
    tokens  = min(_RATE_LIMIT_BURST, tokens + elapsed * _RATE_LIMIT_RPS)
    if tokens < 1:
        _rate_buckets[ip] = (now, tokens)
        return False
    _rate_buckets[ip] = (now, tokens - 1)
    return True

@app.middleware("http")
async def _rate_limit_middleware(request: Request, call_next):
    if request.method in ("POST",):
        ip = request.client.host if request.client else "unknown"
        if not _check_rate_limit(ip):
            return JSONResponse({"detail": "Rate limit exceeded — max 10 req/s"}, status_code=429)
    return await call_next(request)


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Liveness + readiness probe."""
    checks: dict[str, str] = {}

    # LLM
    try:
        provider = validate_llm_config()
        checks["llm"] = f"ok ({provider})"
    except RuntimeError as e:
        checks["llm"] = f"error: {e}"

    # Vector store
    try:
        from ..retrieval.vector_store import _settings_initialised
        checks["embedding_model"] = "loaded" if _settings_initialised else "not_loaded_yet"
    except Exception as e:
        checks["embedding_model"] = f"error: {e}"

    # LangChain
    try:
        from ..retrieval.rag_pipeline import _HAS_LANGCHAIN
        checks["langchain"] = "ok" if _HAS_LANGCHAIN else "not_installed"
    except Exception:
        checks["langchain"] = "not_installed"

    status = "ok" if all("error" not in v and "not_installed" not in v
                         for v in checks.values()) else "degraded"
    code   = 200 if status == "ok" else 503
    return JSONResponse({"status": status, "checks": checks}, status_code=code)


@app.post("/query", response_model=QueryResponse)
async def rag_query(req: QueryRequest) -> QueryResponse:
    """
    Run the equity RAG pipeline for one analyst question.
    Returns the answer, sources used, relevance score, and latency.
    """
    t0 = time.perf_counter()
    try:
        from ..retrieval.rag_pipeline import run as _run
        result = _run(
            question     = req.question,
            company_name = req.company_name or req.ticker,
            ticker       = req.ticker.upper(),
            session_id   = req.session_id,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception(f"[api] /query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    latency_ms = (time.perf_counter() - t0) * 1000
    answer = result.get("final_response") or result.get("response", "")
    if not answer:
        raise HTTPException(status_code=500, detail="Pipeline returned empty response")

    return QueryResponse(
        question           = req.question,
        ticker             = req.ticker.upper(),
        answer             = answer,
        sources_used       = result.get("sources_used", []),
        relevance_score    = result.get("relevance_score", 0.0),
        confidence_score   = result.get("confidence_score", 0.0),
        groundedness_score = result.get("groundedness_score", 0.0),
        latency_ms         = round(latency_ms, 1),
    )


@app.post("/stream")
async def rag_stream(req: StreamQueryRequest) -> StreamingResponse:
    """
    Stream the RAG response token-by-token using Server-Sent Events (SSE).

    Each SSE event carries one token:
        data: <token>\\n\\n

    The final event signals completion:
        data: [DONE]\\n\\n

    Clients should concatenate tokens until they receive [DONE].
    """
    try:
        from ..retrieval.rag_pipeline import stream_run as _stream_run
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"RAG pipeline unavailable: {e}")

    async def _event_generator() -> AsyncGenerator[str, None]:
        try:
            async for token in _stream_run(
                question     = req.question,
                company_name = req.company_name or req.ticker,
                ticker       = req.ticker.upper(),
                session_id   = req.session_id,
            ):
                # Escape newlines in token so SSE framing stays intact
                safe_token = token.replace("\n", "\\n")
                yield f"data: {safe_token}\n\n"
        except Exception as exc:
            logger.exception(f"[api] /stream error: {exc}")
            yield f"data: [ERROR] {exc}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx proxy buffering
        },
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(req: IngestRequest) -> IngestResponse:
    """Ingest a document into the per-ticker knowledge base."""
    try:
        from ..retrieval.vector_store import ingest_document as _ingest, collection_size
        chunks = _ingest(req.text, req.metadata, ticker=req.ticker.upper())
        total  = collection_size(req.ticker.upper())
    except Exception as e:
        logger.exception(f"[api] /ingest failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingest error: {e}")

    return IngestResponse(
        ticker       = req.ticker.upper(),
        chunks_added = chunks,
        total_chunks = total,
    )


@app.get("/collection/{ticker}")
async def collection_info(ticker: str) -> dict:
    """Return chunk count for a ticker's knowledge base."""
    try:
        from ..retrieval.vector_store import collection_size
        return {"ticker": ticker.upper(), "chunks": collection_size(ticker.upper())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collection/{ticker}")
async def clear_collection(ticker: str) -> dict:
    """Drop a ticker's knowledge base (irreversible)."""
    try:
        from ..retrieval.vector_store import clear_company
        clear_company(ticker.upper())
        return {"ticker": ticker.upper(), "status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
