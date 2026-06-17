# Equity Research Platform — Retrieval Subsystem Reference

The `retrieval/` package provides the full RAG pipeline embedded inside the equity
research platform. It is called automatically during each research run (filings and
transcripts are indexed in Phases A2 and B) and is accessible to every agent via
`self.rag_query(question, state)`. It can also be called standalone via the HTTP API
or directly from Python.

For manual ingestion of additional documents:
```bash
python -m equity_research.retrieval.ingest --ticker AAPL --file ./report.pdf
python -m equity_research.retrieval.ingest --ticker AAPL --url https://ir.apple.com/...
python -m equity_research.retrieval.ingest --ticker AAPL --status
```

---

## Overview

The retrieval subsystem is a 9-node LangGraph pipeline integrated into the 20-agent
research workflow. Key design decisions vs. a naive RAG setup:

| Layer | Library / Model | Role |
|---|---|---|
| Orchestration | **LangGraph** | 9-node state machine, conditional routing, retry loops |
| Embeddings | `BAAI/bge-small-en-v1.5` (~130 MB local) | Dense vector encoding; L2-normalised for cosine sim |
| Vector store | **FAISS** `IndexFlatIP` | Per-ticker persistent child + parent dual indices |
| Chunking | **SmartChunker** | Recursive / contextual / semantic — auto-detected per doc |
| Hybrid retrieval | **BM25 + RRF** | Keyword scoring merged with dense rank via Reciprocal Rank Fusion |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision re-scoring of the candidate pool |
| Query enhancement | **HyDE** | Hypothetical doc embedding blended 50/50 with query vector |
| Compression | **ContextCompressor** | Keyword filter → LLM extraction → Jaccard dedup |
| Memory | **ConversationStore** | Session-keyed sliding window with LLM rolling summary |
| Guardrails | **GuardrailsChecker** | Rule-based + LLM faithfulness + composite confidence score |
| Evaluation | **RAGASEvaluator** | Context relevance, faithfulness, answer relevance (geometric mean) |
| Tools | LangChain `@tool` wrappers | SEC EDGAR, web search, Wikipedia, yfinance, calculator |
| Streaming | FastAPI `StreamingResponse` | SSE token-by-token delivery via `stream_run()` |

---

## Architecture

```
User Query (+ session_id for memory)
    │
    ▼
[1] query_rewriter        — Optimise query + inject conversation history
    │                       + generate HyDE embedding on iteration 1
    ▼
[2] query_decomposer      — Detect multi-hop; split compound questions
    │
    ▼
[3] detail_checker        — Router: retrieval needed? or parametric LLM?
    ├── No  ────────────────────────────────────────────────────────────┐
    │                                                                   │
    ▼ Yes                                                               │
[4] source_selector       — Agentic: plan retrieval strategy           │
    │                       produces `retrieval_plan` (ordered steps)  │
    ▼                                                                   │
[5] retriever             — Parallel multi-source fetch                │
    │                       vector_db / internet / tools_apis          │
    ▼                                                                   │
[6] context_compressor    — LLM extraction + Jaccard dedup             │
    │                                                                   │
    └─────────────────────────────────────────────── [7] response_generator
                                                              │
                                                              ▼
                                                     [8] relevance_checker
                                                         + GuardrailsChecker
                                                         (groundedness + confidence)
                                                              │
                                                    ┌─────────┴──────────┐
                                               score ≥ threshold     score < threshold
                                                    │                    │  (max 5 loops)
                                                    ▼                    ▼
                                              save to memory      query_rewriter [1]
                                                    │
                                                    ▼
                                               final_response
```

---

## Module Reference

### `retrieval/chunking.py` — SmartChunker

Three splitting strategies, selected manually or via auto-detect:

| Strategy | Trigger heuristic | Behaviour |
|---|---|---|
| `contextual` | Header density > 2 per 50 lines | Splits on 10-K section headers (`ITEM`, `PART`, `MD&A`, `Risk Factors`); stores `section_header` in chunk metadata |
| `recursive` | High number density or default | Hierarchical separator: `\n\n` → `\n` → `. ` → ` ` |
| `semantic` | Low number density + long avg line length | Embedding-similarity boundary detection; falls back to `recursive` if embed_fn unavailable |

**Auto-detect (`auto_detect_strategy`):**
1. Header density > 2 per 50 lines → `contextual`
2. Avg line length > 120 chars → `semantic`
3. Otherwise → `recursive`

**Multi-vector ingest:** parent chunks use `SmartChunker` in `auto` mode; child chunks always use `recursive` (precision over strategy).

```python
from equity_research.retrieval.chunking import SmartChunker

chunker = SmartChunker(mode="auto", chunk_size=512)
texts, metadatas = chunker.split(filing_text, base_metadata={"doc_type": "10-K"})
```

---

### `retrieval/vector_store.py` — FAISS Multi-Vector Store

**Storage:** `data/faiss_index/<TICKER>/` — four files per ticker:

| File | Contents |
|---|---|
| `child_index.faiss` | FAISS `IndexFlatIP` over child (256-token) embeddings |
| `parent_index.faiss` | FAISS `IndexFlatIP` over parent (1024-token) embeddings |
| `child_docs.json` | Child chunk texts + `parent_id` back-references |
| `parent_docs.json` | Parent chunk texts + original metadata |

**Embedding:** `BAAI/bge-small-en-v1.5` (384-dim), L2-normalised before indexing so inner-product = cosine similarity. Fallback: `all-MiniLM-L6-v2`.

**Retrieval pipeline (per `query()` call):**

```
1. Dense retrieval       — FAISS IndexFlatIP over child vectors
                           candidates = max(top_k × 4, 20)
                           optional: pass hyde_vec for blended HyDE embedding
2. Parent lookup         — map child hits → parent chunk texts
3. Metadata filter       — post-retrieval filter on any metadata field
4. BM25 scoring          — rank-bm25 IDF-weighted term frequency per query term
5. Reciprocal Rank Fusion — merge dense rank + BM25 rank (RRF constant = 60)
6. Cross-encoder rerank  — ms-marco-MiniLM-L-6-v2; runs when candidates > top_k
7. Return top-k parent chunks
```

**Public API:**

```python
from equity_research.retrieval.vector_store import (
    ingest_document,   # (text, metadata, ticker) → int (chunks added)
    ingest_texts,      # (texts, metadatas, ticker) → int
    query,             # (question, ticker, top_k, metadata_filter, hyde_vec) → list[str]
    collection_size,   # (ticker) → int
    clear_company,     # (ticker) → None
)
```

---

### `retrieval/hyde.py` — HyDE

Hypothetical Document Embeddings improve retrieval on abstract/analytical queries where
the query text is semantically distant from the answer's vocabulary (e.g. "How did Apple
manage working capital?" vs. a document section titled "Cash Conversion Cycle").

```
query_text
    │
    ├── embed → query_vec
    │
    ├── LLM generates ~100-word hypothetical answer passage
    │       └── embed → hypo_vec
    │
    └── blended_vec = 0.5 × query_vec + 0.5 × hypo_vec  (L2-normalised)
                ↓
        FAISS search with blended_vec
```

Only runs on iteration 1. Returns `None` on any failure — callers use plain query embedding.

```python
from equity_research.retrieval.hyde import HyDE

hyde = HyDE(llm_fn=agent.llm_analyze, embed_fn=_embed)
blended = hyde.embed("What drove APEX's margin expansion?", company_name="APEX", ticker="APEX")
```

---

### `retrieval/compression.py` — ContextCompressor

Reduces LLM context window by 50–70% before generation. Four stages:

1. **Keyword filter** — drop chunks with < 10% query-term overlap
2. **LLM extraction** — extract relevant sentences per chunk; `[IRRELEVANT]` → dropped
3. **Jaccard dedup** — remove chunks with > 85% overlap on first 200 chars
4. **Char budget** — truncate total to `compression_max_chars` (default 8 000)

```python
from equity_research.retrieval.compression import ContextCompressor

compressor = ContextCompressor(llm_fn=agent.llm_analyze, keyword_threshold=0.10)
result = compressor.compress(query, chunks, max_output_chars=8000)
# result.compression_ratio, result.compressed_chunks
```

---

### `retrieval/memory.py` — ConversationStore

Session-keyed sliding window injected into `query_rewriter` so multi-turn
questions ("What about the Q2 figure?") resolve without re-retrieval.

**Compression:** when `total_chars > memory_max_chars`, older turns (keeping last 3
verbatim) are LLM-summarised into a rolling `session.summary` string.

```python
from equity_research.retrieval.memory import ConversationStore

store = ConversationStore.get()   # thread-safe singleton
store.add_exchange(session_id="s1", question="...", answer="...", sources=["10-K"])
context = store.get_context("s1", max_chars=3000)
store.evict_stale(ttl_seconds=3600)
```

---

### `retrieval/guardrails.py` — GuardrailsChecker

Three-layer faithfulness check after every generation:

| Layer | Method | Output field |
|---|---|---|
| Rule-based | Regex-extract numbers + dates from response; check presence in context | `hallucinated_numbers`, `rule_score` |
| LLM faithfulness | Ask LLM to identify unsupported factual claims | `unsupported_claims`, `llm_score` |
| Composite confidence | `0.40×groundedness + 0.35×relevance + 0.25×retrieval_quality` | `confidence_score` |

`grounded = True` when `groundedness_score ≥ RAG_CONFIG.groundedness_threshold` (default 0.70).
Both scores are returned in the `/query` API response.

```python
from equity_research.retrieval.guardrails import GuardrailsChecker

checker = GuardrailsChecker(llm_fn=agent.llm_analyze, groundedness_threshold=0.70)
result = checker.check(query, response, context_chunks,
                       relevance_score=0.88, retrieval_quality=0.9)
print(result.confidence_score, result.grounded)
```

---

### `retrieval/evaluation.py` — RAGASEvaluator

LLM-as-judge RAGAS metrics — no `ragas` package required. Disabled by default
(`RAG_CONFIG.ragas_enabled = False`); enable for offline evaluation runs.

| Metric | Judges | Aggregation |
|---|---|---|
| `context_relevance` | Are retrieved chunks relevant to the question? | Per-chunk score 0–3, normalised to 0–1 |
| `faithfulness` | Are response claims grounded in context? | `supported / total` claims |
| `answer_relevance` | Does the response address the question? | Holistic 0–1 |
| `ragas_score` | — | Geometric mean of all three |

```python
from equity_research.retrieval.evaluation import RAGASEvaluator

evaluator = RAGASEvaluator(llm_fn=agent.llm_analyze)
result = evaluator.evaluate(question, context_chunks, response)
print(result.ragas_score)   # 0.0–1.0

# Batch evaluation
results = evaluator.evaluate_batch(questions, context_batches, responses)
stats   = evaluator.summary_stats(results)
```

---

### `retrieval/rag_pipeline.py` — 9-Node LangGraph Pipeline

#### Node reference

| # | Node | Key behaviour |
|---|---|---|
| 1 | `query_rewriter` | Normalise + expand; inject `ConversationStore` session context; generate HyDE embedding on iteration 1 |
| 2 | `query_decomposer` | Detect multi-hop; split compound questions into `sub_queries` list |
| 3 | `detail_checker` | Router: needs retrieval? or parametric LLM answer? |
| 4 | `source_selector` | Agentic: outputs `selected_source` + `retrieval_plan` (ordered source list) |
| 5 | `retriever` | Parallel dispatch to `vector_db` / `internet` / `tools_apis` / `combined`; dedup; priority sort |
| 6 | `context_compressor` | `ContextCompressor.compress()` if `RAG_CONFIG.compression_enabled` |
| 7 | `response_generator` | Synthesise grounded answer from `compressed_context` (or raw chunks) |
| 8 | `relevance_checker` | Score answer vs. context; call `GuardrailsChecker`; save accepted exchange to `ConversationStore` |

Max retry loops: 5. On each retry, `query_rewriter` SIMPLIFIES the query (not expands) to break low-relevance cycles.

#### State schema (`EquityRAGState`)

```python
class EquityRAGState(TypedDict, total=False):
    # Identity
    company_name:        str
    ticker:              str
    session_id:          str          # links to ConversationStore

    # Pipeline
    original_query:      str
    rewritten_query:     str
    sub_queries:         list[str]
    is_multi_hop:        bool
    needs_retrieval:     bool
    retrieval_reason:    str
    selected_source:     str          # "vector_db" | "internet" | "tools_apis" | "combined"
    retrieval_plan:      list[str]    # agentic ordered source steps
    retrieved_context:   list[str]
    retrieval_metadata:  list[dict]
    compressed_context:  list[str]    # after context_compressor
    response:            str
    final_response:      str
    sources_used:        list[str]

    # Relevance + guardrails
    is_relevant:         bool
    relevance_score:     float
    relevance_feedback:  str
    groundedness_score:  float
    confidence_score:    float
    hallucinated_claims: list[str]
    iteration:           int

    # HyDE
    hyde_embedding:      Optional[list]   # float list of blended embedding

    # Conversation log
    messages:            Annotated[list[BaseMessage], add_messages]
```

#### Entry points

```python
from equity_research.retrieval.rag_pipeline import run, stream_run

# Synchronous — full 9-node pipeline
result = run(
    question     = "What was APEX FY2024 FCF?",
    company_name = "APEX Technologies",
    ticker       = "APEX",
    session_id   = "analyst-1",   # optional; enables conversation memory
)
# Keys: final_response, sources_used, relevance_score,
#       confidence_score, groundedness_score, hallucinated_claims

# Streaming — runs nodes 1–6 synchronously then streams generation token-by-token
async for token in stream_run(question, company_name, ticker, session_id):
    print(token, end="", flush=True)

# Convenience wrapper — returns just the answer string
from equity_research.retrieval.rag_pipeline import query
answer = query("What is APEX's gross margin?", ticker="APEX")
```

---

### `retrieval/tools.py` — LangChain Tool Wrappers

| Tool | Function | Notes |
|---|---|---|
| `web_search` | Tavily (preferred) or DuckDuckGo fallback | Requires `TAVILY_API_KEY`; falls back silently |
| `sec_edgar_search` | SEC EDGAR full-text search scoped to ticker | Uses `dateRange`/`startdt` params |
| `financial_snapshot` | yfinance price / market-cap / P/E | No API key; cached by yfinance |
| `wikipedia_lookup` | Wikipedia article summary | Gated by `_is_background_query` — only for background/overview queries |
| `calculator` | AST-safe arithmetic evaluator | Blocks `eval`, lambdas, exponents > 10 000 |

**Calculator security:** only `Num`, `BinOp` (+−×÷^), `UnaryOp`, and named `math.*`
functions are allowed. String operands and `__import__` raise `ValueError`.

**Wikipedia gating:** frozenset check on 30+ keywords (`history`, `founded`, `overview`,
`sector`, `headquarters`, etc.). Financial-figure queries are never routed to Wikipedia.

---

## RAGConfig — All Hyperparameters

All RAG settings live in one dataclass (`core/config.py`). Override at runtime:

```python
from equity_research.core.config import RAG_CONFIG
RAG_CONFIG.top_k = 8
RAG_CONFIG.hyde_enabled = False
RAG_CONFIG.ragas_enabled = True    # for offline evaluation
```

| Field | Default | Description |
|---|---|---|
| `model_name` | `BAAI/bge-small-en-v1.5` | Primary embedding model |
| `fallback_model` | `all-MiniLM-L6-v2` | Fallback if BGE unavailable |
| `chunking_mode` | `auto` | `auto` \| `recursive` \| `contextual` \| `semantic` |
| `child_chunk_size` | 256 | Tokens per child chunk |
| `child_chunk_overlap` | 32 | Token overlap for child chunks |
| `parent_chunk_size` | 1024 | Tokens per parent chunk |
| `parent_chunk_overlap` | 128 | Token overlap for parent chunks |
| `semantic_threshold` | 0.75 | Cosine similarity boundary for semantic splitting |
| `top_k` | 5 | Final chunks returned per query |
| `candidate_multiplier` | 4 | Candidate pool = `top_k × multiplier` |
| `min_candidates` | 20 | Minimum FAISS candidates regardless of top_k |
| `hyde_enabled` | `True` | HyDE on iteration 1 |
| `compression_enabled` | `True` | Context compression before generation |
| `compression_max_chars` | 8000 | Character budget after compression |
| `memory_max_turns` | 10 | Max raw turns in session window |
| `memory_max_chars` | 4000 | Character budget before LLM compression |
| `groundedness_threshold` | 0.70 | Min groundedness to pass guardrails |
| `confidence_threshold` | 0.60 | Min composite confidence for final answer |
| `ragas_enabled` | `False` | Enable RAGAS evaluation (3 extra LLM calls) |
| `vector_backend` | `faiss` | Currently only `faiss` |

---

## Data Flow Example

```
User: "What drove APEX's gross margin improvement in FY2024?"

→ [1] query_rewriter:
    session context = "" (first turn)
    rewritten_query = "APEX Technologies FY2024 gross margin expansion drivers"
    hyde_vec = embed(blend(query, hypothetical_passage))

→ [2] query_decomposer:
    is_multi_hop = False
    sub_queries   = ["APEX Technologies FY2024 gross margin expansion drivers"]

→ [3] detail_checker:
    needs_retrieval = True  (specific financial figure, recent data)

→ [4] source_selector:
    selected_source = "vector_db"
    retrieval_plan  = ["vector_db"]  (corpus has 320 chunks for APEX)

→ [5] retriever:
    FAISS child search with hyde_vec → 20 candidates
    parent lookup → 20 parent chunks
    BM25 scoring + RRF merge
    cross-encoder rerank → top 5 parent chunks
    retrieved_context = [chunk1, chunk2, chunk3, chunk4, chunk5]

→ [6] context_compressor:
    keyword filter: keeps 4/5 chunks
    LLM extraction: reduces each chunk to relevant sentences
    Jaccard dedup: no duplicates found
    compressed_context = 3 200 chars (from 7 800 original)

→ [7] response_generator:
    "APEX's gross margin improved 210 bps to 42.3% in FY2024,
     driven by: (1) product mix shift toward higher-margin software
     subscriptions, (2) materials cost deflation of 8% YoY, and
     (3) manufacturing scale benefits at the new Austin facility..."

→ [8] relevance_checker:
    relevance_score    = 0.91
    groundedness_score = 0.88  (all figures found in context)
    confidence_score   = 0.87  (composite)
    is_relevant        = True
    → save to ConversationStore session "analyst-1"
    → final_response published
```

---

## HTTP API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness + readiness — LLM, embedding model, LangChain |
| `POST` | `/query` | Full 9-node pipeline (sync); returns answer + scores |
| `POST` | `/stream` | Token-by-token SSE streaming |
| `POST` | `/ingest` | Ingest a document into a ticker's FAISS store |
| `GET` | `/collection/{ticker}` | Chunk count for a ticker |
| `DELETE` | `/collection/{ticker}` | Drop a ticker's FAISS store (irreversible) |

**POST /query request:**
```json
{
  "question":     "What was APEX FY2024 FCF?",
  "ticker":       "APEX",
  "company_name": "APEX Technologies",
  "session_id":   "analyst-1"
}
```

**POST /query response:**
```json
{
  "question":           "...",
  "ticker":             "APEX",
  "answer":             "APEX's FY2024 FCF was $2.1B...",
  "sources_used":       ["vector_db"],
  "relevance_score":    0.94,
  "confidence_score":   0.87,
  "groundedness_score": 0.91,
  "latency_ms":         312.4
}
```

**POST /stream** — `text/event-stream`, one token per event:
```
data: APEX\n\n
data: 's\n\n
data:  FY2024\n\n
...
data: [DONE]\n\n
```

---

## Module Dependency Map

```
rag_pipeline.py
  ├── vector_store.py      (FAISS, BM25, cross-encoder)
  │     └── chunking.py   (SmartChunker for ingest)
  ├── hyde.py              (HyDE embedding blend)
  ├── compression.py       (ContextCompressor)
  ├── memory.py            (ConversationStore)
  ├── guardrails.py        (GuardrailsChecker)
  ├── tools.py             (LangChain tool wrappers)
  └── core/config.py       (RAGConfig, LLM_CONFIG, validate_llm_config)

evaluation.py              (standalone — does not import pipeline)
  └── core/config.py

api/server.py
  ├── rag_pipeline.py      (run, stream_run)
  ├── vector_store.py      (ingest_document, collection_size, clear_company)
  └── core/config.py       (validate_llm_config, EMBEDDING_CONFIG)
```

---

## Extending the Retrieval System

### Adding a new tool

1. Add a `@tool`-decorated function in `retrieval/tools.py`
2. Wire it into `retriever` node in `rag_pipeline.py` under the appropriate source type
3. Add a security test in `tests/rag_backtest.py::run_security_tests()`

### Changing the embedding model

```python
from equity_research.core.config import RAG_CONFIG
RAG_CONFIG.model_name = "BAAI/bge-base-en-v1.5"   # 768-dim, better quality
```

Then clear existing indices — different models have incompatible embedding dimensions:

```python
from equity_research.retrieval.vector_store import clear_company
clear_company("TICKER")   # deletes data/faiss_index/<TICKER>/
```

### Adding a new chunking strategy

1. Add a function `my_split(text, chunk_size, chunk_overlap) -> list[str]` in `chunking.py`
2. Add a `ChunkingMode` literal
3. Add a branch in `SmartChunker.split()` and `auto_detect_strategy()`

### Disabling components for speed

```python
from equity_research.core.config import RAG_CONFIG
RAG_CONFIG.hyde_enabled        = False   # skip HyDE (saves 1 LLM call)
RAG_CONFIG.compression_enabled = False   # skip compression (faster, larger context)
RAG_CONFIG.ragas_enabled       = False   # skip RAGAS (default; saves 3 LLM calls)
```
