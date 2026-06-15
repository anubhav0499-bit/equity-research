# Equity Research Platform — Retrieval Subsystem Reference

The `retrieval/` package provides the multi-agent RAG pipeline embedded inside
the equity research platform. It is called automatically during each research run
(filings and transcripts are indexed in Phases A2 and B) and is accessible to
every agent via `self.rag_query(question, state)`.

For manual ingestion of additional documents:
```bash
python -m equity_research.retrieval.ingest --ticker AAPL --file ./report.pdf
python -m equity_research.retrieval.ingest --ticker AAPL --url https://ir.apple.com/...
python -m equity_research.retrieval.ingest --ticker AAPL --status
```

---

## Overview

The retrieval subsystem is a multi-agent RAG pipeline built on three
complementary frameworks integrated into the 17-agent research workflow:

| Layer | Library | Role |
|---|---|---|
| Orchestration | **LangGraph** | State machine, conditional routing, retry loops |
| Retrieval | **LlamaIndex** | Optimised chunking, embedding, and semantic retrieval |
| Tools & LLMs | **LangChain** | LLM abstraction, tool wrappers, prompt templates |
| Embeddings | HuggingFace `bge-large-en-v1.5` | Local embeddings — no API key required |
| Vector store | ChromaDB | Persistent on-disk, one collection per company ticker |

---

## Architecture

```
User Query
    │
    ▼
[1] Query Rewriter        — Optimises the raw query for retrieval
    │
    ▼
[2] Detail Checker        — Decides: answer from parametric knowledge or retrieve?
    ├── No retrieval needed ──────────────────────────────────────────────┐
    │                                                                     │
    ▼                                                                     │
[3] Source Selector       — Picks the best source(s)                     │
    │                                                                     │
    ▼                                                                     │
[4] Retriever             — Fetches context from the selected source(s)  │
    │                                                                     │
    └──────────────────────────────────────────────────────────────────── ▼
                                                                   [5] Response Generator
                                                                          │
                                                                          ▼
                                                                   [6] Relevance Checker
                                                                     ├── Score ≥ threshold → Final Response
                                                                     └── Score < threshold → back to [1] (max 5 loops)
```

The graph is compiled once at import time (`graph.py`) and shared across all calls.
LangGraph merges partial state dicts from each node — nodes only return the fields
they modify, everything else is inherited from the prior state.

---

## Node Reference

### Node 1 — Query Rewriter (`query_rewriter`)

**Purpose:** Transform the raw user query into the most effective retrieval query.

**Inputs from state:**
- `original_query` — the user's verbatim input
- `rewritten_query` — the previous rewrite (on loop iterations > 1)
- `relevance_feedback` — the relevance checker's critique from the previous iteration

**Outputs to state:**
- `rewritten_query` — the optimised retrieval query
- `iteration` — incremented by 1 each pass

**Behaviour:** Expands abbreviations, adds synonyms, strips filler words, and on
retry iterations incorporates `relevance_feedback` to address what was missing.

**Prompt key:** `_REWRITE_PROMPT`

---

### Node 2 — Detail Checker (`detail_checker`)

**Purpose:** Decide whether the LLM can answer reliably from parametric knowledge
alone, or whether external retrieval is needed.

**Inputs from state:**
- `rewritten_query`

**Outputs to state:**
- `needs_retrieval: bool`
- `retrieval_reason: str`

**When retrieval is NOT needed:**
- General explanations and definitions
- Widely-known historical facts
- Reasoning, summarisation, or creative tasks

**When retrieval IS needed:**
- Real-time or recent information (prices, news, events)
- Specific documents the user has ingested
- Precise facts the LLM may hallucinate
- Domain-specific data (medical, legal, financial)

**Routing:** If `needs_retrieval = False` → jumps directly to Node 5. Otherwise → Node 3.

**Prompt key:** `_DETAIL_PROMPT`

---

### Node 3 — Source Selector (`source_selector`)

**Purpose:** Choose which retrieval source(s) will best satisfy the query.

**Inputs from state:**
- `rewritten_query`
- `retrieval_reason`
- live `collection_size()` from the vector store

**Outputs to state:**
- `selected_source: str` — one of `"vector_db"`, `"internet"`, `"tools_apis"`, `"combined"`
- `source_rationale: str`

**Source logic:**

| Value | Description | Use when |
|---|---|---|
| `vector_db` | ChromaDB local knowledge base | Query is about ingested documents / PDFs |
| `internet` | Tavily (preferred) or DuckDuckGo | Time-sensitive, current events, live data |
| `tools_apis` | Wikipedia + calculator + URL fetcher | Encyclopaedic facts, arithmetic, specific URLs |
| `combined` | All three sources in sequence | Complex queries needing background + real-time data |

**Prompt key:** `_SOURCE_PROMPT`

---

### Node 4 — Retriever (`retriever`)

**Purpose:** Execute the retrieval strategy and collect raw context chunks.

**Inputs from state:**
- `rewritten_query`
- `selected_source`

**Outputs to state:**
- `retrieved_context: list[str]` — raw text chunks
- `retrieval_metadata: list[dict]` — per-chunk provenance (`{"source": "vector_db"}` etc.)
- `sources_used: list[str]` — de-duplicated list of source names

**Internal dispatch:**

```
"vector_db"   → vs.retrieve(query, top_k=TOP_K)
"internet"    → T.web_search.run(query)
"tools_apis"  → T.wikipedia_lookup.run(query)
"combined"    → all three in sequence
```

The retriever does not call the LLM — it is a pure I/O node.

---

### Node 5 — Response Generator (`response_generator`)

**Purpose:** Synthesise a candidate answer from the query and retrieved context.

**Inputs from state:**
- `rewritten_query`
- `retrieved_context` — joined with `"\n\n---\n\n"` separators
- `sources_used`

**Outputs to state:**
- `response: str` — the candidate answer
- `messages` — appended with an `AIMessage`

**Context handling:**
Context is capped at 12,000 characters. If truncated, a `[... content truncated for length ...]`
marker is appended and a warning is logged so the model knows its context is partial.

**Prompt key:** `_RESPONSE_PROMPT`

---

### Node 6 — Relevance Checker (`relevance_checker`)

**Purpose:** Self-assess whether the candidate answer adequately addresses the query.

**Inputs from state:**
- `original_query`
- `rewritten_query`
- `response`

**Outputs to state:**
- `is_relevant: bool`
- `relevance_score: float` — 0.0 to 1.0
- `relevance_feedback: str` — critique for the rewriter on failure
- `final_response: str` — set only when `is_relevant = True`

**Scoring rubric:**

| Score | Meaning |
|---|---|
| 1.0 | Perfectly answers the question, well-structured, sources cited |
| 0.8 | Good answer, minor gaps |
| 0.6 | Partially answers, misses key aspects |
| 0.4 | Vague, off-topic, or factually questionable |
| < 0.4 | Wrong, hallucinated, or completely irrelevant |

**Routing:**
- `is_relevant = True` → `END` (publishes `final_response`)
- `is_relevant = False` and `iteration < MAX_ITERATIONS` → loops back to Node 1
- `is_relevant = False` and `iteration >= MAX_ITERATIONS` → `END` with best available `response`

**Prompt key:** `_RELEVANCE_PROMPT`

---

## State Schema (`RAGState`)

Defined in `state.py` as a `TypedDict` with `total=False` (all fields optional).
LangGraph merges partial dicts — a node returning `{"rewritten_query": "x"}` does
not overwrite any other field.

```python
class RAGState(TypedDict, total=False):
    # Input
    original_query: str           # user's raw query, never modified

    # Node 1 outputs
    rewritten_query: str          # optimised query
    iteration: int                # current loop count (1-based)

    # Node 2 outputs
    needs_retrieval: bool
    retrieval_reason: str

    # Node 3 outputs
    selected_source: str          # "vector_db" | "internet" | "tools_apis" | "combined"
    source_rationale: str

    # Node 4 outputs
    retrieved_context: list[str]
    retrieval_metadata: list[dict]
    sources_used: list[str]

    # Node 5 outputs
    response: str                 # candidate answer

    # Node 6 outputs
    is_relevant: bool
    relevance_score: float
    relevance_feedback: str

    # Conversation log (merged via add_messages, never overwritten)
    messages: Annotated[list[BaseMessage], add_messages]

    # Final output
    final_response: str           # set when relevance check passes
```

The `messages` field uses LangGraph's `add_messages` reducer — new messages are
appended rather than replacing the existing list.

---

## LLM Configuration (`config.py`)

### Provider Auto-Detection

`get_llm()` checks environment variables in this order and returns the first match:

1. `OPENAI_API_KEY` → `ChatOpenAI`
2. `ANTHROPIC_API_KEY` → `ChatAnthropic`
3. `GOOGLE_API_KEY` → `ChatGoogleGenerativeAI`
4. `GROQ_API_KEY` → `ChatGroq`
5. `AZURE_OPENAI_API_KEY` → `AzureChatOpenAI`
6. `OLLAMA_BASE_URL` or `OLLAMA_MODEL` → `ChatOllama`
7. No key found → `ChatOllama` (local Ollama, last resort)

### Supported Providers

| `LLM_PROVIDER` | Key env var | Model env var | Default model |
|---|---|---|---|
| `openai` | `OPENAI_API_KEY` | `OPENAI_MODEL` | `gpt-4o-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | `ANTHROPIC_MODEL` | `claude-sonnet-4-6` |
| `google` | `GOOGLE_API_KEY` | `GOOGLE_MODEL` | `gemini-1.5-pro` |
| `groq` | `GROQ_API_KEY` | `GROQ_MODEL` | `llama-3.1-70b-versatile` |
| `azure` | `AZURE_OPENAI_API_KEY` | `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` |
| `ollama` | _(none)_ | `OLLAMA_MODEL` | `llama3.1` |

### OpenAI-Compatible Endpoints

Any vLLM, LM Studio, or custom endpoint works with the `openai` provider:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=not-needed
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_MODEL=your-model-name
```

### Caching

`get_llm()` is decorated with `@lru_cache(maxsize=None)`. Each unique `(temperature,)`
argument is cached independently — the same LLM client object is reused across all
nodes in a process lifetime.

---

## Vector Store (`vector_store.py`)

### Storage

- **Backend:** ChromaDB `PersistentClient` — data survives process restarts
- **Default path:** `./chroma_db` (overridden by `CHROMA_PERSIST_DIR`)
- **Collection name:** `agentic_rag`
- **Embeddings:** HuggingFace `bge-small-en-v1.5` (local, no API key needed)
- **Chunk size:** 512 tokens, 64-token overlap (`SentenceSplitter`)

### Singleton Initialisation

The index, Chroma client, and collection are module-level globals, initialised
lazily on the first call to any public API. A `threading.Lock` (double-checked
locking) prevents race conditions under concurrent access.

### Public API

```python
# Add raw text strings
ingest_texts(texts: list[str], metadatas: list[dict] | None = None) -> int

# Add a file (PDF, TXT, MD, HTML)
ingest_file(path: str | Path) -> int

# Fetch and add a web page
ingest_url(url: str) -> int

# Semantic search
retrieve(query: str, top_k: int = TOP_K) -> list[str]

# Number of documents in the collection
collection_size() -> int
```

### System Documents

On startup, `ensure_system_docs_ingested()` checks for a `.system_docs_loaded`
sentinel file in `CHROMA_PERSIST_DIR`. If absent, all `.md` files in the `docs/`
directory (relative to the package root) are ingested automatically, then the
sentinel is written. This makes the system self-documenting — you can ask it
questions about itself.

---

## Tools Registry (`tools.py`)

All tools are LangChain `@tool`-decorated functions and can be used directly by
LLM agents or called from within retriever node via `.run()`.

| Tool | Function | Use case |
|---|---|---|
| `web_search` | Tavily → DuckDuckGo fallback | Live internet search |
| `wikipedia_lookup` | Wikipedia API | Encyclopaedic facts, definitions |
| `vector_db_search` | Wraps `vs.retrieve()` | Query local knowledge base |
| `fetch_url` | `requests` + BeautifulSoup | Scrape any URL |
| `calculator` | AST-safe eval | Arithmetic and unit conversions |

All tools are exported as `ALL_TOOLS = [web_search, wikipedia_lookup, vector_db_search, fetch_url, calculator]`.

### Web Search Fallback

`_get_search()` is a lazy singleton. If `TAVILY_API_KEY` is set, it returns
`TavilySearchResults(max_results=8)`. Otherwise it falls back to
`DuckDuckGoSearchResults(num_results=8)` — no key required.

### Wikipedia Singleton

`_get_wiki()` constructs `WikipediaQueryRun` once and caches it. Top-2 results,
4,000 character cap per article.

### Calculator

Uses `ast.parse` + a safe `_eval()` walker that only allows `ast.Constant`,
`ast.BinOp`, `ast.UnaryOp`, and `ast.Expression` nodes. All other node types
raise `ValueError`. Supports `+`, `-`, `*`, `/`, `//`, `%`, `**`.

---

## Graph Wiring (`graph.py`)

```
Entry: query_rewriter
  ↓  (edge)
detail_checker
  ↓  (conditional)
  ├── needs_retrieval=True  → source_selector → retriever → response_generator
  └── needs_retrieval=False ──────────────────────────────→ response_generator
                                                               ↓  (edge)
                                                         relevance_checker
                                                               ↓  (conditional)
                                                  ├── is_relevant=True / max_iter → END
                                                  └── is_relevant=False → query_rewriter
```

The compiled graph is stored in the module-level `_compiled` variable and built
once on the first call to `get_graph()`.

### `run(query: str) -> dict`

Calls `app.invoke()`, returns the complete final state dict. If `final_response`
is absent (max iterations exhausted without acceptance), falls back to `response`.

### `stream(query: str) -> Iterator[tuple[str, dict]]`

Calls `app.stream()`, yields `(node_name, partial_state_update)` tuples. Use
`--stream` CLI flag or `run_streaming()` in `main.py` to see live pipeline trace.

---

## Configuration Reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | _(auto-detect)_ | Force a specific provider: `openai`, `anthropic`, `google`, `groq`, `azure`, `ollama` |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENAI_API_KEY` | — | OpenAI (or compatible) API key |
| `OPENAI_BASE_URL` | _(OpenAI default)_ | Custom base URL for OpenAI-compatible endpoints |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name for OpenAI provider |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Model name for Anthropic provider |
| `GOOGLE_API_KEY` | — | Google Generative AI key |
| `GOOGLE_MODEL` | `gemini-1.5-pro` | Model for Google provider |
| `GROQ_API_KEY` | — | Groq API key |
| `GROQ_MODEL` | `llama-3.1-70b-versatile` | Model for Groq provider |
| `AZURE_OPENAI_API_KEY` | — | Azure OpenAI key |
| `AZURE_OPENAI_ENDPOINT` | — | Azure endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` | Azure deployment name |
| `AZURE_OPENAI_API_VERSION` | `2024-08-01-preview` | Azure API version |
| `OLLAMA_MODEL` | `llama3.1` | Model name for Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `TAVILY_API_KEY` | — | Tavily search key (omit to use DuckDuckGo) |
| `MAX_ITERATIONS` | `5` | Maximum query-rewrite loops before forcing an answer |
| `TOP_K_RETRIEVAL` | `5` | Number of vector store chunks returned per query |
| `RELEVANCE_THRESHOLD` | `0.7` | Minimum score for the relevance checker to accept an answer |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Directory for the ChromaDB persistent store |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |

---

## CLI Reference

### `main.py` — Interactive Chat

```bash
# Interactive mode
python main.py

# One-shot query
python main.py "What is LangGraph?"

# Streaming pipeline trace
python main.py --stream "Explain how transformers work"
```

**In-chat commands:**

| Command | Effect |
|---|---|
| `/ingest <path>` | Index a local file (PDF, TXT, MD) or a URL |
| `/clear` | Delete all documents from the vector store |
| `/quit` | Exit |

### `ingest.py` — Document Ingestion

```bash
python ingest.py --file ./report.pdf
python ingest.py --url https://arxiv.org/abs/1706.03762
python ingest.py --text "Some raw text to index"
python ingest.py --dir ./documents/          # recursive, .pdf/.txt/.md/.html
python ingest.py --file ./data.pdf --metadata '{"source": "arxiv", "year": 2024}'
```

---

## Data Flow Example

```
User: "What is the current Fed funds rate?"

→ query_rewriter:
    rewritten_query = "Federal Reserve federal funds rate current 2024"

→ detail_checker:
    needs_retrieval = True   (real-time data)
    retrieval_reason = "Fed rate is time-sensitive, requires live data"

→ source_selector:
    selected_source = "internet"
    rationale = "Current central bank rates require live web search"

→ retriever:
    calls T.web_search.run("Federal Reserve federal funds rate current 2024")
    retrieved_context = ["As of [date], the Fed funds rate is ..."]
    sources_used = ["web_search"]

→ response_generator:
    synthesises answer from context + query

→ relevance_checker:
    score = 0.87 → is_relevant = True
    final_response = <answer>
```

---

## Module Dependency Map

```
main.py
  └── graph.py
        ├── state.py         (RAGState TypedDict)
        └── agents.py
              ├── config.py  (get_llm, constants)
              ├── state.py
              ├── vector_store.py
              └── tools.py
                    ├── config.py  (get_search_tool)
                    └── vector_store.py

ingest.py
  └── vector_store.py
        └── config.py
```

---

## Extending the System

### Adding a New Tool

1. Add a `@tool`-decorated function to `tools.py`
2. Append it to `ALL_TOOLS`
3. Optionally import and call it in the `retriever` node (`agents.py`) for
   pipeline-native access

### Adding a New LLM Provider

1. Add a detection branch in `_detect_provider()` in `config.py`
2. Add a construction branch in `get_llm()` returning a `BaseChatModel` subclass
3. Document the required env vars in `.env.example`

### Adding a New Source

1. Add a new literal to `selected_source` type hints in `state.py`
2. Add the source description to `_SOURCE_PROMPT` in `agents.py`
3. Add a dispatch branch in the `retriever` node
4. Update `"combined"` to include the new source

### Changing Chunk Strategy

Edit `_init_llama_settings()` and the `SentenceSplitter` constructors in
`vector_store.py`. Changing `chunk_size` requires re-ingesting all documents
(old embeddings were computed with the previous chunk boundaries).
