"""
Equity-Research RAG Pipeline — LangGraph multi-agent retrieval.

Architecture (7 nodes):

    Query
      ↓
  [1] query_rewriter       — Optimise query; on retry, SIMPLIFY rather than expand
      ↓
  [2] query_decomposer     — Detect multi-hop; split compound questions into sub-queries
      ↓
  [3] detail_checker       — Need retrieval or LLM parametric knowledge?
      ├── No  ────────────────────────────────────────────────────────────┐
      ↓ Yes                                                               │
  [4] source_selector      — vector_db / internet / tools_apis / combined │
      ↓                                                                   │
  [5] retriever            — Parallel fetch; deduplicate; priority sort  │
      └──────────────────────────────────────────── [6] response_generator
                                                          ↓
                                                   [7] relevance_checker
                                                       (context-grounded,
                                                        not self-grading)
                                                          ├── Yes → END
                                                          └── No  → loop (max 5)
"""

from __future__ import annotations
import concurrent.futures
import json
import re
import threading
from functools import lru_cache
from typing import Annotated, Optional
from typing_extensions import TypedDict

_HAS_LANGCHAIN = False
try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    _HAS_LANGCHAIN = True
except ImportError:
    class BaseMessage:  # type: ignore[no-redef]
        pass
    HumanMessage = None           # type: ignore[assignment]
    AIMessage    = None           # type: ignore[assignment]
    ChatPromptTemplate = None     # type: ignore[assignment]
    StateGraph   = None           # type: ignore[assignment]
    END          = "__end__"      # type: ignore[assignment]
    def add_messages(left, right):  # type: ignore[misc]
        return (left or []) + (right or [])
from loguru import logger


def _make_prompt(messages: list):
    """Build a ChatPromptTemplate, or return None when langchain_core is absent."""
    if not _HAS_LANGCHAIN:
        return None
    return ChatPromptTemplate.from_messages(messages)

try:
    from . import vector_store as vs
    _HAS_VS = True
except ImportError:
    vs = None   # type: ignore[assignment]
    _HAS_VS = False

try:
    from . import tools as T
    _HAS_TOOLS = True
except ImportError:
    T = None    # type: ignore[assignment]
    _HAS_TOOLS = False

from ..core.config import (
    LLM_CONFIG, OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY,
    GOOGLE_API_KEY, TOGETHER_API_KEY, OPENROUTER_API_KEY,
)

MAX_ITERATIONS      = 5
RELEVANCE_THRESHOLD = 0.70
TOP_K               = 5

# Wikipedia is only useful for background/definitional queries
_BACKGROUND_KEYWORDS = frozenset([
    "what is", "what are", "define", "explain", "background", "overview",
    "history", "sector", "industry", "how does", "introduction", "founded",
])


def _is_background_query(query: str) -> bool:
    """Return True only for queries asking for background or definitional context."""
    q_lower = query.lower()
    return any(kw in q_lower for kw in _BACKGROUND_KEYWORDS)


# ── LangChain LLM bridge ──────────────────────────────────────────────────

def _resolved_provider() -> str:
    """Resolve the effective LLM provider from config and available API keys."""
    backend = LLM_CONFIG.provider
    if backend != "auto":
        return backend
    if GROQ_API_KEY:         return "groq"
    if OPENAI_API_KEY:       return "openai"
    if ANTHROPIC_API_KEY:    return "anthropic"
    if TOGETHER_API_KEY:     return "together"
    if OPENROUTER_API_KEY:   return "openrouter"
    if GOOGLE_API_KEY:       return "gemini"
    return "ollama"


@lru_cache(maxsize=None)
def _get_llm(temperature: float = 0.0, provider: str = ""):
    """Return a cached LangChain LLM. Cache key is (temperature, provider)."""
    backend = provider or _resolved_provider()
    model   = LLM_CONFIG.primary_model

    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, api_key=OPENAI_API_KEY)
    if backend == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature,
                             api_key=ANTHROPIC_API_KEY, max_tokens=4096)
    if backend == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature, groq_api_key=GROQ_API_KEY)
    if backend in ("gemini", "google"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature,
                                      google_api_key=GOOGLE_API_KEY)
    if backend in ("together", "openrouter"):
        from langchain_openai import ChatOpenAI
        base_url = ("https://api.together.xyz/v1" if backend == "together"
                    else "https://openrouter.ai/api/v1")
        key = TOGETHER_API_KEY if backend == "together" else OPENROUTER_API_KEY
        return ChatOpenAI(model=model, temperature=temperature, base_url=base_url, api_key=key)
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=LLM_CONFIG.primary_model or "llama3.1",
        base_url=LLM_CONFIG.ollama_base_url,
        temperature=temperature,
    )


def _llm():
    return _get_llm(temperature=0.0, provider=_resolved_provider())


# ── State ─────────────────────────────────────────────────────────────────

class EquityRAGState(TypedDict, total=False):
    # Company context (set once at pipeline entry)
    company_name:     str
    ticker:           str

    # Pipeline fields
    original_query:   str
    rewritten_query:  str
    sub_queries:      list[str]    # atomic sub-queries from query_decomposer
    is_multi_hop:     bool
    needs_retrieval:  bool
    retrieval_reason: str
    selected_source:  str          # "vector_db" | "internet" | "tools_apis" | "combined"
    source_rationale: str
    retrieved_context:  list[str]
    retrieval_metadata: list[dict]
    response:         str
    is_relevant:      bool
    relevance_score:  float
    relevance_feedback: str
    iteration:        int
    messages:         Annotated[list[BaseMessage], add_messages]
    final_response:   str
    sources_used:     list[str]


# ── JSON helper ───────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    clean = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    start = clean.find("{")
    if start == -1:
        return {}
    depth, end = 0, -1
    for i, ch in enumerate(clean[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return {}
    try:
        return json.loads(clean[start:end + 1])
    except json.JSONDecodeError:
        return {}


# ── Node 1 — Query Rewriter ───────────────────────────────────────────────

_REWRITE_PROMPT = _make_prompt([
    ("system", """You are an expert query optimiser for an equity research RAG system.
Transform the analyst's question into the most effective retrieval query for searching
company filings, earnings transcripts, analyst reports, and financial databases.

Guidelines:
- Expand tickers to full company names (and vice versa)
- Include relevant financial terminology and synonyms
- Make the fiscal period explicit if implied (e.g. "last quarter" → "Q3 FY2024")
- On RETRY iterations: SIMPLIFY the query — fewer terms, more specific. Focus on the
  exact data gap identified in the feedback. Do not make the query broader.

Respond with ONLY valid JSON:
{{"rewritten_query": "<optimised query>", "rationale": "<brief explanation>"}}"""),
    ("human", """Company: {company_name} ({ticker})
Original query: {original_query}
Iteration: {iteration}
Previous query: {previous_query}
Relevance feedback: {feedback}"""),
])


def query_rewriter(state: EquityRAGState) -> dict:
    iteration = state.get("iteration", 0) + 1
    orig      = state.get("original_query", "")
    ticker    = state.get("ticker", "")
    company   = state.get("company_name", ticker)

    result = (_REWRITE_PROMPT | _llm()).invoke({
        "company_name":   company,
        "ticker":         ticker,
        "original_query": orig,
        "iteration":      iteration,
        "previous_query": state.get("rewritten_query", ""),
        "feedback":       state.get("relevance_feedback", ""),
    })
    parsed = _parse_json(result.content)
    rw     = parsed.get("rewritten_query") or orig

    logger.debug(f"[rag:{ticker}] rewriter iter={iteration} → {rw[:80]}")
    return {
        "rewritten_query": rw,
        "iteration":       iteration,
        "messages":        [HumanMessage(content=orig)] if iteration == 1 else [],
    }


# ── Node 2 — Query Decomposer ─────────────────────────────────────────────

_DECOMPOSE_PROMPT = _make_prompt([
    ("system", """You are a query analyst for an equity research RAG system.
Determine if the query requires information from multiple documents or time periods
(multi-hop), and if so, decompose it into simpler atomic sub-queries.

Multi-hop indicators:
- Comparing figures across two periods ("Did Q3 actuals beat Q3 guidance?")
- Cross-document synthesis ("How did executive pay change vs revenue growth?")
- Derived metrics requiring two separately-sourced inputs

Single-hop: simple factual lookup from one document ("What was APEX revenue in FY2023?")

For single-hop: return the original query as the only sub-query.
For multi-hop: decompose into 2-3 atomic sub-queries, each answerable from one source.
Each sub-query must be self-contained — include company name and fiscal period explicitly.

Respond ONLY with valid JSON:
{{"is_multi_hop": true/false, "sub_queries": ["<query1>", "<query2>"]}}"""),
    ("human", "Query: {query}\nCompany: {company_name} ({ticker})"),
])


def query_decomposer(state: EquityRAGState) -> dict:
    query   = state.get("rewritten_query", state.get("original_query", ""))
    ticker  = state.get("ticker", "")
    company = state.get("company_name", ticker)

    result = (_DECOMPOSE_PROMPT | _llm()).invoke({
        "query":        query,
        "company_name": company,
        "ticker":       ticker,
    })
    parsed       = _parse_json(result.content)
    is_multi_hop = bool(parsed.get("is_multi_hop", False))
    sub_queries  = parsed.get("sub_queries", [])
    if not sub_queries or not isinstance(sub_queries, list):
        sub_queries = [query]

    logger.debug(
        f"[rag:{ticker}] decomposer: multi_hop={is_multi_hop} "
        f"sub_queries={len(sub_queries)}"
    )
    return {"sub_queries": sub_queries, "is_multi_hop": is_multi_hop}


# ── Node 3 — Detail Checker ───────────────────────────────────────────────

_DETAIL_PROMPT = _make_prompt([
    ("system", """You are an equity research RAG orchestrator.
Decide whether the analyst's question needs external retrieval (filings, web, tools)
or can be answered from the LLM's parametric knowledge.

Retrieval IS needed for:
- Specific financial figures (revenue, EPS, margins) for a particular period
- Management guidance, risk factors, or contractual terms
- Recent news, regulatory filings, or events after the LLM's training cut-off
- Calculations requiring live or company-specific data

Retrieval NOT needed for:
- General accounting or valuation methodology questions
- Definitional or conceptual questions
- Requests to reason over data already provided

Respond with ONLY valid JSON:
{{"needs_retrieval": true/false, "reason": "<one sentence>"}}"""),
    ("human", "Query: {query}\nCompany: {company_name} ({ticker})"),
])


def detail_checker(state: EquityRAGState) -> dict:
    query  = state.get("rewritten_query", state.get("original_query", ""))
    result = (_DETAIL_PROMPT | _llm()).invoke({
        "query":        query,
        "company_name": state.get("company_name", ""),
        "ticker":       state.get("ticker", ""),
    })
    parsed = _parse_json(result.content)
    needs  = bool(parsed.get("needs_retrieval", True))
    logger.debug(f"[rag] detail_checker: needs_retrieval={needs}")
    return {"needs_retrieval": needs, "retrieval_reason": parsed.get("reason", "")}


# ── Node 4 — Source Selector ──────────────────────────────────────────────

_SOURCE_PROMPT = _make_prompt([
    ("system", """You are a retrieval strategy selector for an equity research platform.

Available sources:
  "vector_db"   — Indexed filings, transcripts, and reports for this specific company.
                  Best for: specific revenue figures, MD&A passages, risk factors,
                  earnings call quotes, and anything in the ingested document corpus.
  "internet"    — Live web search (Tavily/DuckDuckGo) for current events.
                  Best for: latest analyst ratings, news after filing dates, price reactions.
  "tools_apis"  — SEC EDGAR full-text search, financial snapshot (yfinance), Wikipedia.
                  Best for: cross-filing keyword search, live price/multiple data, definitions.
  "combined"    — All three. Use for complex questions needing both document depth and live data.

Local corpus size: {kb_size} indexed chunks.
Prefer "vector_db" when corpus ≥ 10 chunks and the query is about historical filings or figures.

Respond with ONLY valid JSON:
{{"selected_source": "<vector_db|internet|tools_apis|combined>",
  "rationale": "<one sentence>"}}"""),
    ("human", "Query: {query}\nCompany: {company_name} ({ticker})\nReason for retrieval: {reason}"),
])


def source_selector(state: EquityRAGState) -> dict:
    ticker = state.get("ticker", "UNKNOWN")
    query  = state.get("rewritten_query", state.get("original_query", ""))
    kb_sz  = vs.collection_size(ticker) if vs is not None else 0

    result = (_SOURCE_PROMPT | _llm()).invoke({
        "query":        query,
        "company_name": state.get("company_name", ""),
        "ticker":       ticker,
        "reason":       state.get("retrieval_reason", ""),
        "kb_size":      kb_sz,
    })
    parsed = _parse_json(result.content)
    source = parsed.get("selected_source", "combined")
    if source not in {"vector_db", "internet", "tools_apis", "combined"}:
        source = "combined"

    logger.debug(f"[rag:{ticker}] source={source} kb_size={kb_sz}")
    return {"selected_source": source, "source_rationale": parsed.get("rationale", "")}


# ── Node 5 — Retriever ────────────────────────────────────────────────────

# Context priority — lower index = inserted first = less likely to be dropped by 12K cap
_SOURCE_PRIORITY = {
    "vector_db":          0,
    "sec_edgar":          1,
    "financial_snapshot": 2,
    "web_search":         3,
    "wikipedia":          4,
}


def retriever(state: EquityRAGState) -> dict:
    ticker      = state.get("ticker", "UNKNOWN")
    query       = state.get("rewritten_query", state.get("original_query", ""))
    source      = state.get("selected_source", "combined")
    company     = state.get("company_name", ticker)
    sub_queries = state.get("sub_queries") or [query]
    primary_q   = sub_queries[0]

    chunk_meta_pairs: list[tuple[str, dict]] = []

    # ── Individual source fetchers ────────────────────────────────────────

    def _from_vector_db(q: str) -> list[tuple[str, dict]]:
        if vs is None:
            return []
        try:
            return [(txt, {"source": "vector_db"})
                    for txt in vs.query(q, ticker=ticker, top_k=TOP_K)]
        except Exception as e:
            logger.warning(f"[rag:{ticker}] vector_db failed: {e}")
            return []

    def _from_internet(q: str) -> list[tuple[str, dict]]:
        if T is None:
            return []
        try:
            r = T.web_search.run(f"{company} {ticker} {q}")
            if r and "failed" not in r.lower():
                return [(str(r), {"source": "web_search"})]
        except Exception as e:
            logger.warning(f"[rag:{ticker}] web_search failed: {e}")
        return []

    def _from_edgar(q: str) -> list[tuple[str, dict]]:
        if T is None:
            return []
        try:
            r = T.sec_edgar_search.run(f"{ticker} {q}")
            if r and "failed" not in r.lower():
                return [(r, {"source": "sec_edgar"})]
        except Exception as e:
            logger.warning(f"[rag:{ticker}] edgar failed: {e}")
        return []

    def _from_snapshot() -> list[tuple[str, dict]]:
        if T is None:
            return []
        try:
            if any(kw in query.lower()
                   for kw in ("price", "p/e", "multiple", "market cap", "valuation")):
                r = T.financial_snapshot.run(ticker)
                if r and "failed" not in r.lower():
                    return [(r, {"source": "financial_snapshot"})]
        except Exception as e:
            logger.warning(f"[rag:{ticker}] snapshot failed: {e}")
        return []

    def _from_wikipedia(q: str) -> list[tuple[str, dict]]:
        if T is None or not _is_background_query(q):
            return []
        try:
            r = T.wikipedia_lookup.run(f"{company} {q}")
            if r and "failed" not in r.lower():
                return [(r, {"source": "wikipedia"})]
        except Exception as e:
            logger.warning(f"[rag:{ticker}] wikipedia failed: {e}")
        return []

    # ── Dispatch by source ────────────────────────────────────────────────

    if source == "vector_db":
        # For multi-hop: retrieve for each sub-query to span multiple documents
        for sq in sub_queries:
            chunk_meta_pairs.extend(_from_vector_db(sq))

    elif source == "internet":
        chunk_meta_pairs.extend(_from_internet(primary_q))

    elif source == "tools_apis":
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(_from_edgar,     primary_q),
                pool.submit(_from_snapshot),
                pool.submit(_from_wikipedia, primary_q),
            ]
            for f in concurrent.futures.as_completed(futures):
                chunk_meta_pairs.extend(f.result())

    else:  # combined — run all sources in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            vdb_futures = [pool.submit(_from_vector_db, sq) for sq in sub_queries]
            net_f  = pool.submit(_from_internet, primary_q)
            edg_f  = pool.submit(_from_edgar,    primary_q)
            snap_f = pool.submit(_from_snapshot)
            wiki_f = pool.submit(_from_wikipedia, primary_q)

            # Collect vector_db results first (highest priority)
            for f in vdb_futures:
                chunk_meta_pairs.extend(f.result())
            for f in [edg_f, snap_f, net_f, wiki_f]:
                chunk_meta_pairs.extend(f.result())

    # ── Deduplicate by 200-char fingerprint ──────────────────────────────
    seen:   set[str] = set()
    deduped: list[tuple[str, dict]] = []
    for (text, meta) in chunk_meta_pairs:
        fp = text[:200].strip()
        if fp not in seen:
            seen.add(fp)
            deduped.append((text, meta))

    # Sort by source priority so highest-quality sources fill the 12K context window first
    deduped.sort(key=lambda x: _SOURCE_PRIORITY.get(x[1].get("source", ""), 99))

    chunk_texts = [t for t, _ in deduped]
    metadata    = [m for _, m in deduped]

    logger.debug(f"[rag:{ticker}] retriever: {len(chunk_texts)} unique chunks from {source}")
    return {
        "retrieved_context":  chunk_texts,
        "retrieval_metadata": metadata,
        "sources_used":       list({m["source"] for m in metadata}),
    }


# ── Node 6 — Response Generator ───────────────────────────────────────────

_RESPONSE_PROMPT = _make_prompt([
    ("system", """You are a senior equity research analyst.
Answer the analyst's question precisely and concisely using the retrieved context below.
- Quote specific figures (revenue, EPS, margins) with fiscal periods.
- Cite the source (filing type, date) when available in the context.
- If the context does not contain the answer, say so explicitly — do not invent figures.
- Use bullet points for lists of facts; prose for narrative answers.

Company: {company_name} ({ticker})

Retrieved Context:
{context}

Sources used: {sources}"""),
    ("human", "{query}"),
])


def response_generator(state: EquityRAGState) -> dict:
    ticker  = state.get("ticker", "UNKNOWN")
    query   = state.get("rewritten_query", state.get("original_query", ""))
    chunks  = state.get("retrieved_context", [])
    sources = state.get("sources_used", [])

    # Chunks are already priority-sorted by the retriever; tail drops lowest-priority first
    context = "\n\n---\n\n".join(chunks) if chunks else "No external context retrieved."
    if len(context) > 12000:
        logger.warning(f"[rag:{ticker}] context truncated: {len(context)} → 12000 chars")
        context = context[:12000] + "\n\n[... content truncated for length ...]"

    result = (_RESPONSE_PROMPT | _llm()).invoke({
        "query":        query,
        "company_name": state.get("company_name", ticker),
        "ticker":       ticker,
        "context":      context,
        "sources":      ", ".join(sources) if sources else "parametric knowledge",
    })
    return {
        "response": result.content,
        "messages": [AIMessage(content=result.content)],
    }


# ── Node 7 — Relevance Checker (context-grounded) ────────────────────────
# Previously: LLM graded its own response (self-evaluation bias).
# Now: LLM grades response against the retrieved context (faithfulness check).

_RELEVANCE_PROMPT = _make_prompt([
    ("system", """You are a quality-assurance judge for equity research.
Evaluate whether the generated answer is faithfully grounded in the retrieved context below.

CRITICAL: A response that sounds correct but uses figures NOT present in the retrieved
context should score LOW. You are testing context-faithfulness, not general correctness.

Score 0.0 to 1.0:
  1.0 — All cited figures explicitly appear in the context; answer directly addresses the query
  0.8 — Most figures are in context; minor gaps in citation specificity
  0.6 — Partially grounded; some figures not supported by context or wrong period
  0.4 — Mostly from LLM knowledge rather than the provided context
  < 0.4 — Contradicts context, or the context was ignored entirely

Acceptance threshold: {threshold}

Retrieved Context (first 3,000 chars for grounding check):
{context_preview}

Respond with ONLY valid JSON:
{{"is_relevant": true/false, "score": <0.0–1.0>,
  "feedback": "<what was missing or mis-grounded — be specific about which figures lacked context support>"}}"""),
    ("human", """Original query: {original_query}
Company: {company_name} ({ticker})
Generated response: {response}"""),
])


def relevance_checker(state: EquityRAGState) -> dict:
    ticker  = state.get("ticker", "UNKNOWN")
    chunks  = state.get("retrieved_context", [])

    # Provide a representative slice of context for faithfulness grading
    context_preview = "\n\n".join(chunks[:3])[:3000] if chunks else "No context retrieved."

    result = (_RELEVANCE_PROMPT | _llm()).invoke({
        "original_query":  state.get("original_query", ""),
        "company_name":    state.get("company_name", ticker),
        "ticker":          ticker,
        "response":        state.get("response", ""),
        "threshold":       RELEVANCE_THRESHOLD,
        "context_preview": context_preview,
    })
    parsed   = _parse_json(result.content)
    score    = float(parsed.get("score", 0.5))
    relevant = bool(parsed.get("is_relevant", score >= RELEVANCE_THRESHOLD))
    feedback = parsed.get("feedback", "")

    logger.debug(f"[rag:{ticker}] relevance score={score:.2f} accepted={relevant}")

    if relevant:
        return {
            "is_relevant":        True,
            "relevance_score":    score,
            "relevance_feedback": feedback,
            "final_response":     state.get("response", ""),
        }
    return {
        "is_relevant":        False,
        "relevance_score":    score,
        "relevance_feedback": feedback,
    }


# ── Routing ───────────────────────────────────────────────────────────────

def _route_after_detail(state: EquityRAGState) -> str:
    return "source_selector" if state.get("needs_retrieval", True) else "response_generator"


def _route_after_relevance(state: EquityRAGState) -> str:
    if state.get("is_relevant", False):
        return "END"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        logger.warning("[rag] max iterations reached — returning best answer")
        return "END"
    return "query_rewriter"


# ── Graph ─────────────────────────────────────────────────────────────────

_compiled     = None
_compile_lock = threading.Lock()


def _get_graph():
    global _compiled
    if _compiled is not None:
        return _compiled
    with _compile_lock:
        if _compiled is not None:
            return _compiled
        g = StateGraph(EquityRAGState)
        g.add_node("query_rewriter",     query_rewriter)
        g.add_node("query_decomposer",   query_decomposer)
        g.add_node("detail_checker",     detail_checker)
        g.add_node("source_selector",    source_selector)
        g.add_node("retriever",          retriever)
        g.add_node("response_generator", response_generator)
        g.add_node("relevance_checker",  relevance_checker)

        g.set_entry_point("query_rewriter")
        g.add_edge("query_rewriter",     "query_decomposer")
        g.add_edge("query_decomposer",   "detail_checker")
        g.add_edge("source_selector",    "retriever")
        g.add_edge("retriever",          "response_generator")
        g.add_edge("response_generator", "relevance_checker")

        g.add_conditional_edges(
            "detail_checker", _route_after_detail,
            {"source_selector": "source_selector",
             "response_generator": "response_generator"},
        )
        g.add_conditional_edges(
            "relevance_checker", _route_after_relevance,
            {"END": END, "query_rewriter": "query_rewriter"},
        )
        _compiled = g.compile()
    return _compiled


# ── Public API ────────────────────────────────────────────────────────────

def run(
    question:     str,
    company_name: str = "",
    ticker:       str = "UNKNOWN",
) -> dict:
    """
    Run the full equity RAG pipeline for one question about a company.
    Returns the final state dict with `final_response`, `sources_used`, `relevance_score`.
    """
    if not _HAS_LANGCHAIN:
        raise RuntimeError(
            "LangChain/LangGraph is not installed. Run:\n"
            "  pip install langchain-core langchain-openai langgraph\n"
            "and set at least one LLM API key in .env"
        )
    from ..core.config import validate_llm_config
    validate_llm_config()   # raises RuntimeError with instructions if no provider configured
    app         = _get_graph()
    final_state = app.invoke({
        "original_query": question,
        "company_name":   company_name,
        "ticker":         ticker.upper(),
        "iteration":      0,
        "messages":       [],
        "sub_queries":    [question],
        "is_multi_hop":   False,
    })
    if not final_state.get("final_response") and final_state.get("response"):
        final_state["final_response"] = final_state["response"]
    return final_state


def query(
    question:     str,
    company_name: str = "",
    ticker:       str = "UNKNOWN",
) -> str:
    """Convenience wrapper — returns just the answer string."""
    result = run(question, company_name=company_name, ticker=ticker)
    return result.get("final_response", result.get("response", ""))
