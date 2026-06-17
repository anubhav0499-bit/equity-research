"""
Equity-Research RAG Pipeline — enhanced LangGraph multi-agent retrieval.

Architecture (9 nodes):

    Query (+ session memory context)
      ↓
  [1] query_rewriter       — Optimise query + HyDE (hypothetical doc embedding)
      ↓
  [2] query_decomposer     — Detect multi-hop; split compound questions
      ↓
  [3] detail_checker       — Needs retrieval? or parametric LLM?
      ├── No  ─────────────────────────────────────────────────────────┐
      ↓ Yes                                                            │
  [4] source_selector      — Agentic: plan retrieval strategy          │
      ↓                                                                │
  [5] retriever            — Parallel multi-source fetch              │
      ↓                                                                │
  [6] context_compressor   — LLM extraction + redundancy removal      │
      └─────────────────────────────────────── [7] response_generator ┘
                                                        ↓
                                                [8] relevance_checker
                                                    + guardrails
                                                    (faithfulness / confidence)
                                                        ├── OK  → END
                                                        └── Bad → loop (max 5)

New capabilities vs previous version:
  - HyDE: hypothetical doc embedding blended with query embedding
  - Context compression: LLM extracts relevant passages before generation
  - Conversation memory: session history injected into query rewriter
  - Guardrails: groundedness check + confidence score on every response
  - Streaming: stream_run() async generator for SSE delivery
  - Agentic source selector: can plan multi-step retrieval
"""

from __future__ import annotations
import concurrent.futures
import json
import re
import threading
from functools import lru_cache
from typing import Annotated, AsyncGenerator, Optional
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
    LLM_CONFIG, RAG_CONFIG,
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY,
    GOOGLE_API_KEY, TOGETHER_API_KEY, OPENROUTER_API_KEY,
)

MAX_ITERATIONS      = 5
RELEVANCE_THRESHOLD = 0.70
TOP_K               = RAG_CONFIG.top_k

_BACKGROUND_KEYWORDS = frozenset([
    "what is", "what are", "define", "explain", "background", "overview",
    "history", "sector", "industry", "how does", "introduction", "founded",
])


def _is_background_query(query: str) -> bool:
    q_lower = query.lower()
    return any(kw in q_lower for kw in _BACKGROUND_KEYWORDS)


# ── LangChain LLM bridge ──────────────────────────────────────────────────────

def _resolved_provider() -> str:
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


def _llm_fn(system: str, user: str) -> str:
    """Simple (system, user) → str wrapper for modules that need it."""
    if not _HAS_LANGCHAIN:
        return ""
    prompt = _make_prompt([("system", system), ("human", "{user}")])
    result = (prompt | _llm()).invoke({"user": user})
    return getattr(result, "content", str(result))


# ── State ─────────────────────────────────────────────────────────────────────

class EquityRAGState(TypedDict, total=False):
    # Company context
    company_name:     str
    ticker:           str
    session_id:       str        # for conversation memory

    # Pipeline fields
    original_query:   str
    rewritten_query:  str
    sub_queries:      list[str]
    is_multi_hop:     bool
    needs_retrieval:  bool
    retrieval_reason: str
    selected_source:  str
    source_rationale: str
    retrieval_plan:   list[str]  # agentic: ordered retrieval steps
    retrieved_context:  list[str]
    retrieval_metadata: list[dict]
    compressed_context: list[str]  # after context_compressor
    response:         str
    is_relevant:      bool
    relevance_score:  float
    relevance_feedback: str
    iteration:        int
    messages:         Annotated[list[BaseMessage], add_messages]
    final_response:   str
    sources_used:     list[str]

    # Guardrails
    groundedness_score: float
    confidence_score:   float
    hallucinated_claims: list[str]

    # HyDE
    hyde_embedding:   Optional[list]   # float list of blended embedding


# ── JSON helper ────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    clean = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    start = clean.find("{")
    if start == -1:
        return {}
    depth, end = 0, -1
    for i, ch in enumerate(clean[start:], start):
        if ch == "{":   depth += 1
        elif ch == "}": depth -= 1
        if depth == 0:
            end = i
            break
    if end == -1:
        return {}
    try:
        return json.loads(clean[start:end + 1])
    except json.JSONDecodeError:
        return {}


# ── Node 1 — Query Rewriter (+ HyDE) ─────────────────────────────────────────

_REWRITE_PROMPT = _make_prompt([
    ("system", """You are an expert query optimiser for an equity research RAG system.
Transform the analyst's question into the most effective retrieval query.

Guidelines:
- Expand tickers to full company names (and vice versa)
- Include relevant financial terminology and synonyms
- Make the fiscal period explicit if implied
- On RETRY: SIMPLIFY the query — fewer terms, more specific
- If conversation history is provided, resolve pronouns and follow-up references

Respond with ONLY valid JSON:
{{"rewritten_query": "<optimised query>", "rationale": "<brief>"}}"""),
    ("human", """Company: {company_name} ({ticker})
Original query: {original_query}
Conversation history: {conv_history}
Iteration: {iteration}
Previous query: {previous_query}
Relevance feedback: {feedback}"""),
])


def query_rewriter(state: EquityRAGState) -> dict:
    iteration = state.get("iteration", 0) + 1
    orig      = state.get("original_query", "")
    ticker    = state.get("ticker", "")
    company   = state.get("company_name", ticker)
    session_id = state.get("session_id", "")

    # Inject conversation memory
    conv_history = ""
    if session_id:
        try:
            from .memory import ConversationStore
            conv_history = ConversationStore.get().get_context(session_id, max_chars=1500)
        except Exception:
            pass

    result = (_REWRITE_PROMPT | _llm()).invoke({
        "company_name":   company,
        "ticker":         ticker,
        "original_query": orig,
        "conv_history":   conv_history or "None",
        "iteration":      iteration,
        "previous_query": state.get("rewritten_query", ""),
        "feedback":       state.get("relevance_feedback", ""),
    })
    parsed = _parse_json(result.content)
    rw     = parsed.get("rewritten_query") or orig

    # HyDE: generate hypothetical doc and blend with query embedding (iteration=1 only)
    hyde_embedding = None
    if RAG_CONFIG.hyde_enabled and iteration == 1 and vs is not None:
        try:
            from .hyde import HyDE
            from .vector_store import _embed
            hyde = HyDE(llm_fn=_llm_fn, embed_fn=_embed)
            blended = hyde.embed(rw, company_name=company, ticker=ticker)
            if blended is not None:
                hyde_embedding = blended.tolist()
        except Exception as exc:
            logger.debug(f"[rag] HyDE failed ({exc}), using plain query embedding")

    logger.debug(f"[rag:{ticker}] rewriter iter={iteration} → {rw[:80]}")
    return {
        "rewritten_query": rw,
        "iteration":       iteration,
        "hyde_embedding":  hyde_embedding,
        "messages":        [HumanMessage(content=orig)] if iteration == 1 else [],
    }


# ── Node 2 — Query Decomposer ─────────────────────────────────────────────────

_DECOMPOSE_PROMPT = _make_prompt([
    ("system", """You are a query analyst for an equity research RAG system.
Determine if the query requires multi-hop reasoning across documents.

Multi-hop: comparing two periods, cross-document synthesis, derived metrics.
Single-hop: simple factual lookup from one document.

Respond ONLY with valid JSON:
{{"is_multi_hop": true/false, "sub_queries": ["<query1>", ...]}}"""),
    ("human", "Query: {query}\nCompany: {company_name} ({ticker})"),
])


def query_decomposer(state: EquityRAGState) -> dict:
    query   = state.get("rewritten_query", state.get("original_query", ""))
    result  = (_DECOMPOSE_PROMPT | _llm()).invoke({
        "query":        query,
        "company_name": state.get("company_name", ""),
        "ticker":       state.get("ticker", ""),
    })
    parsed       = _parse_json(result.content)
    is_multi_hop = bool(parsed.get("is_multi_hop", False))
    sub_queries  = parsed.get("sub_queries") or [query]
    if not isinstance(sub_queries, list) or not sub_queries:
        sub_queries = [query]
    return {"sub_queries": sub_queries, "is_multi_hop": is_multi_hop}


# ── Node 3 — Detail Checker ───────────────────────────────────────────────────

_DETAIL_PROMPT = _make_prompt([
    ("system", """You are an equity research RAG orchestrator.
Decide whether external retrieval is needed.

Retrieval IS needed for: specific figures, guidance, recent events, post-cutoff data.
Retrieval NOT needed for: methodology questions, definitions, reasoning over provided data.

Respond ONLY: {{"needs_retrieval": true/false, "reason": "<one sentence>"}}"""),
    ("human", "Query: {query}\nCompany: {company_name} ({ticker})"),
])


def detail_checker(state: EquityRAGState) -> dict:
    query  = state.get("rewritten_query", state.get("original_query", ""))
    result = (_DETAIL_PROMPT | _llm()).invoke({
        "query": query, "company_name": state.get("company_name", ""),
        "ticker": state.get("ticker", ""),
    })
    parsed = _parse_json(result.content)
    needs  = bool(parsed.get("needs_retrieval", True))
    return {"needs_retrieval": needs, "retrieval_reason": parsed.get("reason", "")}


# ── Node 4 — Agentic Source Selector ─────────────────────────────────────────

_SOURCE_PROMPT = _make_prompt([
    ("system", """You are an agentic retrieval planner for an equity research platform.

Available sources:
  "vector_db"   — Indexed filings and transcripts for this company.
  "internet"    — Live web search for current events and news.
  "tools_apis"  — SEC EDGAR, live price snapshot, Wikipedia.
  "combined"    — All three in parallel.

You can also plan a multi-step retrieval sequence by providing a "retrieval_plan"
— an ordered list of source names to try. The retriever executes them in order
and stops when enough context is collected.

Local corpus: {kb_size} indexed chunks.
Prefer "vector_db" when corpus ≥ 10 chunks and query is about historical data.

Respond with ONLY valid JSON:
{{"selected_source": "<source>",
  "retrieval_plan": ["<step1>", "<step2>"],
  "rationale": "<one sentence>"}}"""),
    ("human", "Query: {query}\nCompany: {company_name} ({ticker})\nReason: {reason}"),
])


def source_selector(state: EquityRAGState) -> dict:
    ticker = state.get("ticker", "UNKNOWN")
    query  = state.get("rewritten_query", state.get("original_query", ""))
    kb_sz  = vs.collection_size(ticker) if vs is not None else 0

    result = (_SOURCE_PROMPT | _llm()).invoke({
        "query": query, "company_name": state.get("company_name", ""),
        "ticker": ticker, "reason": state.get("retrieval_reason", ""),
        "kb_size": kb_sz,
    })
    parsed = _parse_json(result.content)
    source = parsed.get("selected_source", "combined")
    if source not in {"vector_db", "internet", "tools_apis", "combined"}:
        source = "combined"
    plan = parsed.get("retrieval_plan") or [source]

    return {
        "selected_source":  source,
        "retrieval_plan":   plan,
        "source_rationale": parsed.get("rationale", ""),
    }


# ── Node 5 — Retriever ────────────────────────────────────────────────────────

_SOURCE_PRIORITY = {"vector_db": 0, "sec_edgar": 1, "financial_snapshot": 2,
                    "web_search": 3, "wikipedia": 4}


def retriever(state: EquityRAGState) -> dict:
    ticker      = state.get("ticker", "UNKNOWN")
    query       = state.get("rewritten_query", state.get("original_query", ""))
    source      = state.get("selected_source", "combined")
    company     = state.get("company_name", ticker)
    sub_queries = state.get("sub_queries") or [query]
    primary_q   = sub_queries[0]
    hyde_vec    = state.get("hyde_embedding")

    import numpy as np
    hyde_arr = np.array(hyde_vec, dtype=np.float32) if hyde_vec else None

    chunk_meta_pairs: list[tuple[str, dict]] = []

    def _from_vector_db(q: str) -> list[tuple[str, dict]]:
        if vs is None:
            return []
        try:
            results = vs.query(q, ticker=ticker, top_k=TOP_K, hyde_vec=hyde_arr)
            return [(txt, {"source": "vector_db"}) for txt in results]
        except Exception as exc:
            logger.warning(f"[rag:{ticker}] vector_db failed: {exc}")
            return []

    def _from_internet(q: str) -> list[tuple[str, dict]]:
        if T is None:
            return []
        try:
            r = T.web_search.run(f"{company} {ticker} {q}")
            return [(str(r), {"source": "web_search"})] if r and "failed" not in r.lower() else []
        except Exception:
            return []

    def _from_edgar(q: str) -> list[tuple[str, dict]]:
        if T is None:
            return []
        try:
            r = T.sec_edgar_search.run(f"{ticker} {q}")
            return [(r, {"source": "sec_edgar"})] if r and "failed" not in r.lower() else []
        except Exception:
            return []

    def _from_snapshot() -> list[tuple[str, dict]]:
        if T is None:
            return []
        if not any(kw in query.lower() for kw in ("price", "p/e", "multiple", "market cap")):
            return []
        try:
            r = T.financial_snapshot.run(ticker)
            return [(r, {"source": "financial_snapshot"})] if r and "failed" not in r.lower() else []
        except Exception:
            return []

    def _from_wikipedia(q: str) -> list[tuple[str, dict]]:
        if T is None or not _is_background_query(q):
            return []
        try:
            r = T.wikipedia_lookup.run(f"{company} {q}")
            return [(r, {"source": "wikipedia"})] if r and "failed" not in r.lower() else []
        except Exception:
            return []

    # Dispatch
    if source == "vector_db":
        for sq in sub_queries:
            chunk_meta_pairs.extend(_from_vector_db(sq))
    elif source == "internet":
        chunk_meta_pairs.extend(_from_internet(primary_q))
    elif source == "tools_apis":
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futs = [pool.submit(_from_edgar, primary_q),
                    pool.submit(_from_snapshot),
                    pool.submit(_from_wikipedia, primary_q)]
            for f in concurrent.futures.as_completed(futs):
                chunk_meta_pairs.extend(f.result())
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            vdb_futs = [pool.submit(_from_vector_db, sq) for sq in sub_queries]
            net_f    = pool.submit(_from_internet, primary_q)
            edg_f    = pool.submit(_from_edgar,    primary_q)
            snap_f   = pool.submit(_from_snapshot)
            wiki_f   = pool.submit(_from_wikipedia, primary_q)
            for f in vdb_futs:
                chunk_meta_pairs.extend(f.result())
            for f in [edg_f, snap_f, net_f, wiki_f]:
                chunk_meta_pairs.extend(f.result())

    # Deduplicate
    seen: set[str] = set()
    deduped: list[tuple[str, dict]] = []
    for (text, meta) in chunk_meta_pairs:
        fp = text[:200].strip()
        if fp not in seen:
            seen.add(fp)
            deduped.append((text, meta))

    deduped.sort(key=lambda x: _SOURCE_PRIORITY.get(x[1].get("source", ""), 99))

    chunk_texts = [t for t, _ in deduped]
    metadata    = [m for _, m in deduped]
    logger.debug(f"[rag:{ticker}] retriever: {len(chunk_texts)} unique chunks from {source}")
    return {
        "retrieved_context":  chunk_texts,
        "retrieval_metadata": metadata,
        "sources_used":       list({m["source"] for m in metadata}),
    }


# ── Node 6 — Context Compressor ───────────────────────────────────────────────

def context_compressor(state: EquityRAGState) -> dict:
    """LLM extraction + redundancy removal — only runs when compression is enabled."""
    if not RAG_CONFIG.compression_enabled:
        return {"compressed_context": state.get("retrieved_context", [])}

    chunks = state.get("retrieved_context", [])
    query  = state.get("rewritten_query", state.get("original_query", ""))

    try:
        from .compression import ContextCompressor
        compressor = ContextCompressor(
            llm_fn=_llm_fn,
            keyword_threshold=0.10,
        )
        result = compressor.compress(query, chunks, max_output_chars=RAG_CONFIG.compression_max_chars)
        compressed = result.compressed_chunks
    except Exception as exc:
        logger.warning(f"[rag] context_compressor failed ({exc}); using raw chunks")
        compressed = chunks

    return {"compressed_context": compressed}


# ── Node 7 — Response Generator ───────────────────────────────────────────────

_RESPONSE_PROMPT = _make_prompt([
    ("system", """You are a senior equity research analyst.
Answer the analyst's question precisely using the retrieved context.
- Quote specific figures (revenue, EPS, margins) with fiscal periods.
- Cite source type when available.
- If context does not contain the answer, say so explicitly — do not invent figures.
- Use bullet points for fact lists; prose for narrative answers.

Company: {company_name} ({ticker})

Retrieved Context:
{context}

Sources: {sources}"""),
    ("human", "{query}"),
])


def response_generator(state: EquityRAGState) -> dict:
    ticker  = state.get("ticker", "UNKNOWN")
    query   = state.get("rewritten_query", state.get("original_query", ""))
    chunks  = state.get("compressed_context") or state.get("retrieved_context", [])
    sources = state.get("sources_used", [])

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


# ── Node 8 — Relevance Checker + Guardrails ───────────────────────────────────

_RELEVANCE_PROMPT = _make_prompt([
    ("system", """You are a QA judge for equity research — context-faithfulness check.
Score whether the generated answer is grounded in the retrieved context.

1.0 — All figures explicitly in context; directly answers query
0.8 — Mostly grounded; minor citation gaps
0.6 — Partially grounded; some figures unsupported
0.4 — Mostly parametric / not context-grounded
< 0.4 — Contradicts context or ignores it

Threshold: {threshold}

Retrieved Context (first 3,000 chars):
{context_preview}

Respond ONLY valid JSON:
{{"is_relevant": true/false, "score": <0.0–1.0>,
  "feedback": "<specific figures lacking context support>"}}"""),
    ("human", "Original query: {original_query}\nCompany: {company_name} ({ticker})\nResponse: {response}"),
])


def relevance_checker(state: EquityRAGState) -> dict:
    ticker  = state.get("ticker", "UNKNOWN")
    chunks  = state.get("compressed_context") or state.get("retrieved_context", [])
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

    # Guardrails
    groundedness_score  = 0.0
    confidence_score    = 0.0
    hallucinated_claims: list[str] = []
    if chunks and state.get("response"):
        try:
            from .guardrails import GuardrailsChecker
            checker = GuardrailsChecker(
                llm_fn=_llm_fn,
                groundedness_threshold=RAG_CONFIG.groundedness_threshold,
            )
            retrieval_quality = min(1.0, len(chunks) / max(TOP_K, 1))
            gr = checker.check(
                query              = state.get("original_query", ""),
                response           = state.get("response", ""),
                context_chunks     = chunks,
                relevance_score    = score,
                retrieval_quality  = retrieval_quality,
            )
            groundedness_score  = gr.groundedness_score
            confidence_score    = gr.confidence_score
            hallucinated_claims = gr.hallucinated_numbers + gr.unsupported_claims
        except Exception as exc:
            logger.debug(f"[rag] guardrails check failed ({exc})")

    logger.debug(
        f"[rag:{ticker}] relevance={score:.2f} grounded={groundedness_score:.2f} "
        f"confidence={confidence_score:.2f} accepted={relevant}"
    )

    if relevant:
        # Persist exchange in conversation memory
        session_id = state.get("session_id", "")
        if session_id:
            try:
                from .memory import ConversationStore
                ConversationStore.get().add_exchange(
                    session_id = session_id,
                    question   = state.get("original_query", ""),
                    answer     = state.get("response", ""),
                    sources    = state.get("sources_used", []),
                    llm_fn     = _llm_fn,
                    max_chars  = RAG_CONFIG.memory_max_chars,
                    max_turns  = RAG_CONFIG.memory_max_turns,
                )
            except Exception:
                pass

        return {
            "is_relevant":        True,
            "relevance_score":    score,
            "relevance_feedback": feedback,
            "groundedness_score": groundedness_score,
            "confidence_score":   confidence_score,
            "hallucinated_claims": hallucinated_claims,
            "final_response":     state.get("response", ""),
        }

    return {
        "is_relevant":        False,
        "relevance_score":    score,
        "relevance_feedback": feedback,
        "groundedness_score": groundedness_score,
        "confidence_score":   confidence_score,
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_after_detail(state: EquityRAGState) -> str:
    return "source_selector" if state.get("needs_retrieval", True) else "response_generator"


def _route_after_relevance(state: EquityRAGState) -> str:
    if state.get("is_relevant", False):
        return "END"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        logger.warning("[rag] max iterations reached — returning best answer")
        return "END"
    return "query_rewriter"


# ── Graph ─────────────────────────────────────────────────────────────────────

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
        g.add_node("context_compressor", context_compressor)
        g.add_node("response_generator", response_generator)
        g.add_node("relevance_checker",  relevance_checker)

        g.set_entry_point("query_rewriter")
        g.add_edge("query_rewriter",     "query_decomposer")
        g.add_edge("query_decomposer",   "detail_checker")
        g.add_edge("source_selector",    "retriever")
        g.add_edge("retriever",          "context_compressor")
        g.add_edge("context_compressor", "response_generator")
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


# ── Public API ─────────────────────────────────────────────────────────────────

def run(
    question:     str,
    company_name: str = "",
    ticker:       str = "UNKNOWN",
    session_id:   str = "",
) -> dict:
    """
    Run the full equity RAG pipeline synchronously.
    Includes HyDE, context compression, memory, and guardrails.
    Returns state dict with `final_response`, `sources_used`, `relevance_score`,
    `confidence_score`, `groundedness_score`.
    """
    if not _HAS_LANGCHAIN:
        raise RuntimeError(
            "LangChain/LangGraph is not installed. Run:\n"
            "  pip install langchain-core langchain-openai langgraph\n"
            "and set at least one LLM API key in .env"
        )
    from ..core.config import validate_llm_config
    validate_llm_config()

    app         = _get_graph()
    final_state = app.invoke({
        "original_query": question,
        "company_name":   company_name,
        "ticker":         ticker.upper(),
        "session_id":     session_id,
        "iteration":      0,
        "messages":       [],
        "sub_queries":    [question],
        "is_multi_hop":   False,
    })
    if not final_state.get("final_response") and final_state.get("response"):
        final_state["final_response"] = final_state["response"]
    return final_state


async def stream_run(
    question:     str,
    company_name: str = "",
    ticker:       str = "UNKNOWN",
    session_id:   str = "",
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields tokens as they are generated.
    Runs the pipeline up to the response_generator stage, then streams the
    final LLM response token by token for SSE delivery.
    """
    if not _HAS_LANGCHAIN:
        raise RuntimeError("LangChain not installed")
    from ..core.config import validate_llm_config
    validate_llm_config()

    # Run pipeline phases 1–6 synchronously to get compressed context
    app   = _get_graph()
    state: EquityRAGState = {
        "original_query": question,
        "company_name":   company_name,
        "ticker":         ticker.upper(),
        "session_id":     session_id,
        "iteration":      0,
        "messages":       [],
        "sub_queries":    [question],
        "is_multi_hop":   False,
    }

    # Run nodes 1–6 (up to but not including response_generator)
    for node_fn in (query_rewriter, query_decomposer, detail_checker):
        state.update(node_fn(state))

    if state.get("needs_retrieval", True):
        state.update(source_selector(state))
        state.update(retriever(state))
        state.update(context_compressor(state))

    # Stream the final generation
    chunks  = state.get("compressed_context") or state.get("retrieved_context", [])
    sources = state.get("sources_used", [])
    context = "\n\n---\n\n".join(chunks) if chunks else "No external context retrieved."
    if len(context) > 12000:
        context = context[:12000] + "\n\n[... truncated ...]"

    system_msg = (
        f"You are a senior equity research analyst. Answer using the retrieved context. "
        f"Company: {company_name} ({ticker}). Sources: {', '.join(sources) or 'parametric'}"
    )
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    # LangChain streaming
    llm     = _llm()
    prompt  = _make_prompt([("system", system_msg), ("human", "{user}")])
    full    = prompt | llm

    full_response = ""
    async for chunk in full.astream({"user": user_msg}):
        token = getattr(chunk, "content", str(chunk))
        if token:
            full_response += token
            yield token

    # Post-stream: save to memory
    if session_id and full_response:
        try:
            from .memory import ConversationStore
            ConversationStore.get().add_exchange(
                session_id=session_id, question=question, answer=full_response,
                sources=sources, llm_fn=_llm_fn,
                max_chars=RAG_CONFIG.memory_max_chars,
                max_turns=RAG_CONFIG.memory_max_turns,
            )
        except Exception:
            pass


def query(
    question:     str,
    company_name: str = "",
    ticker:       str = "UNKNOWN",
) -> str:
    """Convenience wrapper — returns just the answer string."""
    result = run(question, company_name=company_name, ticker=ticker)
    return result.get("final_response", result.get("response", ""))
