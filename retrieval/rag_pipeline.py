"""
Equity-Research RAG Pipeline — LangGraph multi-agent retrieval.

Integrates with the equity research platform via:
  - Per-ticker vector store (LlamaIndex + ChromaDB)
  - Platform LLM config (same provider / API keys as the 17-agent workflow)
  - Equity-specific tools: web_search, sec_edgar_search, financial_snapshot

Architecture (6 nodes, same as the generic agentic_rag but equity-aware):

    Query
      ↓
  [1] query_rewriter       — Optimise query for financial document retrieval
      ↓
  [2] detail_checker       — Need retrieval or answer from LLM knowledge?
      ├── No  ──────────────────────────────────────────────────────────────┐
      ↓ Yes                                                                 │
  [3] source_selector      — Vector store, internet, tools, or combined?   │
      ↓                                                                     │
  [4] retriever            — Fetch context from chosen source(s)           │
      └────────────────────────────────────────────────────────────────── [5] response_generator
                                                                              ↓
                                                                          [6] relevance_checker
                                                                              ├── Yes → final_response
                                                                              └── No  → loop (max 5)
"""

from __future__ import annotations
import json
import re
import threading
from functools import lru_cache
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from loguru import logger

from . import vector_store as vs
from . import tools as T
from ..core.config import (
    LLM_CONFIG, OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY,
    GOOGLE_API_KEY, TOGETHER_API_KEY, OPENROUTER_API_KEY,
)

MAX_ITERATIONS      = 5
RELEVANCE_THRESHOLD = 0.70
TOP_K               = 5


# ── LangChain LLM bridge ──────────────────────────────────────────────────
# Build a LangChain BaseChatModel from the same provider/keys the platform uses.

@lru_cache(maxsize=None)
def _get_llm(temperature: float = 0.0):
    """Return a cached LangChain LLM using the platform's configured provider."""
    backend = LLM_CONFIG.provider
    if backend == "auto":
        if GROQ_API_KEY:        backend = "groq"
        elif OPENAI_API_KEY:    backend = "openai"
        elif ANTHROPIC_API_KEY: backend = "anthropic"
        elif TOGETHER_API_KEY:  backend = "together"
        elif OPENROUTER_API_KEY: backend = "openrouter"
        elif GOOGLE_API_KEY:    backend = "gemini"
        else:                   backend = "ollama"

    model = LLM_CONFIG.primary_model

    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature,
                         api_key=OPENAI_API_KEY)
    if backend == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature,
                             api_key=ANTHROPIC_API_KEY, max_tokens=4096)
    if backend == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature,
                        groq_api_key=GROQ_API_KEY)
    if backend in ("gemini", "google"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature,
                                      google_api_key=GOOGLE_API_KEY)
    if backend in ("together", "openrouter"):
        from langchain_openai import ChatOpenAI
        base_url = ("https://api.together.xyz/v1" if backend == "together"
                    else "https://openrouter.ai/api/v1")
        key = TOGETHER_API_KEY if backend == "together" else OPENROUTER_API_KEY
        return ChatOpenAI(model=model, temperature=temperature,
                         base_url=base_url, api_key=key)
    # Ollama default
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=LLM_CONFIG.primary_model or "llama3.1",
        base_url=LLM_CONFIG.ollama_base_url,
        temperature=temperature,
    )


def _llm():
    return _get_llm(temperature=0.0)


# ── State ─────────────────────────────────────────────────────────────────

class EquityRAGState(TypedDict, total=False):
    # Company context (set once at pipeline entry)
    company_name:    str
    ticker:          str

    # Pipeline fields
    original_query:  str
    rewritten_query: str
    needs_retrieval: bool
    retrieval_reason: str
    selected_source: str        # "vector_db" | "internet" | "tools_apis" | "combined"
    source_rationale: str
    retrieved_context: list[str]
    retrieval_metadata: list[dict]
    response:        str
    is_relevant:     bool
    relevance_score: float
    relevance_feedback: str
    iteration:       int
    messages:        Annotated[list[BaseMessage], add_messages]
    final_response:  str
    sources_used:    list[str]


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

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert query optimiser for an equity research RAG system.
Transform the analyst's question into the most effective retrieval query for searching
company filings, earnings transcripts, analyst reports, and financial databases.

Guidelines:
- Expand tickers to full company names (and vice versa)
- Include relevant financial terminology and synonyms
- Make the fiscal period explicit if implied (e.g. "last quarter" → "Q3 FY2024")
- On retry iterations, incorporate the relevance feedback to fix gaps

Respond with ONLY valid JSON:
{{"rewritten_query": "<optimised query>", "rationale": "<brief explanation>"}}"""),
    ("human", """Company: {company_name} ({ticker})
Original query: {original_query}
Iteration: {iteration}
Previous query: {previous_query}
Relevance feedback: {feedback}"""),
])


def query_rewriter(state: EquityRAGState) -> dict:
    iteration  = state.get("iteration", 0) + 1
    orig       = state.get("original_query", "")
    ticker     = state.get("ticker", "")
    company    = state.get("company_name", ticker)

    result = (_REWRITE_PROMPT | _llm()).invoke({
        "company_name":  company,
        "ticker":        ticker,
        "original_query": orig,
        "iteration":     iteration,
        "previous_query": state.get("rewritten_query", ""),
        "feedback":      state.get("relevance_feedback", ""),
    })
    parsed = _parse_json(result.content)
    rw = parsed.get("rewritten_query") or orig

    logger.debug(f"[rag:{ticker}] rewriter iter={iteration} → {rw[:80]}")
    return {
        "rewritten_query": rw,
        "iteration":       iteration,
        "messages":        [HumanMessage(content=orig)] if iteration == 1 else [],
    }


# ── Node 2 — Detail Checker ───────────────────────────────────────────────

_DETAIL_PROMPT = ChatPromptTemplate.from_messages([
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
    query = state.get("rewritten_query", state.get("original_query", ""))
    result = (_DETAIL_PROMPT | _llm()).invoke({
        "query":        query,
        "company_name": state.get("company_name", ""),
        "ticker":       state.get("ticker", ""),
    })
    parsed = _parse_json(result.content)
    needs  = bool(parsed.get("needs_retrieval", True))
    logger.debug(f"[rag] detail_checker: needs_retrieval={needs}")
    return {"needs_retrieval": needs, "retrieval_reason": parsed.get("reason", "")}


# ── Node 3 — Source Selector ──────────────────────────────────────────────

_SOURCE_PROMPT = ChatPromptTemplate.from_messages([
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

Respond with ONLY valid JSON:
{{"selected_source": "<vector_db|internet|tools_apis|combined>",
  "rationale": "<one sentence>"}}"""),
    ("human", "Query: {query}\nCompany: {company_name} ({ticker})\nReason for retrieval: {reason}"),
])


def source_selector(state: EquityRAGState) -> dict:
    ticker = state.get("ticker", "UNKNOWN")
    query  = state.get("rewritten_query", state.get("original_query", ""))
    kb_sz  = vs.collection_size(ticker)

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


# ── Node 4 — Retriever ────────────────────────────────────────────────────

def retriever(state: EquityRAGState) -> dict:
    ticker  = state.get("ticker", "UNKNOWN")
    query   = state.get("rewritten_query", state.get("original_query", ""))
    source  = state.get("selected_source", "combined")
    company = state.get("company_name", ticker)

    chunks:   list[str]  = []
    metadata: list[dict] = []

    def _from_vector_db():
        c = vs.query(query, ticker=ticker, top_k=TOP_K)
        for txt in c:
            chunks.append(txt)
            metadata.append({"source": "vector_db"})

    def _from_internet():
        result = T.web_search.run(f"{company} {ticker} {query}")
        if result and "failed" not in result.lower():
            chunks.append(str(result))
            metadata.append({"source": "web_search"})

    def _from_tools():
        # SEC EDGAR
        edgar = T.sec_edgar_search.run(f"{ticker} {query}")
        if edgar and "failed" not in edgar.lower():
            chunks.append(edgar)
            metadata.append({"source": "sec_edgar"})
        # Financial snapshot for live price/multiple queries
        if any(kw in query.lower() for kw in ("price", "p/e", "multiple", "market cap", "valuation")):
            snap = T.financial_snapshot.run(ticker)
            if snap and "failed" not in snap.lower():
                chunks.append(snap)
                metadata.append({"source": "financial_snapshot"})
        # Wikipedia for background/industry context
        wiki = T.wikipedia_lookup.run(f"{company} {query}")
        if wiki and "failed" not in wiki.lower():
            chunks.append(wiki)
            metadata.append({"source": "wikipedia"})

    if source == "vector_db":
        _from_vector_db()
    elif source == "internet":
        _from_internet()
    elif source == "tools_apis":
        _from_tools()
    else:  # combined
        _from_vector_db()
        _from_internet()
        _from_tools()

    logger.debug(f"[rag:{ticker}] retriever: {len(chunks)} chunks from {source}")
    return {
        "retrieved_context": chunks,
        "retrieval_metadata": metadata,
        "sources_used": list({m["source"] for m in metadata}),
    }


# ── Node 5 — Response Generator ───────────────────────────────────────────

_RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior equity research analyst.
Answer the analyst's question precisely and concisely using the retrieved context below.
- Quote specific figures (revenue, EPS, margins) with fiscal periods.
- Cite the source (filing type, date) when available.
- If the context does not answer the question, say so and answer from your own knowledge.
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


# ── Node 6 — Relevance Checker ────────────────────────────────────────────

_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a quality-assurance judge for equity research.
Evaluate whether the generated answer fully addresses the analyst's question.

Score 0.0 to 1.0:
  1.0 — Precise figures cited with periods, sources identified, fully answers the question
  0.8 — Good answer, minor gaps or could be more specific
  0.6 — Partially answers; missing key metrics or wrong period
  0.4 — Vague, off-topic, or lacks the quantitative specificity needed
  < 0.4 — Incorrect, hallucinated, or completely irrelevant to the query

Acceptance threshold: {threshold}

Respond with ONLY valid JSON:
{{"is_relevant": true/false, "score": <0.0–1.0>,
  "feedback": "<what was missing, to guide a better retrieval query>"}}"""),
    ("human", """Original query: {original_query}
Rewritten query: {rewritten_query}
Company: {company_name} ({ticker})
Generated response: {response}"""),
])


def relevance_checker(state: EquityRAGState) -> dict:
    ticker = state.get("ticker", "UNKNOWN")
    result = (_RELEVANCE_PROMPT | _llm()).invoke({
        "original_query":  state.get("original_query", ""),
        "rewritten_query": state.get("rewritten_query", ""),
        "company_name":    state.get("company_name", ticker),
        "ticker":          ticker,
        "response":        state.get("response", ""),
        "threshold":       RELEVANCE_THRESHOLD,
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
        logger.warning(f"[rag] max iterations reached — returning best answer")
        return "END"
    return "query_rewriter"


# ── Graph ─────────────────────────────────────────────────────────────────

_compiled = None
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
        g.add_node("detail_checker",     detail_checker)
        g.add_node("source_selector",    source_selector)
        g.add_node("retriever",          retriever)
        g.add_node("response_generator", response_generator)
        g.add_node("relevance_checker",  relevance_checker)

        g.set_entry_point("query_rewriter")
        g.add_edge("query_rewriter",     "detail_checker")
        g.add_edge("source_selector",    "retriever")
        g.add_edge("retriever",          "response_generator")
        g.add_edge("response_generator", "relevance_checker")

        g.add_conditional_edges("detail_checker", _route_after_detail,
                                {"source_selector": "source_selector",
                                 "response_generator": "response_generator"})
        g.add_conditional_edges("relevance_checker", _route_after_relevance,
                                {"END": END, "query_rewriter": "query_rewriter"})
        _compiled = g.compile()
    return _compiled


# ── Public API ────────────────────────────────────────────────────────────

def run(
    question: str,
    company_name: str = "",
    ticker: str = "UNKNOWN",
) -> dict:
    """
    Run the full equity RAG pipeline for one question about a company.
    Returns the final state dict with `final_response`, `sources_used`, `relevance_score`.
    """
    app = _get_graph()
    final_state = app.invoke({
        "original_query": question,
        "company_name":   company_name,
        "ticker":         ticker.upper(),
        "iteration":      0,
        "messages":       [],
    })
    if not final_state.get("final_response") and final_state.get("response"):
        final_state["final_response"] = final_state["response"]
    return final_state


def query(
    question: str,
    company_name: str = "",
    ticker: str = "UNKNOWN",
) -> str:
    """
    Convenience wrapper — returns just the answer string.
    Use this from BaseAgent.rag_query().
    """
    result = run(question, company_name=company_name, ticker=ticker)
    return result.get("final_response", result.get("response", ""))
