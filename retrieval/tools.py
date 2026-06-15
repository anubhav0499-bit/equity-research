"""
LangChain tools for the equity-research RAG pipeline.

Tools
-----
web_search          Tavily (preferred) or DuckDuckGo fallback
sec_edgar_search    SEC EDGAR full-text search scoped to a ticker
financial_snapshot  Quick price / market-cap / P/E via yfinance
wikipedia_lookup    Wikipedia article summary
calculator          AST-safe arithmetic evaluator
"""

from __future__ import annotations
import ast
import operator
from typing import Optional
from langchain_core.tools import tool
from loguru import logger

from ..core.config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY,
    GOOGLE_API_KEY, ACQUISITION_CONFIG,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os


# ── Web Search ────────────────────────────────────────────────────────────

_search_tool = None

def _get_search():
    global _search_tool
    if _search_tool is not None:
        return _search_tool
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        from langchain_community.tools.tavily_search import TavilySearchResults
        _search_tool = TavilySearchResults(
            max_results=8,
            include_answer=True,
            include_raw_content=True,
            tavily_api_key=tavily_key,
        )
        logger.debug("[retrieval.tools] search: Tavily")
    else:
        from langchain_community.tools import DuckDuckGoSearchResults
        _search_tool = DuckDuckGoSearchResults(num_results=8)
        logger.debug("[retrieval.tools] search: DuckDuckGo (no Tavily key)")
    return _search_tool


@tool
def web_search(query: str) -> str:
    """
    Search the open internet for current financial news, analyst commentary,
    and real-time market data. Use for recent events, earnings reactions,
    regulatory news, and anything time-sensitive.
    """
    try:
        return str(_get_search().run(query))
    except Exception as e:
        logger.warning(f"[retrieval.tools] web_search failed: {e}")
        return f"Search failed: {e}"


# ── SEC EDGAR ─────────────────────────────────────────────────────────────

@tool
def sec_edgar_search(query: str) -> str:
    """
    Search SEC EDGAR full-text search for filings related to the query.
    Useful for finding 10-K risk factors, MD&A passages, proxy statements,
    and 8-K disclosures by keyword across a company's filing history.
    Format query as: '<ticker> <keywords>', e.g. 'AAPL revenue recognition'.
    """
    try:
        import httpx
        params = {
            "q": query,
            "dateRange": "custom",
            "startdt": "2019-01-01",
            "forms": "10-K,10-Q,8-K,20-F",
            "_source": "hits.hits._source",
        }
        url = "https://efts.sec.gov/LATEST/search-index?q={q}&forms={forms}".format(**{
            "q": query.replace(" ", "+"),
            "forms": "10-K,10-Q,8-K",
        })
        r = httpx.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={"q": query, "forms": "10-K,10-Q,8-K", "hits.hits.total.value": 5},
            headers={"User-Agent": ACQUISITION_CONFIG.sec_user_agent},
            timeout=15,
        )
        if r.status_code != 200:
            return f"EDGAR search returned status {r.status_code}"
        data = r.json()
        hits = data.get("hits", {}).get("hits", [])[:5]
        if not hits:
            return "No EDGAR filings found for this query."
        lines = []
        for h in hits:
            src = h.get("_source", {})
            lines.append(
                f"[{src.get('file_date', '?')}] {src.get('form_type', '?')} — "
                f"{src.get('entity_name', '?')}: {src.get('file_num', '')} "
                f"https://www.sec.gov/Archives/{src.get('file_path', '')}"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"[retrieval.tools] sec_edgar_search failed: {e}")
        return f"EDGAR search failed: {e}"


# ── Financial Snapshot ────────────────────────────────────────────────────

@tool
def financial_snapshot(ticker: str) -> str:
    """
    Fetch a quick financial snapshot for a ticker: current price, market cap,
    P/E ratio, EV/EBITDA, revenue (TTM), and 52-week range.
    Use for real-time valuation context during research.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker.strip().upper())
        info = t.info
        fields = {
            "Current Price":  info.get("currentPrice") or info.get("regularMarketPrice"),
            "Market Cap":     info.get("marketCap"),
            "P/E (TTM)":      info.get("trailingPE"),
            "P/E (Forward)":  info.get("forwardPE"),
            "EV/EBITDA":      info.get("enterpriseToEbitda"),
            "Revenue (TTM)":  info.get("totalRevenue"),
            "52-week High":   info.get("fiftyTwoWeekHigh"),
            "52-week Low":    info.get("fiftyTwoWeekLow"),
            "Beta":           info.get("beta"),
        }
        lines = [f"{k}: {v:,.2f}" if isinstance(v, float) else f"{k}: {v}"
                 for k, v in fields.items() if v is not None]
        return f"Financial snapshot — {ticker.upper()}\n" + "\n".join(lines)
    except Exception as e:
        logger.warning(f"[retrieval.tools] financial_snapshot failed for {ticker}: {e}")
        return f"Financial snapshot failed: {e}"


# ── Wikipedia ─────────────────────────────────────────────────────────────

_wiki_tool = None

def _get_wiki():
    global _wiki_tool
    if _wiki_tool is None:
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper
        _wiki_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=4000)
        )
    return _wiki_tool


@tool
def wikipedia_lookup(topic: str) -> str:
    """
    Look up a company, industry, concept, or regulation on Wikipedia.
    Use for background context: business model overview, industry history,
    regulatory framework, or when the query needs encyclopaedic grounding.
    """
    try:
        return _get_wiki().run(topic)
    except Exception as e:
        logger.warning(f"[retrieval.tools] wikipedia_lookup failed: {e}")
        return f"Wikipedia lookup failed: {e}"


# ── Calculator ────────────────────────────────────────────────────────────

_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    raise ValueError(f"Expression type not allowed: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely. Supports +, -, *, /, //, %, **.
    Use for computing valuation metrics, growth rates, margin calculations,
    and any arithmetic needed during research.
    Examples: '(120 - 95) / 95 * 100' → 26.32 (% change)
    """
    try:
        result = _safe_eval(ast.parse(expression.strip(), mode="eval"))
        return str(round(result, 6) if isinstance(result, float) else result)
    except Exception as e:
        return f"Calculation error: {e}"


# ── Registry ──────────────────────────────────────────────────────────────

ALL_TOOLS = [web_search, sec_edgar_search, financial_snapshot, wikipedia_lookup, calculator]
