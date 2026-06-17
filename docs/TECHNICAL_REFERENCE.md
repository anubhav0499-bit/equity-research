# Equity Intelligence Research Platform — Technical Reference

> **Handover document.** Covers architecture, configuration, data flow, deployment,
> testing, and extension points. Written against commit `3fa8c0b` (2026-06-17).

---

## Table of Contents

1. [Platform Overview](#1-platform-overview)
2. [Repository Layout](#2-repository-layout)
3. [Setup & Installation](#3-setup--installation)
4. [Configuration Reference](#4-configuration-reference)
5. [LLM Provider Cascade](#5-llm-provider-cascade)
6. [17-Agent Pipeline](#6-17-agent-pipeline)
7. [RAG Subsystem](#7-rag-subsystem)
8. [FastAPI HTTP Server](#8-fastapi-http-server)
9. [Forensic Scoring Models](#9-forensic-scoring-models)
10. [Storage Layer](#10-storage-layer)
11. [Running the Platform](#11-running-the-platform)
12. [Testing & Evaluation](#12-testing--evaluation)
13. [Performance Benchmarks](#13-performance-benchmarks)
14. [Known Limitations](#14-known-limitations)
15. [Deployment Checklist](#15-deployment-checklist)
16. [Extension Guide](#16-extension-guide)

---

## 1. Platform Overview

The platform produces institutional-grade equity research reports autonomously.
Given a company name, it retrieves SEC filings, earnings transcripts, and market
data; runs forensic, valuation, and risk models; and writes a 15 000–25 000-word
DOCX report — without human intervention.

**Key numbers**

| Dimension | Value |
|---|---|
| Agents | 20 |
| Orchestration phases | 6 (A–F) |
| Research sequence steps | 11 (macro → industry → business → management → financial → risks → accounting → governance → forecast → valuation → thesis) |
| RAG pipeline nodes | 7 (LangGraph) |
| Embedding model | `BAAI/bge-large-en-v1.5` (1024-dim) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector DB | ChromaDB (per-ticker persistent collections) |
| LLM providers supported | OpenAI, Anthropic, Groq, Gemini, Together, OpenRouter, Ollama |
| Compliance checks | 17 (7 Indian standards + 10 global standards) |
| Backtest score | 92.7 / 100 (MRR 0.976, Security 100/100, Scalability 96/100) |

---

## 2. Repository Layout

```
equity_research/
├── main.py                     CLI entry point
├── config.yaml                 YAML overrides (env vars take precedence)
├── requirements.txt            All pip dependencies
├── .env.example                Copy to .env and fill API keys
│
├── core/
│   ├── config.py               All typed settings, validate_llm_config()
│   ├── llm_manager.py          Provider-agnostic LLM wrapper
│   ├── logging_setup.py        Loguru configuration
│   └── research_philosophy.py  CIO research philosophy — RESEARCH_SEQUENCE (11 steps),
│                               AGENT_SPECS (20 agents), RAG_DOCUMENT_TAG_FIELDS,
│                               SOURCE_PRIORITY, EVIDENCE_RULES, REPORT_SECTIONS_20
│
├── agents/                     20 individual research agents
│   ├── base_agent.py           Abstract base — retry, audit, timing, _latest_fin()
│   ├── company_profiling.py    Phase A  — ticker resolution, sector, exchange
│   ├── filing_retrieval.py     Phase A2 — SEC EDGAR / BSE / NSE filings
│   ├── financial_extraction.py Phase B  — P&L, balance sheet, cash flow (5-yr)
│   ├── market_data.py          Phase B  — live price, market cap, P/E
│   ├── transcript_retrieval.py Phase B  — earnings call transcripts
│   ├── historical_data.py      Phase B  — 7-year price & volume history
│   ├── accounting_quality.py   Phase C  — accruals, revenue quality
│   ├── forensic_accounting.py  Phase C  — Beneish M-score, Altman Z, Piotroski;
│   │                                      9 frameworks + 10-case fraud learning corpus
│   ├── risk_analysis.py        Phase C  — macro/credit/operational risks
│   ├── earnings_quality.py     Phase C  — beat/miss patterns, guidance quality
│   ├── industry_intelligence.py Phase C — Porter Five Forces, TAM, attractiveness score
│   ├── management_governance.py Phase C — governance/credibility/capital-allocation scores,
│   │                                      board independence, promoter pledging, RPT analysis
│   ├── esg_sustainability.py   Phase C  — BRSR (India) + ISSB/SASB/GRI/TCFD (Global)
│   ├── financial_modeling_agent.py Phase D — 5-yr income/FCF model
│   ├── valuation_agent.py      Phase D  — DCF, EV/EBITDA, P/E band comps
│   ├── scenario_analysis.py    Phase D  — bull/base/bear scenarios
│   ├── narrative_agent.py      Phase E  — ThesisComponent + variant perception +
│   │                                      structured report sections (~25 000 words)
│   ├── compliance_agent.py     Phase E  — 17 checks: SEBI RA Regs, LODR, Companies Act
│   │                                      2013, Ind AS, IFRS, IOSCO, CFA, OECD,
│   │                                      ISSB S1+S2, SASB, GRI (jurisdiction-gated)
│   └── report_generation.py    Phase F  — DOCX assembly, charts, formatting
│
├── orchestrator/
│   ├── workflow.py             ResearchOrchestrator — phase A–F coordination
│   └── state.py                ResearchState dataclass — shared agent memory
│
├── retrieval/
│   ├── __init__.py             Lazy imports (no eager llama_index at import time)
│   ├── vector_store.py         ChromaDB + LlamaIndex + BM25+RRF+CrossEncoder
│   ├── rag_pipeline.py         7-node LangGraph pipeline
│   ├── tools.py                LangChain tools: calculator, SEC, web, Wikipedia
│   └── ingest.py               Document ingestion helpers
│
├── api/
│   ├── __init__.py
│   └── server.py               FastAPI production server
│
├── forensics/
│   ├── beneish.py              Beneish M-score (8-variable, threshold −1.78)
│   ├── altman.py               Altman Z-score (EM model; safe > 2.60)
│   └── piotroski.py            Piotroski F-score (9-point, ≥ 7 strong)
│
├── modeling/
│   ├── financial_model.py      5-year income statement / FCF projection
│   └── forecaster.py           Revenue growth + margin regression
│
├── valuation/
│   ├── dcf.py                  WACC-discounted FCF + terminal value
│   └── relative.py             EV/EBITDA and P/E peer comps
│
├── storage/
│   ├── storage_manager.py      Per-run file organisation
│   ├── audit_trail.py          Agent start/end/error event log (JSONL)
│   └── database.py             SQLite run registry + DuckDB analytics
│
├── reporting/
│   └── docx_generator.py       python-docx DOCX builder with charts
│
├── models/
│   ├── research.py             ResearchState, AgentOutput, AgentStatus, Finding,
│   │                           ThesisComponent (frozen), ThesisCase (frozen),
│   │                           DocumentTag, SourceType, EvidenceLevel
│   ├── company.py              CompanyProfile
│   ├── financials.py           FinancialHistory, FinancialPeriod
│   ├── valuation.py            ValuationSummary, ScenarioSet
│   └── report.py               ReportConfig, ReportSection
│
├── tests/
│   ├── stress_test.py          48-test infrastructure stress suite
│   └── rag_backtest.py         45-query RAG evaluation + 32 security tests
│
└── data/
    └── chroma_db/<TICKER>/     Per-ticker ChromaDB persistent collections
```

---

## 3. Setup & Installation

### Prerequisites

- Python 3.10+ (tested on 3.14)
- 8 GB RAM minimum (16 GB recommended for BGE-large + ChromaDB in process)
- 2 GB disk for models (BGE-large downloads automatically from HuggingFace)

### Install

```bash
git clone https://github.com/anubhav0499-bit/equity-research.git
cd equity-research

pip install -r requirements.txt

# Optional: faster PDF extraction
pip install pymupdf pdfplumber

# Optional: Playwright for dynamic page scraping
playwright install chromium
```

### Environment

```bash
cp .env.example .env
# Edit .env — add at least one LLM API key (see Section 5)
```

### Verify installation

```bash
python main.py --check
```

Expected: all core packages show `✓`.

---

## 4. Configuration Reference

Configuration is layered: **environment variables** override **config.yaml** override
**dataclass defaults** in `core/config.py`.

### Environment variables (`.env`)

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | No | `auto` \| `openai` \| `anthropic` \| `groq` \| `gemini` \| `together` \| `openrouter` \| `ollama` |
| `OPENAI_API_KEY` | One of these | LLM provider key |
| `ANTHROPIC_API_KEY` | One of these | |
| `GROQ_API_KEY` | One of these | Free tier: console.groq.com |
| `GOOGLE_API_KEY` | One of these | |
| `TOGETHER_API_KEY` | One of these | |
| `OPENROUTER_API_KEY` | One of these | |
| `FMP_API_KEY` | No | Financial Modeling Prep (financial data) |
| `ALPHA_VANTAGE_KEY` | No | Alpha Vantage (market data fallback) |
| `POLYGON_API_KEY` | No | Polygon.io (market data fallback) |
| `TAVILY_API_KEY` | No | Tavily web search (RAG tool) |
| `ER_REPORTS_DIR` | No | Override default report output path |
| `LOG_LEVEL` | No | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` (default: `INFO`) |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` |
| `OLLAMA_MODEL` | No | `qwen2.5:7b` |

### config.yaml keys

```yaml
llm:
  temperature: 0.1       # LLM sampling temperature
  max_tokens: 4096       # Max tokens per LLM call
  timeout: 300           # Seconds before LLM call times out
  max_retries: 3

modeling:
  min_history_years: 5   # Minimum financial history required
  forecast_years: 5      # DCF projection horizon
  terminal_growth_rate_default: 3.0   # % pa
  wacc_floor_pct: 6.0
  wacc_ceiling_pct: 20.0
  peer_group_size: 5     # Number of peers for relative valuation

report:
  min_word_count: 15000
  target_word_count: 25000
  firm_name: "Equity Intelligence Research"

forensics:
  beneish_manipulation_threshold: -1.78   # Score > threshold → manipulation risk
  altman_em_safe: 2.60                    # Z > 2.60 → safe zone
  piotroski_strong: 7                     # F ≥ 7 → strong
```

### Output paths

| Platform | Default path |
|---|---|
| Linux / macOS | `~/equity_research_reports/` |
| Windows | `%USERPROFILE%\equity_research_reports\` |
| Google Colab | `/content/equity_research_reports/` |
| Kaggle | `/kaggle/working/equity_research_reports/` |
| Custom | Set `ER_REPORTS_DIR` in `.env` |

---

## 5. LLM Provider Cascade

`validate_llm_config()` in `core/config.py` is called at startup and before every
RAG pipeline run. It returns the active provider name or raises `RuntimeError` with
setup instructions.

**Provider selection logic:**

```
if LLM_PROVIDER != "auto":
    if provider == "ollama" → return "ollama"   # no key needed
    if key is set          → return provider
    else                   → raise RuntimeError

if LLM_PROVIDER == "auto":
    check groq → openai → anthropic → together → openrouter → gemini
    first provider with a non-empty key wins
    fallback: "ollama" (always last, no key required)
```

**Default models per provider:**

| Provider | Primary model | Fast model |
|---|---|---|
| OpenAI | `gpt-4o` | `gpt-4o-mini` |
| Anthropic | `claude-opus-4-8` | `claude-haiku-4-5-20251001` |
| Groq | `llama-3.3-70b-versatile` | `llama-3.1-8b-instant` |
| Gemini | `gemini-1.5-pro` | `gemini-2.0-flash` |
| Together | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | `meta-llama/Llama-3.1-8B-Instruct-Turbo` |
| OpenRouter | `anthropic/claude-opus-4-8` | `google/gemini-flash-1.5` |
| Ollama | `qwen2.5:7b` | `phi3.5:3.8b` |

Override model names in `config.yaml` under `llm.primary_model` / `llm.fast_model`.

---

## 6. 20-Agent CIO Pipeline

Research follows an 11-step mandatory sequence encoded in `core/research_philosophy.py`:
macro → industry → business → management → financial → risks → accounting → governance → forecast → valuation → thesis.
The orchestrator acts as CIO: it coordinates agents and aggregates risk scores; it does not analyze directly.

### Orchestration phases

```
Phase A  (sequential)
  01 CompanyProfilingAgent      — ticker, sector, exchange, CIK, Bloomberg ID
  02 FilingRetrievalAgent       — SEC EDGAR 10-K/10-Q/8-K; BSE/NSE for Indian cos.

Phase B  (parallel, 4 agents)
  03 FinancialExtractionAgent   — P&L, balance sheet, cash flow (5 yrs, GAAP/IFRS)
  04 MarketDataAgent            — live price, market cap, P/E, 52-wk range (yfinance)
  12 TranscriptRetrievalAgent   — last 4 earnings call transcripts
  13 HistoricalDataAgent        — 7-year daily OHLCV, beta, correlations

Phase C  (parallel, 7 agents)
  05 AccountingQualityAgent     — revenue recognition, accrual ratio, DSRI, GMI
  06 ForensicAccountingAgent    — Beneish M-score, Altman Z, Piotroski F;
                                   9 frameworks + 10-case fraud learning corpus
  09 RiskAnalysisAgent          — macro/credit/operational risk matrix
  14 EarningsQualityAgent       — EPS beat/miss streaks, guidance revision bias
  17 IndustryIntelligenceAgent  — Porter Five Forces, TAM, attractiveness score
  18 ManagementGovernanceAgent  — governance/credibility/capital-allocation scores,
                                   board independence, promoter pledging, RPT analysis
  19 ESGSustainabilityAgent     — BRSR (India) + ISSB S1+S2 / SASB / GRI / TCFD (Global)

Phase D  (sequential)
  07 FinancialModelingAgent     — 5-yr P&L, FCF, revenue/margin drivers
  08 ValuationAgent             — DCF + EV/EBITDA comps + P/E band → target price
  15 ScenarioAnalysisAgent      — bull/base/bear outcomes with probability weights

Phase E  (sequential)
  10 NarrativeGenerationAgent   — ThesisComponent (variant perception + Bull/Base/Bear
                                   scenarios) + ~25 000-word structured report
  11 ComplianceValidationAgent  — 17 checks across Indian + Global regulatory standards

Phase F
  16 ReportGenerationAgent      — DOCX assembly, charts, executive summary, cover page
```

### Risk weight matrix (Phase C → `overall_risk_score`)

| Agent | Weight |
|---|---|
| ForensicAccounting | 0.18 |
| AccountingQuality | 0.13 |
| RiskAnalysis | 0.13 |
| EarningsQuality | 0.10 |
| ManagementGovernance | 0.10 |
| Valuation | 0.09 |
| IndustryIntelligence | 0.08 |
| FinancialExtraction | 0.07 |
| ESGSustainability | 0.05 |
| FinancialModeling | 0.04 |
| Compliance | 0.03 |
| **Total** | **1.00** |

### Agent contract (`base_agent.py`)

Every agent inherits `BaseAgent` and must:
1. Set class attributes `AGENT_ID: str` and `AGENT_NAME: str`
2. Implement `run(state: ResearchState) -> AgentOutput`
3. Return only typed `AgentOutput` — no side-channel data
4. Never raise exceptions from `run()` — return `AgentOutput(status=FAILED, error=...)`

`BaseAgent.execute()` wraps `run()` with:
- Timing and Loguru structured logging
- Audit trail events (`audit_trail.py`)
- Automatic retry (3 attempts, 2-second backoff)
- Database run-status updates (`database.py`)

**Shared helpers available to every agent:**

| Helper | Signature | Purpose |
|---|---|---|
| `make_finding` | `(type, title, detail, evidence, risk_level, confidence, …)` | Create a typed `Finding` |
| `red_flag` | `(title, detail, evidence, risk_level, **kwargs)` | Shorthand for `FindingType.RED_FLAG` |
| `green_flag` | `(title, detail, evidence, confidence, **kwargs)` | Shorthand for `FindingType.GREEN_FLAG` |
| `llm_analyze` | `(system_prompt, user_prompt, max_tokens, json_mode)` | LLM call via `LLMManager` |
| `rag_query` | `(question, state, top_k)` | Per-ticker RAG retrieval |
| `get_financial_series` | `(state, field)` | Multi-year time series for one metric |
| `_latest_fin` | `(financial_history)` | Flatten most-recent year into a flat dict |

### Shared state (`orchestrator/state.py`)

`ResearchState` is passed to every agent. Key fields:

| Field | Type | Set by |
|---|---|---|
| `run_id` | str | Orchestrator |
| `company_name`, `ticker` | str | ProfilingAgent |
| `company_profile` | dict | ProfilingAgent |
| `financial_history` | FinancialHistory | ExtractionAgent |
| `valuation_summary` | ValuationSummary | ValuationAgent |
| `target_price` | float | ValuationAgent |
| `investment_rating` | str | Orchestrator (derived from valuation) |
| `overall_risk_score` | float | RiskAnalysisAgent |
| `report_sections` | dict[str, str] | NarrativeAgent |
| `agent_outputs` | dict[str, AgentOutput] | All agents |
| `validation_results` | list | ComplianceAgent |
| `critical_findings` | int | Forensic + Risk agents |
| `thesis` | `Optional[ThesisComponent]` | NarrativeAgent |

`ThesisComponent` (frozen Pydantic model in `models/research.py`) holds:
- `variant_perception` — one-sentence differentiated view vs. consensus
- `consensus_view`, `our_view`, `why_consensus_is_wrong`
- `catalysts: list[str]`, `key_risks: list[str]`
- `bull_case`, `base_case`, `bear_case` — each a `ThesisCase` with `scenario`, `narrative`,
  `target_price`, `return_potential_pct`, `key_assumptions`, `probability ∈ [0, 1]`

### Research philosophy (`core/research_philosophy.py`)

Single authoritative module imported by every agent. Contains:

| Symbol | Purpose |
|---|---|
| `RESEARCH_SEQUENCE` | Ordered list of 11 mandatory research steps |
| `PHILOSOPHY_RULE` | Plain-English CIO rule set (objectivity, evidence, independence) |
| `CIO_ROLE` | CIO mandate: coordinate, never analyse directly |
| `AGENT_SPECS` | Dict keyed by agent id → `{name, role, responsibilities, deliverables, frameworks}` |
| `RAG_DOCUMENT_TAG_FIELDS` | 9 required metadata fields for every ingested document |
| `SOURCE_PRIORITY` | Ordered source preference chain (primary → secondary → tertiary) |
| `EVIDENCE_RULES` | `EvidenceLevel` thresholds (HIGH ≥ 3 sources, MEDIUM = 2, LOW = 1) |
| `REPORT_SECTIONS_20` | Ordered 20-section tuples used to structure the DOCX output |

### Compliance framework (`agents/compliance_agent.py`)

17 checks across two jurisdictions. `is_indian` / `is_us` country detection gates jurisdiction-specific checks.

**Indian standards (7 checks):**
- SEBI Research Analyst Regulations 2014 — applies universally (governs the research output)
- SEBI LODR (Listing Obligations) — Indian-listed companies only
- Companies Act 2013 (Section 149) — board composition ≥ 33% independent
- Ind AS convergence with IFRS — Indian-listed companies only
- RBI Prudential Guidelines — Banking sector
- IRDAI Regulations — Insurance sector
- AMFI Framework — Mutual fund sector

**Global standards (10 checks):**
- IFRS / IAS accounting standards
- US GAAP — exact set lookup (`{"united states", "usa", "us", "united states of america"}`)
- IOSCO Principles for Financial Benchmarks
- CFA Institute Research Objectivity Standards
- OECD Corporate Governance Principles (governance score sentinel-checked; 0 ≠ 50)
- ISSB S1 (general sustainability disclosures) — WARN placeholder if ESG agent absent
- ISSB S2 (climate-related disclosures) — WARN placeholder if ESG agent absent
- SASB Standards — WARN placeholder if ESG agent absent
- GRI Standards — WARN placeholder if ESG agent absent
- UN PRI alignment

WARN placeholders ensure the compliance score denominator stays consistent even when the ESG agent fails.

---

## 7. RAG Subsystem

The RAG subsystem is independent of the 17-agent pipeline. It answers analyst
questions grounded on ingested company documents. The main pipeline uses it inside
`retriever` nodes; it can also be called standalone via the API or directly.

### 7.1 Vector store (`retrieval/vector_store.py`)

**Storage:** One ChromaDB `PersistentClient` per ticker, stored at
`data/chroma_db/<TICKER>/`. Collections are named `er_<ticker_lowercase>`.

**Embedding model:** `BAAI/bge-large-en-v1.5` (1024-dimensional, loaded via
LlamaIndex `HuggingFaceEmbedding`). Downloaded automatically on first use (~1.4 GB).
Requires `sentence-transformers` or `llama-index-embeddings-huggingface`.

**Chunking:** `SentenceSplitter`, chunk_size=512 tokens, overlap=100 tokens.

**Retrieval pipeline (per query):**

```
query(question, ticker, top_k=5, metadata_filter=None)

1. Cache check          — (question, ticker, top_k, filter) → list[str], TTL=5 min
2. Dense retrieval      — VectorIndexRetriever(top_k × 4 candidates, min 20)
                          gated by threading.Semaphore(min(cpu_count, 8)) to limit
                          concurrent BGE encoding on CPU
3. Metadata filtering   — post-retrieval filter by any metadata field
4. BM25 keyword scoring — TF (k1=1.5, b=0.75) per query term
5. Reciprocal Rank Fusion — merge dense rank + keyword rank (RRF constant=60)
6. Cross-encoder rerank — if sentence-transformers installed; runs when
                          len(candidates) > top_k (skipped on small corpora to
                          avoid ms-marco score inversion on financial text)
7. Return top-k + write cache
```

**Cache eviction:** Up to 500 entries stored; at 500+, all entries older than
TTL=300s are evicted. Cache is in-process (not shared across workers).

**Public API:**

```python
from equity_research.retrieval.vector_store import (
    ingest_document,   # (text, metadata, ticker) → int (chunks added)
    ingest_texts,      # (texts, metadatas, ticker) → int
    query,             # (question, ticker, top_k, metadata_filter) → list[str]
    collection_size,   # (ticker) → int
    clear_company,     # (ticker) → None
)
```

### 7.2 RAG pipeline (`retrieval/rag_pipeline.py`)

A 7-node LangGraph `StateGraph`. Requires `langchain-core` and `langgraph` to
execute; can be imported without them (graceful stubs) for testing.

**Nodes:**

| # | Node | Responsibility |
|---|---|---|
| 1 | `query_rewriter` | Normalise query; on retry loops, SIMPLIFY not expand |
| 2 | `query_decomposer` | Detect multi-hop; split compound questions into sub-queries |
| 3 | `detail_checker` | Router: needs retrieval? or parametric LLM knowledge? |
| 4 | `source_selector` | Pick: `vector_db` / `internet` / `tools_apis` / `combined` |
| 5 | `retriever` | Parallel fetch from selected sources; dedup; priority sort |
| 6 | `response_generator` | Synthesise grounded answer from retrieved context |
| 7 | `relevance_checker` | Score answer vs. context (not self-grading); loop if low |

Max retry loops: 5. On the 5th loop, `query_rewriter` simplifies the query to
break out of low-relevance cycles.

**Entry point:**

```python
from equity_research.retrieval.rag_pipeline import run

result = run(
    question="What was APEX's FY2023 free cash flow?",
    company_name="APEX Technologies",
    ticker="APEX",
)
# result keys: final_response, sources_used, relevance_score, ...
```

**`run()` validates LLM config before building the graph.** If no API key is set,
it raises `RuntimeError` with setup instructions.

### 7.3 LangChain tools (`retrieval/tools.py`)

| Tool | Function | Notes |
|---|---|---|
| `web_search` | Tavily (preferred) or DuckDuckGo fallback | Requires `TAVILY_API_KEY` for Tavily |
| `sec_edgar_search` | SEC EDGAR full-text search scoped to ticker | Uses `dateRange`, `startdt` params |
| `financial_snapshot` | yfinance price / market-cap / P/E | No API key needed |
| `wikipedia_lookup` | Wikipedia article summary | Gated: only for background/overview queries |
| `calculator` | AST-safe arithmetic evaluator | Blocks `eval`, lambdas, power bombs |

**Calculator security (`_safe_eval`):**
- Allowed nodes: `Num`, `BinOp` (+−×÷^), `UnaryOp`, `Call` (math functions only)
- Blocks: string operands (`non-numeric operands not allowed`), exponents > 10,000
  (prevents `9**387420489` DoS)
- Blocked functions: all except `abs`, `round`, `min`, `max`, `sum`, `math.*`

**Wikipedia gating (`_is_background_query`):**
Frozenset check on 30+ keywords (`history`, `founded`, `overview`, `sector`,
`headquarters`, etc.). Wikipedia is only called for clearly background queries —
not for specific financial figures where it would hallucinate.

**Graceful import:** Both `tools.py` and `rag_pipeline.py` wrap their
`langchain_core` imports in `try/except`. If the package is absent, a no-op stub
replaces the `@tool` decorator so the modules can be imported and tested without
LLM dependencies installed.

---

## 8. FastAPI HTTP Server

`api/server.py` — production REST API over the RAG subsystem.

### Start

```bash
# Development
uvicorn equity_research.api.server:app --host 0.0.0.0 --port 8000

# Production (2 workers, auto-restart)
uvicorn equity_research.api.server:app --host 0.0.0.0 --port 8000 --workers 2
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness + readiness check — LLM, embedding model, LangChain |
| `POST` | `/query` | Run RAG pipeline; returns answer, sources, relevance score, latency |
| `POST` | `/ingest` | Ingest a document into a ticker's knowledge base |
| `GET` | `/collection/{ticker}` | Chunk count for a ticker |
| `DELETE` | `/collection/{ticker}` | Drop a ticker's knowledge base (irreversible) |

### Request / response shapes

**POST /query**
```json
// Request
{ "question": "What was APEX's FY2023 revenue?", "ticker": "APEX", "company_name": "APEX Technologies" }

// Response
{
  "question": "...", "ticker": "APEX",
  "answer": "APEX Technologies reported total revenue of $18.42 billion...",
  "sources_used": ["APEX_10K_FY23_REV", ...],
  "relevance_score": 0.94,
  "latency_ms": 312.4
}
```

**POST /ingest**
```json
// Request
{ "ticker": "APEX", "text": "Full document text...", "metadata": { "doc_type": "10-K", "fiscal_year": "2023" } }

// Response
{ "ticker": "APEX", "chunks_added": 42, "total_chunks": 287 }
```

**GET /health**
```json
{
  "status": "ok",
  "checks": {
    "llm": "ok (groq)",
    "embedding_model": "loaded",
    "langchain": "ok"
  }
}
```
Returns HTTP 503 if any check is in error state.

### Rate limiting

Token-bucket, in-memory, per source IP:
- Sustained rate: **10 requests/second**
- Burst allowance: **20 requests**
- Only applied to `POST` requests
- Returns HTTP 429 when bucket is empty

### Startup validation

`@app.on_event("startup")` calls `validate_llm_config()` and `_ensure_settings()`
(loads BGE-large). Both failures are logged but do not crash the process — instead,
`/health` returns 503 so orchestration systems can detect the misconfiguration.

---

## 9. Forensic Scoring Models

All three models are in `forensics/` and called by `ForensicAccountingAgent`.

### Beneish M-score (`forensics/beneish.py`)

8-variable earnings-manipulation detector. Needs 2 consecutive years of financial data.

| Variable | Meaning |
|---|---|
| DSRI | Days Sales Receivables Index |
| GMI | Gross Margin Index |
| AQI | Asset Quality Index |
| SGI | Sales Growth Index |
| DEPI | Depreciation Index |
| SGAI | SG&A Index |
| TATA | Total Accruals to Total Assets |
| LVGI | Leverage Index |

**Thresholds:**
- M > −1.78 → manipulation risk (flag)
- M > −1.00 → high manipulation risk (critical flag)

### Altman Z-score (`forensics/altman.py`)

Emerging-market model (EM Z-score) — used for all companies including US.

```
Z' = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4

X1 = Working capital / Total assets
X2 = Retained earnings / Total assets
X3 = EBIT / Total assets
X4 = Book value equity / Total liabilities
```

**Thresholds:**
- Z > 2.60 → safe zone
- 1.10 ≤ Z ≤ 2.60 → grey zone
- Z < 1.10 → distress zone (critical flag)

### Piotroski F-score (`forensics/piotroski.py`)

9-binary-signal financial strength score.

| Signal | Category |
|---|---|
| ROA > 0 | Profitability |
| CFO > 0 | Profitability |
| ΔROA > 0 | Profitability |
| CFO > ROA | Profitability |
| ΔLeverage < 0 | Leverage |
| ΔLiquidity > 0 | Leverage |
| No new shares | Leverage |
| ΔGross margin > 0 | Operating efficiency |
| ΔAsset turnover > 0 | Operating efficiency |

**Thresholds:**
- F ≥ 7 → strong (≥ 7 signals positive)
- F ≤ 2 → weak (potential short candidate)

---

## 10. Storage Layer

### Per-run directory structure

```
~/equity_research_reports/<Company>_<RunID>/
├── filings/            Raw filing downloads (10-K PDFs, etc.)
├── transcripts/        Earnings call transcripts
├── data/               Extracted financial data (Parquet)
├── models/             Financial model outputs (Excel / CSV)
├── report/
│   └── <Company>_Research_Report.docx
├── audit/
│   └── audit_trail.jsonl   Agent event log
└── run_metadata.json       Run summary
```

### SQLite run registry (`storage/database.py`)

`data/equity_research.db` — stores:
- Run metadata (company, start/end time, status, output path)
- Agent completion status per run
- Risk flag counts

### DuckDB analytics (`data/analytics.duckdb`)

Not populated by default. Intended for cross-run analysis (financial trend queries
across multiple research runs).

### Audit trail (`storage/audit_trail.py`)

JSONL file appended in real-time. Each line is an event:
```json
{"ts": "2026-06-17T00:12:09Z", "type": "agent_start", "agent_id": "01_profiling", ...}
{"ts": "2026-06-17T00:12:11Z", "type": "agent_end", "agent_id": "01_profiling", "elapsed_s": 2.1, ...}
```

---

## 11. Running the Platform

### Single company (CLI)

```bash
python main.py "Apple"
python main.py "HDFC Bank" --ticker HDFCBANK
python main.py "Infosys" --output /tmp/reports --verbose
```

### Batch (CLI)

```bash
# companies.txt — one company per line, # for comments
python main.py --batch companies.txt
```

### RAG query (Python)

```python
from equity_research.retrieval.vector_store import ingest_document, query

# Ingest a filing
ingest_document(filing_text, {"doc_type": "10-K", "fiscal_year": "2023"}, ticker="APEX")

# Query
chunks = query("What was APEX's gross margin?", ticker="APEX", top_k=5)
```

### RAG pipeline (Python)

```python
from equity_research.retrieval.rag_pipeline import run
result = run("What is APEX's FY2024 revenue guidance?", company_name="APEX", ticker="APEX")
print(result["final_response"])
```

### API server

```bash
uvicorn equity_research.api.server:app --host 0.0.0.0 --port 8000 --workers 2

# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is APEX revenue?","ticker":"APEX","company_name":"APEX"}'
```

### Jupyter / Colab

`quickstart.ipynb` in the repo root walks through the full flow interactively.

---

## 12. Testing & Evaluation

### Stress test suite (`tests/stress_test.py`)

48 infrastructure tests covering:
- Config loading, API key resolution
- LLM provider validation
- Vector store CRUD (ingest, query, clear, metadata filter)
- Concurrent retrieval (thread safety of singletons)
- Cache correctness
- Orchestrator phase ordering
- Storage manager file layout
- Audit trail event sequence

```bash
python -m pytest tests/stress_test.py -v
```

### RAG backtest (`tests/rag_backtest.py`)

End-to-end evaluation against a synthetic ground-truth corpus (5 tickers, 11 docs,
45 retrieval queries, 32 security tests).

```bash
python -m equity_research.tests.rag_backtest
# Output: tests/rag_eval_report.json  +  console dashboard
```

**Scoring breakdown (92.7/100 overall):**

| Component | Weight | Score | Method |
|---|---|---|---|
| Retrieval | 30% | ~80 | P@5, Recall@5, MRR, NDCG@5, HR, CR |
| Generation | 25% | ~97 | Grounding score, hallucination rate, key facts |
| Security | 15% | 100 | 32 tests — injection, fuzzing, DoS, gating |
| Latency | 15% | ~100 | Empirical P95 vs target (<500ms at 1 user) |
| Scalability | 15% | 96 | Empirical P95 vs target (<1500ms at 5 users, <3000ms at 10) |

**Security test categories:**

| Category | Count | Tests |
|---|---|---|
| JSON fuzzing (`_parse_json`) | 10 | Balanced braces, null bytes, unicode bomb, SQL injection, exponential depth |
| Calculator injection | 9 | `__import__`, `eval`, lambda, list comprehension, power bomb, string concat |
| Prompt injection | 5 | Direct, role override, context poisoning, delimiter attack, RTL unicode |
| Wikipedia gating | 6 | Financial vs. background queries |
| SEC EDGAR params | 1 | Date-scoped parameters enforced |
| Large payload DoS | 1 | 140 KB document → chunked in <2s |

### Latency benchmarks (from last backtest run)

| Scenario | P50 | P95 | P99 | RPS | Fail% |
|---|---|---|---|---|---|
| 1 concurrent user | 322 ms | 314 ms | 314 ms | 4 | 0% |
| 5 concurrent users | 971 ms | 1309 ms | 1309 ms | 6 | 0% |
| 10 concurrent users | 1660 ms | 3486 ms | 3486 ms | 11 | 0% |

(Retrieval-only; excludes LLM generation time.)

---

## 13. Performance Benchmarks

### Embedding throughput

| Model | Batch=1 | Batch=32 | Notes |
|---|---|---|---|
| `BAAI/bge-large-en-v1.5` | ~310 ms | ~310 ms | CPU-bound, GIL-serialised |
| `all-MiniLM-L6-v2` (fallback) | ~40 ms | ~15 ms | Faster, lower quality |

### Concurrency model

- Embedding semaphore: `threading.Semaphore(min(cpu_count, 8))`
- 5-minute query result cache prevents redundant encode+retrieve round-trips
- Cache shows hit rate of ~40% in repeated-question workloads (seen in backtest logs)
- P95 degrades at 10+ concurrent users on CPU; use GPU or reduce concurrency on
  CPU-only deployments by lowering `_EMBED_CONCURRENCY`

### Full research run duration (approximate)

| Config | Time |
|---|---|
| Groq / Together (fast LLMs, API) | 8–15 min |
| OpenAI GPT-4o | 15–25 min |
| Ollama qwen2.5:7b (local CPU) | 45–90 min |

---

## 14. Known Limitations

### Retrieval Precision@5 (0.243 in backtest)

P@5 appears low but is a **small-corpus artifact**. The backtest uses 11 synthetic
documents. When top_k=5 and a ticker has only 5 chunks, all 5 are returned regardless
of relevance — the denominator is always 5. In production (10,000+ chunks per ticker),
the cross-encoder reranker filters the candidate pool to genuinely relevant chunks,
and P@5 is expected above 0.5.

MRR=0.976 and Recall@5=1.000 confirm the relevant chunk is always retrieved and
ranked first.

### Cross-encoder gate

The CE reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) only runs when
`len(candidates) > top_k`. For small corpora where candidates == top_k, it is
intentionally skipped because the ms-marco model (trained on web-search passages)
produced score inversions on dense financial text, hurting MRR. Enable by changing
the gate to `>= 2` and test on your production corpus before deploying.

### Cache is not shared across uvicorn workers

The `_query_cache` dict is in-process. With `--workers 2`, each worker has its own
cache. Use Redis + LangChain cache middleware for shared cache in multi-worker
deployments.

### Windows CRLF line endings

The repo is developed on Windows. Git is configured with `autocrlf`. Linux
deployment should not be affected as Python is line-ending agnostic, but diff
output may show CRLF substitutions.

### No PDF ingestion in RAG by default

The vector store accepts raw text. The caller (filing_retrieval agent or the `/ingest`
API endpoint) is responsible for converting PDFs to text before calling `ingest_document`.
Use `PyMuPDF` (`fitz.open(path).get_text("text")`) for extraction.

### Rate limits

- SEC EDGAR: 0.5 req/s enforced in `AcquisitionConfig.rate_limit_rps`
- Groq free tier: 30 req/min (the platform will hit this on Phase B/C parallel runs)
- API server rate limit: 10 req/s per IP (in-memory, resets on restart)

---

## 15. Deployment Checklist

### Minimum requirements

- [ ] Python 3.10+, 8 GB RAM
- [ ] At least one LLM API key set in `.env`
- [ ] `pip install -r requirements.txt` completed without errors
- [ ] `python main.py --check` shows all `✓`
- [ ] `python -m equity_research.tests.rag_backtest` scores ≥ 90/100
- [ ] `GET /health` returns `{"status": "ok", ...}`

### Recommended for production

- [ ] Set `LLM_PROVIDER` explicitly (avoid `auto` cascade in production)
- [ ] Set `ER_REPORTS_DIR` to a persistent mount (not default home)
- [ ] Configure log rotation: `LOG_LEVEL=INFO`, loguru rotation=50 MB
- [ ] Use `--workers 2` minimum for uvicorn (more workers → no cache sharing,
      consider Redis if cache efficiency matters)
- [ ] Pin `requirements.txt` versions with `pip freeze > requirements.lock`
- [ ] Add HF_TOKEN to `.env` to eliminate HuggingFace rate-limit warnings
  ```
  HF_TOKEN=your_hf_token_here
  ```
- [ ] Pre-warm the BGE-large model by calling `_ensure_settings()` at startup
  rather than on first query

### Security notes

- The `/collection/{ticker}` `DELETE` endpoint has no auth guard. Add an
  API-key header check before exposing publicly.
- Rate limiting is in-process. Put NGINX or a WAF in front for production.
- The `calculator` tool is AST-safe but the `web_search` tool executes live web
  requests. Ensure `TAVILY_API_KEY` is sandboxed in your secrets manager.

---

## 16. Extension Guide

### Adding a new agent

1. Add an entry to `AGENT_SPECS` in `core/research_philosophy.py`:
   ```python
   "20_my_agent": {
       "name": "My Custom Agent",
       "role": "...",
       "responsibilities": [...],
       "deliverables": [...],
   }
   ```

2. Create `agents/my_agent.py`:
   ```python
   from .base_agent import BaseAgent
   from ..models.research import AgentOutput, AgentStatus, ResearchState
   from ..core.research_philosophy import AGENT_SPECS

   _SPEC = AGENT_SPECS["20_my_agent"]

   class MyAgent(BaseAgent):
       AGENT_ID   = "20_my_agent"
       AGENT_NAME = "My Custom Agent"

       def run(self, state: ResearchState) -> AgentOutput:
           # ... your logic ...
           return AgentOutput(
               agent_id=self.AGENT_ID,
               agent_name=self.AGENT_NAME,
               status=AgentStatus.COMPLETED,
               payload={"result": ...},
           )
   ```

3. Import and add to the appropriate phase in `orchestrator/workflow.py`.

4. Add `AGENT_ID` to the phase list. For parallel phases (B or C), append
   to `phase_b_agents` or `phase_c_agents`; for sequential phases (D, E),
   call `make(MyAgent).execute(state)` in order.

5. If the agent produces output consumed by `ComplianceAgent`, add its id
   to `_validate_agent_completion`'s `required` set in `compliance_agent.py`.

### Adding a new LLM provider

1. Add the provider key to `PROVIDER_MODELS` in `core/config.py`
2. Add its API key env var (e.g. `MY_PROVIDER_API_KEY`)
3. Add key to the `_keys` dict in `validate_llm_config()`
4. Implement the provider in `core/llm_manager.py` following the existing
   pattern (instantiate the LangChain chat model, store as `self._llm`)

### Adding a new RAG tool

1. Add a new `@tool`-decorated function in `retrieval/tools.py`
2. Wire it into the `source_selector` node in `rag_pipeline.py` under the
   appropriate source type (`tools_apis` or `internet`)
3. Add a security test in `tests/rag_backtest.py::run_security_tests()`

### Changing the embedding model

Edit `core/config.py`:
```python
@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-large-en-v1.5"   # ← change this
    chunk_size: int = 512
    chunk_overlap: int = 100
```

Clear existing ChromaDB collections after changing the model — embeddings
from different models are incompatible:
```python
from equity_research.retrieval.vector_store import clear_company
clear_company("TICKER")
```

---

*Document updated: 2026-06-17. Maintained in `docs/TECHNICAL_REFERENCE.md`.*
*Current commit: `3fa8c0b`. Platform repo: https://github.com/anubhav0499-bit/equity-research*
