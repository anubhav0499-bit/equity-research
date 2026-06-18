# Equity Intelligence Research Platform

Autonomous institutional-grade equity research system. Given a company name or ticker, it produces a fully sourced, compliance-validated DOCX research report without human intervention.

## Architecture

20 agents run in a CIO-orchestrated phased pipeline. The research philosophy — mandatory 11-step sequence, agent specifications, RAG document tagging schema, and evidence rules — is encoded in `core/research_philosophy.py` and imported by all agents.

```
Phase A  Company Profiling → Filing Retrieval
Phase B  Financial Extraction + Market Data + Transcript Retrieval + Historical Data       (parallel)
Phase C  Accounting Quality + Earnings Quality + Forensic Accounting + Risk Analysis
         + Industry Intelligence + Management & Governance + ESG & Sustainability          (parallel)
Phase D  Financial Modeling → Valuation → Scenario Analysis  (sequential)
Phase E  Narrative Generation (+ Thesis Construction) → Compliance Validation  (sequential)
Phase F  Report Generation
```

Each agent returns a typed `AgentOutput` (Pydantic). No free-form text passes between agents. Full audit trail is written to JSONL.

### Risk weight matrix (Phase C → overall risk score)

Weights are aligned with CFA Institute, OECD Corporate Governance Principles, and MSCI Quality Factor research.

| Agent | Weight | Primary rationale |
|-------|--------|-------------------|
| Management & Governance | 20% | Strongest predictor of long-run risk-adjusted return (OECD, Fama-French quality factor) |
| Forensic Accounting | 15% | Beneish/Altman/Piotroski: best early-warning for catastrophic downside |
| Risk Analysis | 13% | Macro / credit / operational risk anchor |
| Industry Intelligence | 12% | Porter Five Forces + moat: structural driver of cash-flow durability |
| Accounting Quality | 10% | Accrual ratio, revenue recognition; MSCI quality factor component |
| Earnings Quality | 8% | Guidance accuracy, beat/miss patterns; predictability premium |
| Valuation | 8% | Valuation risk (secondary to business quality signals) |
| ESG & Sustainability | 7% | ISSB S1+S2 / BRSR: material for energy, materials, utilities |
| Financial Extraction | 4% | Data completeness check; not a fundamental risk driver |
| Compliance Validation | 2% | Largely binary pass/fail; captured upstream by governance weight |
| Financial Modeling | 1% | 5-yr model output — analyst assumption, not independent risk signal |

## Forensic Checks

| Score | Method | Threshold |
|-------|--------|-----------|
| Beneish M-Score | 8 variables — DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA | < −1.78 = unlikely manipulator |
| Piotroski F-Score | 9 signals across profitability, leverage, efficiency | ≥ 7 = strong, ≤ 2 = weak |
| Altman Z-Score | EM variant for non-US companies | > 2.60 = safe zone |
| Sloan Accrual | Operating accruals / net operating assets | < 10% preferred |
| Cash Conversion | OCF / Net Income | > 0.8× preferred |

## Valuation

- **Single WACC** computed via CAPM with country risk premium (Damodaran 2024 data), shared across all methods and scenarios.
- **DCF** — FCFF approach, Gordon Growth terminal value.
- **Relative** — peer median EV/EBITDA and P/E with scenario discounts.
- **SOTP** — segment-level EV/EBITDA with minority interest and conglomerate discount.
- **Scenarios** — Bear / Base / Bull with probability-weighted target price.

## Quick Start

### Requirements
Python 3.10+ · One LLM API key (Groq free tier works)

```bash
# 1. Clone
git clone https://github.com/anubhav0499-bit/equity-research.git
cd equity-research

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — add at least one LLM key (GROQ_API_KEY recommended)

# 4. Run
python main.py "Apple"
python main.py "HDFC Bank" --ticker HDFCBANK.NS
python main.py "Infosys" "TCS" --output /path/to/reports
```

### Check dependencies
```bash
python main.py --check
```

### Batch mode
```bash
# companies.txt — one company per line, # for comments
python main.py --batch companies.txt --output ./reports
```

### RAG query API

Ask analyst questions grounded on ingested filings — independently of the research pipeline:

```bash
# Start the API server
uvicorn equity_research.api.server:app --host 0.0.0.0 --port 8000

# Ingest a document
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","text":"<filing text>","metadata":{"doc_type":"10-K"}}'

# Synchronous query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What was Apple FY2024 revenue?","ticker":"AAPL","session_id":"s1"}'

# Streaming query (Server-Sent Events)
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"Explain Apple gross margin trend","ticker":"AAPL","session_id":"s1"}'
```

## Notebook

`quickstart.ipynb` runs the full pipeline in any Jupyter environment:
- Local Jupyter Lab / VS Code
- Google Colab
- Kaggle
- AWS SageMaker
- Azure ML

## Configuration

All defaults live in `config.yaml`. Override via environment variables with `ER_` prefix:

```bash
export ER_REPORTS_DIR=/my/reports        # output directory
export ER_LLM__MAX_TOKENS=2048          # reduce for speed
export LLM_PROVIDER=groq                 # force specific provider
```

### Supported LLM Providers

Auto-detected in this order: **Groq → OpenAI → Anthropic → Gemini → Together → OpenRouter → Ollama → template**

| Provider | Env Var | Free Tier |
|----------|---------|-----------|
| Groq | `GROQ_API_KEY` | Yes — [console.groq.com](https://console.groq.com) |
| OpenAI | `OPENAI_API_KEY` | No |
| Anthropic | `ANTHROPIC_API_KEY` | No |
| Google Gemini | `GOOGLE_API_KEY` | Limited |
| Together AI | `TOGETHER_API_KEY` | Limited |
| OpenRouter | `OPENROUTER_API_KEY` | Pay-per-token |
| Ollama | none needed | Yes — local |

## Ticker Formats

| Exchange | Format | Example |
|----------|--------|---------|
| NYSE / NASDAQ | plain | `AAPL`, `MSFT` |
| NSE (India) | `.NS` suffix | `TCS.NS`, `INFY.NS` |
| BSE (India) | `.BO` suffix | `500325.BO` |
| LSE (UK) | `.L` | `AZN.L` |
| TSX (Canada) | `.TO` | `RY.TO` |
| ASX (Australia) | `.AX` | `CBA.AX` |
| HKEX | `.HK` | `0700.HK` |
| TSE (Japan) | `.T` | `7203.T` |
| KRX (Korea) | `.KS` | `005930.KS` |

## Output Structure

Each run creates:
```
~/equity_research_reports/<COMPANY>/<run_id>/
  Company_Profile/
  Raw_Filings/
  Parsed_Data/Text/ and Tables/
  Financial_Statements/
  Agent_Outputs/          ← one JSON per agent
  Forecasts/
  Valuation/
  Reports/                ← DOCX report
  Audit_Trail/            ← JSONL audit log + validation report
```

## Project Layout

```
equity_research/
├── agents/               # 20 agent implementations
│   ├── base_agent.py             # abstract base — retry, audit, _latest_fin helper
│   ├── company_profiling.py      # 01
│   ├── filing_retrieval.py       # 02
│   ├── financial_extraction.py   # 03
│   ├── market_data.py            # 04
│   ├── accounting_quality.py     # 05
│   ├── forensic_accounting.py    # 06  (9 frameworks + historical fraud corpus)
│   ├── financial_modeling_agent.py # 07
│   ├── valuation_agent.py        # 08
│   ├── risk_analysis.py          # 09
│   ├── narrative_agent.py        # 10  (ThesisComponent + variant perception)
│   ├── compliance_agent.py       # 11  (17 checks: Indian SEBI/LODR/IndAS + Global IFRS/IOSCO/ISSB)
│   ├── transcript_retrieval.py   # 12
│   ├── historical_data.py        # 13
│   ├── earnings_quality.py       # 14
│   ├── scenario_analysis.py      # 15
│   ├── report_generation.py      # 16
│   ├── industry_intelligence.py  # 17  (Porter Five Forces, TAM, attractiveness score)
│   ├── management_governance.py  # 18  (governance score, promoter pledging, RPT analysis)
│   └── esg_sustainability.py     # 19  (BRSR + ISSB/SASB/GRI/TCFD frameworks)
├── core/
│   ├── config.py                 # all config + env var loading (incl. RAGConfig)
│   ├── llm_manager.py            # multi-provider LLM interface
│   ├── logging_setup.py
│   └── research_philosophy.py    # CIO research philosophy — 11-step sequence, agent specs,
│                                 #   RAG document tag schema, source priority, evidence rules
├── retrieval/
│   ├── vector_store.py           # FAISS IndexFlatIP, child(256)+parent(1024) multi-vector,
│   │                             #   BM25+RRF+cross-encoder, persistence at data/faiss_index/
│   ├── rag_pipeline.py           # 9-node LangGraph pipeline (HyDE, compression, guardrails,
│   │                             #   conversation memory, SSE streaming)
│   ├── chunking.py               # SmartChunker — recursive / contextual / semantic strategies
│   ├── hyde.py                   # HyDE — hypothetical doc embedding blended with query
│   ├── compression.py            # ContextCompressor — keyword filter + LLM extraction + dedup
│   ├── memory.py                 # ConversationStore — session-keyed sliding window + LLM summary
│   ├── guardrails.py             # GuardrailsChecker — rule-based + LLM faithfulness + confidence
│   ├── evaluation.py             # RAGASEvaluator — offline eval: context_relevance, faithfulness, answer_relevance
│   ├── tools.py                  # LangChain tools: SEC EDGAR, web search, Wikipedia, calculator
│   └── ingest.py                 # Document ingestion helpers
├── api/
│   ├── __init__.py
│   └── server.py                 # FastAPI — /query, /stream (SSE), /ingest, /health
├── forensics/
│   ├── beneish.py                # 8-variable Beneish M-Score
│   ├── piotroski.py              # 9-signal F-Score
│   └── altman.py                 # Z-Score (EM + US variants)
├── modeling/
│   ├── financial_model.py        # sector KPI engine (10 sectors)
│   └── forecaster.py             # Bear/Base/Bull 5-year forecasts
├── models/
│   ├── company.py                # CompanyProfile (frozen)
│   ├── financials.py             # IS, BS, CFS, FinancialHistory
│   ├── valuation.py              # WACC, DCF, Relative, SOTP, Scenarios
│   ├── research.py               # AgentOutput, ResearchState, Finding, ThesisComponent,
│   │                             #   ThesisCase, DocumentTag, SourceType, EvidenceLevel
│   └── report.py                 # InstitutionalReport, SectionType
├── orchestrator/
│   └── workflow.py               # ResearchOrchestrator — 6-phase pipeline
├── reporting/
│   └── docx_generator.py         # 11-section DOCX builder
├── storage/
│   ├── storage_manager.py        # per-run file layout
│   ├── audit_trail.py            # append-only JSONL audit log
│   └── database.py               # SQLite + DuckDB
├── valuation/
│   ├── wacc.py                   # CAPM-based WACC (Damodaran 2024)
│   ├── dcf.py                    # FCFF DCF engine
│   ├── relative.py               # peer multiple valuation
│   └── sotp.py                   # sum-of-parts engine
├── config.yaml
├── requirements.txt
├── .env.example
├── main.py                       # CLI
└── quickstart.ipynb              # Jupyter notebook (any platform)
```

## Report Sections

The generated DOCX contains:

1. Cover page — rating, target price, risk score, key data table
2. Executive Summary
3. Investment Thesis (moat, earnings drivers, catalysts)
4. Accounting Quality (forensic scores, revenue recognition, accruals)
5. Financial Statements (5-year historical)
6. Financial Forecasts (5-year Bear/Base/Bull)
7. Valuation (DCF + Relative, single WACC, all scenarios)
8. Risk Analysis (business, financial, regulatory, ESG, country)
9. Scenario Analysis (probability-weighted target, triggers, monitorables)
10. Key Findings Table (red flags + positive indicators)
11. Certification + Regulatory Disclaimer (MiFID II, SEBI, CFA)

## License

MIT
