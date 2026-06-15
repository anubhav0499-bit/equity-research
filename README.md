# Equity Intelligence Research Platform

Autonomous institutional-grade equity research system. Given a company name or ticker, it produces a fully sourced, compliance-validated DOCX research report covering forensic accounting, financial modeling, DCF valuation, and scenario analysis — without human intervention.

## Architecture

17 agents run in a phased pipeline:

```
Phase A  Company Profiling → Filing Retrieval
Phase B  Financial Extraction + Market Data + Transcript Retrieval + Historical Data  (parallel)
Phase C  Accounting Quality + Earnings Quality + Forensic Accounting + Risk Analysis  (parallel)
Phase D  Financial Modeling → Valuation → Scenario Analysis  (sequential)
Phase E  Narrative Generation → Compliance Validation  (sequential)
Phase F  Report Generation
```

Each agent returns a typed `AgentOutput` (Pydantic). No free-form text passes between agents. Full audit trail is written to JSONL.

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
Python 3.9+ · One LLM API key (Groq free tier works)

```bash
# 1. Clone
git clone https://github.com/anubhav0499/equity-intelligence-research.git
cd equity-intelligence-research/equity_research

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
├── agents/               # 17 agent implementations
│   ├── base_agent.py
│   ├── company_profiling.py      # 01
│   ├── filing_retrieval.py       # 02
│   ├── financial_extraction.py   # 03
│   ├── market_data.py            # 04
│   ├── accounting_quality.py     # 05
│   ├── forensic_accounting.py    # 06
│   ├── financial_modeling_agent.py # 07
│   ├── valuation_agent.py        # 08
│   ├── risk_analysis.py          # 09
│   ├── narrative_agent.py        # 10
│   ├── compliance_agent.py       # 11
│   ├── transcript_retrieval.py   # 12
│   ├── historical_data.py        # 13
│   ├── earnings_quality.py       # 14
│   ├── scenario_analysis.py      # 15
│   └── report_generation.py      # 17
├── core/
│   ├── config.py                 # all config + env var loading
│   ├── llm_manager.py            # multi-provider LLM interface
│   └── logging_setup.py
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
│   ├── research.py               # AgentOutput, ResearchState, Finding
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
