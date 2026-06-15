"""
Agent 03 — Financial Statement Extraction Agent
Extracts structured Income Statement, Balance Sheet, and Cash Flow data
from retrieved filings. Maintains full source traceability per data point.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional
from loguru import logger

from .base_agent import BaseAgent
from ..models.financials import (
    IncomeStatement, BalanceSheet, CashFlowStatement, FinancialHistory,
    FinancialSource, DataQuality
)
from ..models.research import AgentOutput, AgentStatus, RiskClassification, FindingType
from ..orchestrator.state import ResearchState
from ..core.config import ACQUISITION_CONFIG


EXTRACTION_SYSTEM = """You are a financial data extraction specialist.
Extract financial statement data from the provided filing text.
Return ONLY valid JSON. Use null for any field not explicitly stated in the source.
Never estimate or interpolate values. Every value must come from the source text.
All monetary values in millions of the reporting currency unless specified otherwise."""


class FinancialExtractionAgent(BaseAgent):
    AGENT_ID = "03_financial_extraction"
    AGENT_NAME = "Financial Statement Extraction Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        profile = state.company_profile or {}
        ticker = profile.get("ticker", state.ticker)
        currency = profile.get("currency", "USD")

        history = FinancialHistory(company_ticker=ticker, currency=currency)
        findings = []
        sources_used = []

        # Step 1: Try yfinance for structured data (fast path)
        yf_history = self._extract_from_yfinance(ticker, profile)
        if yf_history:
            history = yf_history
            sources_used.append(f"yfinance:{ticker}")

        # Step 2: Extract from downloaded filings (text extraction)
        raw_files = self.storage.list_files("Raw_Filings", "pdf") + \
                    self.storage.list_files("Raw_Filings", "html")
        if raw_files:
            filing_history = self._extract_from_filings(raw_files, ticker, currency)
            history = self._merge_histories(history, filing_history)
            sources_used.extend([str(f) for f in raw_files[:10]])

        # Step 3: Validate completeness
        history = self._compute_completeness(history)
        findings.extend(self._assess_data_quality(history))

        # Step 4: Persist to DB
        profile_id = state.agent_outputs.get("01_company_profiling", {})
        self._persist_to_db(history, state.run_id)

        # Step 5: Save serialized history
        self.storage.save_json(
            self._serialize_history(history),
            "financial_history.json",
            "Financial_Statements",
        )

        years_count = len(history.available_years)
        risk_score = 15.0 if years_count >= 5 else (35.0 if years_count >= 3 else 55.0)

        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=(
                f"Extracted financial statements for {ticker}: {years_count} years "
                f"({', '.join(sorted(history.available_years))}). "
                f"Data completeness: {history.data_completeness_pct:.0f}%. "
                f"Currency: {currency}."
            ),
            findings=findings,
            risk_score=risk_score,
            risk_classification=RiskClassification.LOW if years_count >= 5 else RiskClassification.MEDIUM,
            payload={"financial_history": self._serialize_history(history)},
            sources_used=sources_used,
        )

    # ── yfinance extraction ───────────────────────────────────────

    def _extract_from_yfinance(self, ticker: str, profile: dict) -> Optional[FinancialHistory]:
        try:
            import yfinance as yf
            import pandas as pd
            currency = profile.get("currency", "USD")
            exchange = profile.get("exchange", "")
            candidates = self._ticker_candidates(ticker, exchange)

            for t in candidates:
                try:
                    stock = yf.Ticker(t)
                    fin = stock.financials
                    bs  = stock.balance_sheet
                    cf  = stock.cashflow
                    if fin is None or fin.empty:
                        continue

                    history = FinancialHistory(company_ticker=ticker, currency=currency)

                    for col in fin.columns[:7]:
                        year = str(col.year) if hasattr(col, "year") else str(col)[:4]
                        source = FinancialSource(
                            url=f"yfinance:{t}",
                            document_type="financial_statements",
                            fiscal_year=year,
                            extracted_at=__import__("datetime").datetime.now().isoformat(),
                            confidence=0.9,
                        )

                        def gv(df, name, col=col):
                            if df is not None and not df.empty and name in df.index:
                                try:
                                    v = df.loc[name, col]
                                    return float(v) / 1e6 if pd.notna(v) else None
                                except Exception:
                                    return None
                            return None

                        # Income statement
                        revenue = gv(fin, "Total Revenue")
                        if revenue is None:
                            revenue = gv(fin, "Operating Revenue")
                        gross_profit = gv(fin, "Gross Profit")
                        ebit = gv(fin, "Operating Income") or gv(fin, "EBIT")
                        ebitda = gv(fin, "EBITDA") or gv(fin, "Normalized EBITDA")
                        net_income = gv(fin, "Net Income") or gv(fin, "Net Income Common Stockholders")
                        interest_expense = gv(fin, "Interest Expense")
                        interest_income = gv(fin, "Interest Income")
                        tax = gv(fin, "Tax Provision") or gv(fin, "Income Tax Expense")
                        da = gv(fin, "Depreciation And Amortization") or gv(fin, "Reconciled Depreciation")
                        eps_basic = gv(fin, "Basic EPS")
                        eps_diluted = gv(fin, "Diluted EPS")

                        is_ = IncomeStatement(
                            fiscal_year=year, currency=currency, unit="millions",
                            quality=DataQuality.AUDITED, source=source,
                            revenue=revenue, gross_profit=gross_profit,
                            ebitda=ebitda, ebit=ebit, net_income=net_income,
                            interest_expense=abs(interest_expense) if interest_expense else None,
                            interest_income=interest_income,
                            income_tax=abs(tax) if tax else None,
                            depreciation=da, eps_basic=eps_basic, eps_diluted=eps_diluted,
                            cogs=(revenue - gross_profit) if revenue and gross_profit else None,
                        )
                        history.income_statements[year] = is_

                        # Balance sheet
                        if bs is not None and not bs.empty and col in bs.columns:
                            def gbv(name): return gv(bs, name)
                            total_assets = gbv("Total Assets")
                            total_liabilities = gbv("Total Liabilities Net Minority Interest") or gbv("Total Liab")
                            total_equity = gbv("Stockholders Equity") or gbv("Total Stockholder Equity")
                            cash = gbv("Cash And Cash Equivalents") or gbv("Cash Cash Equivalents And Short Term Investments")
                            ar = gbv("Accounts Receivable") or gbv("Net Receivables")
                            inventory = gbv("Inventory")
                            current_assets = gbv("Current Assets")
                            current_liabilities = gbv("Current Liabilities")
                            ltd = gbv("Long Term Debt") or gbv("Long Term Debt And Capital Lease Obligation")
                            std = gbv("Short Term Debt") or gbv("Current Debt")
                            ap = gbv("Accounts Payable")
                            retained = gbv("Retained Earnings")
                            goodwill = gbv("Goodwill")
                            net_ppe = gbv("Net PPE") or gbv("Net Property Plant And Equipment")

                            bs_ = BalanceSheet(
                                fiscal_year=year, currency=currency, unit="millions",
                                quality=DataQuality.AUDITED, source=source,
                                cash_and_equivalents=cash,
                                accounts_receivable=ar, inventory=inventory,
                                total_current_assets=current_assets,
                                net_ppe=net_ppe, goodwill=goodwill,
                                total_assets=total_assets,
                                accounts_payable=ap,
                                short_term_debt=std,
                                total_current_liabilities=current_liabilities,
                                long_term_debt=ltd,
                                total_liabilities=total_liabilities,
                                retained_earnings=retained,
                                total_equity=total_equity,
                            )
                            history.balance_sheets[year] = bs_

                        # Cash flow
                        if cf is not None and not cf.empty and col in cf.columns:
                            def gcv(name): return gv(cf, name)
                            cfo = gcv("Operating Cash Flow") or gcv("Total Cash From Operating Activities")
                            capex = gcv("Capital Expenditure") or gcv("Capital Expenditures")
                            div = gcv("Cash Dividends Paid") or gcv("Dividends Paid")
                            cfi = gcv("Investing Cash Flow") or gcv("Total Cash From Investing Activities")
                            cff = gcv("Financing Cash Flow") or gcv("Total Cash From Financing Activities")
                            buybacks = gcv("Repurchase Of Capital Stock") or gcv("Common Stock Repurchased")

                            cf_ = CashFlowStatement(
                                fiscal_year=year, currency=currency, unit="millions",
                                quality=DataQuality.AUDITED, source=source,
                                cfo=cfo,
                                capex=abs(capex) if capex else None,
                                dividends_paid=abs(div) if div else None,
                                share_buybacks=abs(buybacks) if buybacks else None,
                                cfi=cfi, cff=cff,
                                net_income_cfs=net_income,
                                depreciation_amortization=da,
                            )
                            history.cash_flows[year] = cf_

                    history.available_years = sorted(
                        set(list(history.income_statements.keys()) +
                            list(history.balance_sheets.keys()))
                    )
                    if history.available_years:
                        return history
                except Exception as e:
                    logger.debug(f"yfinance {t}: {e}")
                    continue
        except ImportError:
            logger.warning("yfinance not installed")
        return None

    def _ticker_candidates(self, ticker: str, exchange: str) -> list[str]:
        t = ticker.upper()
        candidates = [t]
        if exchange == "NSE":
            candidates += [t + ".NS", t + ".BO"]
        elif exchange == "BSE":
            candidates += [t + ".BO", t + ".NS"]
        elif exchange in ("NYSE", "NASDAQ"):
            candidates += [t]
        return candidates[:5]

    # ── Filing text extraction ────────────────────────────────────

    def _extract_from_filings(self, files: list[Path], ticker: str, currency: str) -> FinancialHistory:
        history = FinancialHistory(company_ticker=ticker, currency=currency)
        for file_path in files[:10]:
            try:
                text = self._read_file_text(file_path)
                if len(text) < 500:
                    continue
                year = self._infer_year(file_path.name, text)
                if not year:
                    continue
                extracted = self._llm_extract_financials(text[:12000], year, ticker, currency, str(file_path))
                if extracted.get("income_statement"):
                    is_data = extracted["income_statement"]
                    is_data["fiscal_year"] = year
                    is_data["currency"] = currency
                    is_data["unit"] = "millions"
                    try:
                        history.income_statements[year] = IncomeStatement(**is_data)
                    except Exception as e:
                        logger.debug(f"IS parse error for {year}: {e}")
                if extracted.get("balance_sheet"):
                    bs_data = extracted["balance_sheet"]
                    bs_data["fiscal_year"] = year
                    bs_data["currency"] = currency
                    bs_data["unit"] = "millions"
                    try:
                        history.balance_sheets[year] = BalanceSheet(**bs_data)
                    except Exception as e:
                        logger.debug(f"BS parse error for {year}: {e}")
            except Exception as e:
                logger.debug(f"Filing extraction failed for {file_path.name}: {e}")

        history.available_years = sorted(
            set(list(history.income_statements.keys()) + list(history.balance_sheets.keys()))
        )
        return history

    def _llm_extract_financials(
        self, text: str, year: str, ticker: str, currency: str, source_url: str
    ) -> dict:
        prompt = f"""Extract financial statement data from this filing for {ticker} fiscal year {year}.
Currency: {currency} (report in MILLIONS).

SOURCE TEXT:
{text[:8000]}

Return JSON with this structure (use null for missing values, do NOT estimate):
{{
  "income_statement": {{
    "revenue": null,
    "gross_profit": null,
    "ebitda": null,
    "ebit": null,
    "interest_expense": null,
    "income_tax": null,
    "net_income": null,
    "depreciation": null,
    "eps_basic": null,
    "eps_diluted": null
  }},
  "balance_sheet": {{
    "cash_and_equivalents": null,
    "accounts_receivable": null,
    "inventory": null,
    "total_current_assets": null,
    "net_ppe": null,
    "total_assets": null,
    "accounts_payable": null,
    "short_term_debt": null,
    "total_current_liabilities": null,
    "long_term_debt": null,
    "total_liabilities": null,
    "total_equity": null
  }},
  "cash_flow": {{
    "cfo": null,
    "capex": null,
    "dividends_paid": null,
    "cfi": null,
    "cff": null
  }}
}}"""
        try:
            return self.llm.generate_json(prompt, EXTRACTION_SYSTEM, max_tokens=1024)
        except Exception as e:
            logger.debug(f"LLM extraction failed: {e}")
            return {}

    def _read_file_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                import fitz
                doc = fitz.open(str(path))
                return "\n".join(page.get_text() for page in doc[:30])
            except Exception:
                pass
            try:
                import pdfplumber
                with pdfplumber.open(str(path)) as pdf:
                    return "\n".join(p.extract_text() or "" for p in pdf.pages[:30])
            except Exception:
                pass
        try:
            from bs4 import BeautifulSoup
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            if suffix in (".html", ".htm"):
                return BeautifulSoup(content, "html.parser").get_text(separator="\n")
            return content
        except Exception:
            return ""

    def _infer_year(self, filename: str, text: str = "") -> Optional[str]:
        match = re.search(r"20(\d{2})", filename)
        if match:
            return "20" + match.group(1)
        match = re.search(r"fiscal.{0,20}(20\d{2})", text[:2000], re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    # ── Merge and compute ─────────────────────────────────────────

    def _merge_histories(self, primary: FinancialHistory, secondary: FinancialHistory) -> FinancialHistory:
        for year, is_ in secondary.income_statements.items():
            if year not in primary.income_statements:
                primary.income_statements[year] = is_
        for year, bs_ in secondary.balance_sheets.items():
            if year not in primary.balance_sheets:
                primary.balance_sheets[year] = bs_
        for year, cf_ in secondary.cash_flows.items():
            if year not in primary.cash_flows:
                primary.cash_flows[year] = cf_
        primary.available_years = sorted(
            set(list(primary.income_statements.keys()) + list(primary.balance_sheets.keys()))
        )
        return primary

    def _compute_completeness(self, history: FinancialHistory) -> FinancialHistory:
        if not history.available_years:
            object.__setattr__(history, "data_completeness_pct", 0.0)
            return history
        required_per_year = 12
        filled = 0
        for year in history.available_years:
            is_ = history.income_statements.get(year)
            if is_:
                filled += sum(1 for f in ["revenue", "ebitda", "net_income", "ebit"] if getattr(is_, f))
            bs_ = history.balance_sheets.get(year)
            if bs_:
                filled += sum(1 for f in ["total_assets", "total_equity", "total_liabilities",
                                          "cash_and_equivalents", "long_term_debt"] if getattr(bs_, f))
            cf_ = history.cash_flows.get(year)
            if cf_:
                filled += sum(1 for f in ["cfo", "capex", "fcf"] if getattr(cf_, f))
        total_possible = len(history.available_years) * required_per_year
        pct = min(100.0, (filled / total_possible * 100)) if total_possible else 0.0
        object.__setattr__(history, "data_completeness_pct", round(pct, 1))
        return history

    def _assess_data_quality(self, history: FinancialHistory) -> list:
        findings = []
        if len(history.available_years) < 3:
            findings.append(self.red_flag(
                title=f"Insufficient financial history: {len(history.available_years)} years",
                detail="Less than 3 years of financial data prevents trend analysis and meaningful forecasting.",
                evidence=f"Available years: {history.available_years}",
                risk_level=RiskClassification.HIGH,
                confidence=0.95,
            ))
        if history.data_completeness_pct < 60:
            findings.append(self.red_flag(
                title=f"Low financial data completeness: {history.data_completeness_pct:.0f}%",
                detail="Many financial line items could not be extracted. Analysis reliability is reduced.",
                evidence=f"Completeness: {history.data_completeness_pct:.1f}%",
                risk_level=RiskClassification.MEDIUM,
                confidence=0.85,
            ))
        return findings

    def _persist_to_db(self, history: FinancialHistory, run_id: str) -> None:
        try:
            cid = 0
            for year in history.available_years:
                if is_ := history.income_statements.get(year):
                    self.db.save_financial_data(cid, run_id, year, "income_statement",
                                                is_.model_dump(mode="json"))
                if bs_ := history.balance_sheets.get(year):
                    self.db.save_financial_data(cid, run_id, year, "balance_sheet",
                                                bs_.model_dump(mode="json"))
                if cf_ := history.cash_flows.get(year):
                    self.db.save_financial_data(cid, run_id, year, "cash_flow",
                                                cf_.model_dump(mode="json"))
        except Exception as e:
            logger.warning(f"DB persist failed: {e}")

    def _serialize_history(self, history: FinancialHistory) -> dict:
        return history.model_dump(mode="json")
