"""
Agent 01 — Company Profiling Agent
Resolves company identity and produces the master CompanyProfile configuration object.
All downstream agents depend on this output.
"""

from __future__ import annotations
import re
from typing import Optional
from loguru import logger

from .base_agent import BaseAgent
from ..models.company import (
    CompanyProfile, Exchange, Currency, FiscalYearEnd, BusinessModel, ExchangeInfo
)
from ..models.research import AgentOutput, AgentStatus, RiskClassification, FindingType
from ..orchestrator.state import ResearchState


SYSTEM_PROMPT = """You are an expert equity research analyst specializing in company identification and profiling.
Your task is to identify and resolve company details from a company name or ticker symbol.
Always return valid, structured JSON. Never fabricate information you are not certain of — use null for unknown fields.
"""


class CompanyProfilingAgent(BaseAgent):
    AGENT_ID = "01_company_profiling"
    AGENT_NAME = "Company Profiling Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        company_name = state.company_name
        ticker_hint = state.ticker

        # Step 1: Try structured data sources
        profile_data = self._resolve_from_yfinance(company_name, ticker_hint)

        # Step 2: Enrich with LLM if critical fields are missing
        if not profile_data or not profile_data.get("sector"):
            llm_data = self._resolve_with_llm(company_name, ticker_hint, profile_data)
            if llm_data:
                for k, v in llm_data.items():
                    if v and not profile_data.get(k):
                        profile_data[k] = v

        # Step 3: Build validated Pydantic model
        profile = self._build_profile(profile_data, company_name, ticker_hint)

        # Step 4: Assess resolution quality
        findings = []
        confidence = self._compute_confidence(profile)
        if confidence < 0.6:
            findings.append(self.red_flag(
                title=f"Low confidence company resolution: {company_name}",
                detail="Critical company fields could not be resolved from primary sources.",
                evidence=f"Confidence score: {confidence:.0%}. Missing: {self._missing_fields(profile)}",
                risk_level=RiskClassification.HIGH,
                confidence=0.9,
            ))

        self.audit.log(
            self.AGENT_ID, self.AGENT_NAME, "COMPANY_RESOLVED",
            f"Resolved: {profile.name} ({profile.ticker}) on {profile.exchange.value}",
        )

        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=(
                f"Identified {profile.name} ({profile.ticker}) listed on {profile.exchange.value} "
                f"({profile.country}). Sector: {profile.sector}. FY end: {profile.fiscal_year_end.value}. "
                f"Resolution confidence: {confidence:.0%}."
            ),
            findings=findings,
            risk_score=20.0 if confidence > 0.7 else 40.0,
            risk_classification=RiskClassification.LOW if confidence > 0.7 else RiskClassification.MEDIUM,
            confidence=confidence,
            payload={"company_profile": profile.model_dump(mode="json")},
            sources_used=profile_data.get("_sources", []),
        )

    # ── Resolution methods ────────────────────────────────────────

    def _resolve_from_yfinance(self, name: str, ticker: str = "") -> dict:
        try:
            import yfinance as yf
            tickers_to_try = self._candidate_tickers(name, ticker)
            for t in tickers_to_try:
                try:
                    info = yf.Ticker(t).info
                    if info and info.get("longName"):
                        return self._map_yfinance_info(info, t)
                except Exception:
                    continue
        except ImportError:
            logger.warning("yfinance not installed; skipping yfinance resolution")
        return {}

    def _candidate_tickers(self, name: str, hint: str) -> list[str]:
        candidates = []
        if hint:
            h = hint.upper()
            candidates += [h, h + ".NS", h + ".BO"]
        slug = re.sub(r"[^\w]", "", name.upper())[:10]
        candidates += [slug, slug + ".NS", slug + ".BO"]
        return list(dict.fromkeys(candidates))[:10]

    def _map_yfinance_info(self, info: dict, ticker: str) -> dict:
        exchange_raw = info.get("exchange", "").upper()
        exchange_map = {
            "NYQ": "NYSE", "NMS": "NASDAQ", "NGM": "NASDAQ",
            "NSI": "NSE", "BSE": "BSE", "LSE": "LSE",
        }
        exchange_str = exchange_map.get(exchange_raw, exchange_raw or "OTHER")

        currency_raw = (info.get("currency") or "USD").upper()
        try:
            currency = Currency(currency_raw)
        except ValueError:
            currency = Currency.USD

        fy_end_month = info.get("lastFiscalYearEnd", "")
        fiscal_year_end = self._infer_fy_end(fy_end_month, info.get("country", ""))

        return {
            "name": info.get("longName") or info.get("shortName") or ticker,
            "ticker": ticker.split(".")[0],
            "exchange": exchange_str,
            "country": (info.get("country") or "US").upper(),
            "currency": currency.value,
            "fiscal_year_end": fiscal_year_end,
            "sector": info.get("sector") or info.get("sectorDisp") or "",
            "industry": info.get("industry") or info.get("industryDisp") or "",
            "market_cap_usd": info.get("marketCap"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "ir_url": info.get("website") or "",
            "auditor": info.get("auditRisk", ""),
            "free_float_pct": None,
            "isin": "",
            "_sources": [f"yfinance:{ticker}"],
        }

    def _resolve_with_llm(self, name: str, ticker: str, existing: dict) -> dict:
        prompt = f"""Identify the following publicly listed company and provide structured details.

Company: {name}
Ticker hint: {ticker or 'unknown'}
Already resolved: {list(existing.keys()) if existing else 'nothing'}

Return ONLY valid JSON with these fields (use null for unknown):
{{
  "name": "Official registered company name",
  "ticker": "Primary ticker symbol",
  "exchange": "NYSE|NASDAQ|NSE|BSE|LSE|TSX|ASX|HKEX|SGX|EURONEXT|OTHER",
  "country": "2-letter ISO country code (e.g. US, IN, GB)",
  "currency": "USD|INR|GBP|EUR|CAD|AUD|HKD|SGD|JPY|CNY",
  "fiscal_year_end": "MARCH|DECEMBER|JUNE|SEPTEMBER",
  "sector": "GICS sector name",
  "industry": "GICS industry name",
  "business_model": "B2B|B2C|B2B2C|MARKETPLACE|SUBSCRIPTION|TRANSACTION|ASSET_HEAVY|ASSET_LIGHT|FINANCIAL_SERVICES|MIXED",
  "isin": "12-character ISIN or null",
  "ir_url": "Investor relations URL or null",
  "auditor": "Auditor firm name or null"
}}"""
        try:
            return self.llm.generate_json(prompt, SYSTEM_PROMPT, max_tokens=512)
        except Exception as e:
            logger.warning(f"LLM company resolution failed: {e}")
            return {}

    def _build_profile(self, data: dict, fallback_name: str, fallback_ticker: str) -> CompanyProfile:
        def safe_exchange(v: str) -> Exchange:
            try:
                return Exchange(v.upper())
            except (ValueError, AttributeError):
                return Exchange.OTHER

        def safe_currency(v: str) -> Currency:
            try:
                return Currency(v.upper())
            except (ValueError, AttributeError):
                return Currency.USD

        def safe_fy(v: str) -> FiscalYearEnd:
            try:
                return FiscalYearEnd(v.upper())
            except (ValueError, AttributeError):
                return FiscalYearEnd.DECEMBER

        def safe_bm(v: str) -> BusinessModel:
            try:
                return BusinessModel(v.upper())
            except (ValueError, AttributeError):
                return BusinessModel.MIXED

        isin = data.get("isin") or ""
        isin = isin if re.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$", isin) else None

        return CompanyProfile(
            name=data.get("name") or fallback_name,
            ticker=(data.get("ticker") or fallback_ticker or fallback_name[:6]).upper(),
            isin=isin,
            exchange=safe_exchange(data.get("exchange") or "OTHER"),
            country=(data.get("country") or "US").upper()[:2],
            currency=safe_currency(data.get("currency") or "USD"),
            fiscal_year_end=safe_fy(data.get("fiscal_year_end") or "DECEMBER"),
            sector=data.get("sector") or "Unknown",
            industry=data.get("industry") or "Unknown",
            business_model=safe_bm(data.get("business_model") or "MIXED"),
            market_cap_usd=data.get("market_cap_usd"),
            shares_outstanding=data.get("shares_outstanding"),
            ir_url=data.get("ir_url"),
            auditor=data.get("auditor"),
            resolution_confidence=self._compute_confidence_from_data(data),
            resolution_source="yfinance+llm",
        )

    def _compute_confidence(self, profile: CompanyProfile) -> float:
        score = 1.0
        if profile.sector == "Unknown": score -= 0.15
        if profile.industry == "Unknown": score -= 0.10
        if not profile.isin: score -= 0.05
        if not profile.market_cap_usd: score -= 0.10
        if not profile.ir_url: score -= 0.05
        return max(0.3, score)

    def _compute_confidence_from_data(self, data: dict) -> float:
        required = ["name", "ticker", "exchange", "country", "sector", "industry"]
        filled = sum(1 for k in required if data.get(k))
        return round(filled / len(required), 2)

    def _missing_fields(self, profile: CompanyProfile) -> str:
        missing = []
        if profile.sector == "Unknown": missing.append("sector")
        if profile.industry == "Unknown": missing.append("industry")
        if not profile.isin: missing.append("isin")
        if not profile.market_cap_usd: missing.append("market_cap")
        return ", ".join(missing) or "none"

    def _infer_fy_end(self, fy_date_str: str, country: str) -> str:
        country = (country or "").upper()
        if country in ("IN", "IND", "INDIA"):
            return "MARCH"
        if fy_date_str:
            try:
                from datetime import datetime
                dt = datetime.fromtimestamp(int(fy_date_str))
                month_map = {3: "MARCH", 6: "JUNE", 9: "SEPTEMBER", 12: "DECEMBER"}
                return month_map.get(dt.month, "DECEMBER")
            except Exception:
                pass
        return "DECEMBER"
