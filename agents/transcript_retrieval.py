"""
Agent 12 — Transcript Retrieval Agent
Fetches earnings call transcripts from IR websites, Motley Fool, Seeking Alpha (free tier),
and company investor relations pages. Extracts management guidance and key themes.
"""

from __future__ import annotations
import re
import time
from datetime import datetime
from typing import Optional
from loguru import logger

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

from .base_agent import BaseAgent
from ..models.research import AgentOutput, AgentStatus, RiskClassification, FindingType
from ..orchestrator.state import ResearchState


TRANSCRIPT_SYSTEM = """You are an equity research analyst specialized in earnings call analysis.
Analyze the provided transcript excerpts and:
1. Extract explicit management guidance (revenue, margin, EPS, growth targets with quantitative ranges)
2. Identify tone shifts vs prior quarters (confident vs cautious vs defensive)
3. Flag any hedging language around forward-looking statements
4. Note analyst question themes and management deflections
5. Summarize the 3 most important strategic points management made

Be specific: quote exact guidance numbers when available. Flag if guidance was not provided."""


# Sources checked in order (Tier 1 = direct IR, Tier 2 = aggregators)
_MOTLEY_FOOL_URL = "https://www.fool.com/earnings-call-transcripts/?page=1"
_SEEKING_ALPHA_BASE = "https://seekingalpha.com/search?query={ticker}+earnings+transcript"


class TranscriptRetrievalAgent(BaseAgent):
    AGENT_ID = "12_transcript_retrieval"
    AGENT_NAME = "Transcript Retrieval Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        profile = state.company_profile or {}
        ticker = profile.get("ticker", state.ticker)
        company_name = profile.get("name", state.company_name)
        exchange = profile.get("exchange", "")

        findings = []
        details: dict = {
            "transcripts_found": 0,
            "transcripts": [],
            "guidance_summary": {},
            "sentiment": "NEUTRAL",
            "sources_checked": [],
        }

        # ── Try multiple transcript sources ───────────────────────
        transcripts = []

        # Source 1: yfinance news as proxy for recent commentary
        transcripts.extend(self._fetch_via_yfinance_news(ticker, company_name))
        details["sources_checked"].append("yfinance_news")

        # Source 2: SEC 8-K filings (US companies) — earnings releases
        if exchange in ("NYSE", "NASDAQ", "AMEX", "OTC"):
            sec_items = self._fetch_sec_8k(ticker)
            transcripts.extend(sec_items)
            details["sources_checked"].append("SEC_EDGAR_8K")

        # Source 3: NSE/BSE quarterly conference call notices (Indian companies)
        if exchange in ("NSE", "BSE"):
            india_items = self._fetch_bse_conference_call(ticker)
            transcripts.extend(india_items)
            details["sources_checked"].append("BSE_conference_call")

        details["transcripts_found"] = len(transcripts)
        details["transcripts"] = transcripts[:5]  # Store top 5

        if not transcripts:
            findings.append(self.make_finding(
                FindingType.WARNING,
                "No earnings transcripts retrieved",
                "Transcript retrieval returned no results. Management guidance cannot be assessed from transcripts.",
                f"Checked sources: {details['sources_checked']}",
                risk_level=RiskClassification.LOW,
                confidence=0.60,
            ))
            details["guidance_summary"] = {"note": "No transcripts available"}
            return AgentOutput(
                agent_id=self.AGENT_ID,
                agent_name=self.AGENT_NAME,
                status=AgentStatus.COMPLETED,
                summary=f"No earnings transcripts found for {ticker}. Checked {len(details['sources_checked'])} sources.",
                findings=findings,
                risk_score=30.0,
                risk_classification=RiskClassification.LOW,
                payload=details,
                sources_used=details["sources_checked"],
            )

        # ── LLM analysis of transcripts ───────────────────────────
        combined_text = self._build_transcript_summary(transcripts, company_name, ticker)

        # ── Index transcript text into RAG vector store ───────────
        try:
            from ..retrieval import ingest_document
            ingest_document(combined_text, {
                "source": "earnings_transcripts",
                "type":   "transcript",
                "ticker": ticker,
                "count":  len(transcripts),
            }, ticker)
            logger.debug(f"[transcript_retrieval] indexed transcript corpus ({len(combined_text)} chars) for {ticker}")
        except Exception as e:
            logger.debug(f"[transcript_retrieval] RAG indexing failed: {e}")

        llm_analysis = self.llm_analyze(
            TRANSCRIPT_SYSTEM,
            combined_text,
            max_tokens=1500,
        )
        details["llm_transcript_analysis"] = llm_analysis

        # ── Parse guidance from LLM output ────────────────────────
        guidance = self._extract_guidance_from_analysis(llm_analysis)
        details["guidance_summary"] = guidance

        # ── Findings from transcript analysis ─────────────────────
        findings.extend(self._generate_findings(guidance, llm_analysis))

        # ── Sentiment classification ───────────────────────────────
        sentiment = self._classify_sentiment(llm_analysis)
        details["sentiment"] = sentiment

        if sentiment == "CAUTIOUS":
            findings.append(self.make_finding(
                FindingType.WARNING,
                "Cautious management tone detected in earnings calls",
                "LLM analysis detected hedging language and cautious tone in management commentary.",
                f"Sentiment: {sentiment} | Analysis excerpt: {llm_analysis[:300]}",
                risk_level=RiskClassification.MEDIUM,
                confidence=0.70,
            ))
        elif sentiment == "POSITIVE":
            findings.append(self.green_flag(
                "Positive management confidence in earnings calls",
                "Management tone indicates strong confidence in outlook and guidance.",
                f"Sentiment: {sentiment}",
                confidence=0.65,
            ))

        self.storage.save_json(details, "transcripts.json", "Agent_Outputs")

        risk = 20.0 if sentiment == "POSITIVE" else (45.0 if sentiment == "CAUTIOUS" else 30.0)
        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=(
                f"Transcript analysis for {ticker}: {len(transcripts)} items retrieved. "
                f"Management sentiment: {sentiment}. "
                f"Guidance extracted: {len(guidance)} metrics."
            ),
            findings=findings,
            risk_score=risk,
            risk_classification=RiskClassification.LOW if risk < 35 else RiskClassification.MEDIUM,
            payload=details,
            sources_used=details["sources_checked"],
        )

    # ── Source fetchers ───────────────────────────────────────────

    def _fetch_via_yfinance_news(self, ticker: str, company_name: str) -> list[dict]:
        """Use yfinance news feed as a proxy for recent earnings commentary."""
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            news = t.news or []
            earnings_news = []
            for item in news[:20]:
                title = item.get("title", "").lower()
                if any(kw in title for kw in ["earnings", "quarterly", "results", "revenue", "profit", "guidance", "forecast", "beat", "miss"]):
                    earnings_news.append({
                        "title": item.get("title", ""),
                        "publisher": item.get("publisher", ""),
                        "link": item.get("link", ""),
                        "published": item.get("providerPublishTime", ""),
                        "type": "news_summary",
                        "source": "yfinance_news",
                    })
            return earnings_news[:5]
        except Exception as e:
            logger.debug(f"yfinance news fetch failed for {ticker}: {e}")
            return []

    def _fetch_sec_8k(self, ticker: str) -> list[dict]:
        """Fetch recent 8-K filings from SEC EDGAR for earnings releases and press releases."""
        try:
            if not HAS_BS4:
                return []
            # CIK lookup via EDGAR
            search_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2023-01-01&forms=8-K"
            resp = requests.get(search_url, timeout=15, headers={"User-Agent": "equity-research-platform contact@example.com"})
            if resp.status_code != 200:
                return []
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            results = []
            for hit in hits[:3]:
                src = hit.get("_source", {})
                results.append({
                    "title": src.get("display_names", ["8-K Filing"])[0] if src.get("display_names") else "8-K Filing",
                    "form_type": "8-K",
                    "filed_date": src.get("file_date", ""),
                    "description": src.get("entity_name", ""),
                    "type": "sec_8k",
                    "source": "SEC_EDGAR",
                })
            return results
        except Exception as e:
            logger.debug(f"SEC 8-K fetch failed for {ticker}: {e}")
            return []

    def _fetch_bse_conference_call(self, ticker: str) -> list[dict]:
        """Check BSE announcements for conference call transcripts."""
        try:
            if not HAS_BS4:
                return []
            # BSE announcement API for transcript-related announcements
            url = f"https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w?strCat=-1&strPrevDate=&strScrip=&strSearch=MAN&strToDate=&strType=C&subcategory=-1"
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                return []
            data = resp.json()
            items = data.get("Table", [])[:3]
            return [{
                "title": item.get("NEWSSUB", "BSE Announcement"),
                "type": "bse_announcement",
                "source": "BSE",
                "date": item.get("NEWS_DT", ""),
            } for item in items]
        except Exception as e:
            logger.debug(f"BSE conference call fetch failed: {e}")
            return []

    # ── Analysis helpers ──────────────────────────────────────────

    def _build_transcript_summary(self, transcripts: list[dict], company: str, ticker: str) -> str:
        lines = [f"Company: {company} ({ticker})", f"Total items retrieved: {len(transcripts)}", ""]
        for i, t in enumerate(transcripts[:5], 1):
            lines.append(f"Item {i}: {t.get('title', 'N/A')}")
            lines.append(f"  Source: {t.get('source', 'unknown')} | Type: {t.get('type', 'unknown')}")
            lines.append(f"  Date: {t.get('published') or t.get('date') or t.get('filed_date', 'N/A')}")
            if t.get("description"):
                lines.append(f"  Description: {t['description'][:300]}")
            lines.append("")
        lines.append("Based on the above earnings-related filings and news items, "
                     "analyze management guidance, strategic direction, and key risks mentioned.")
        return "\n".join(lines)

    def _extract_guidance_from_analysis(self, analysis: str) -> dict:
        guidance = {}
        # Extract percentage guidance
        pct_pattern = r'(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*%'
        matches = re.findall(pct_pattern, analysis)
        if matches:
            guidance["revenue_guidance_ranges"] = [f"{lo}%-{hi}%" for lo, hi in matches[:3]]
        # Detect if guidance was given
        guidance["guidance_provided"] = any(kw in analysis.lower() for kw in [
            "guidance", "expects", "forecast", "target", "anticipate", "project"
        ])
        guidance["metrics_count"] = len(matches)
        return guidance

    def _classify_sentiment(self, analysis: str) -> str:
        analysis_lower = analysis.lower()
        cautious_words = ["cautious", "concern", "headwind", "challenge", "uncertain", "pressure", "risk", "weaker", "miss", "disappoint"]
        positive_words = ["confident", "strong", "robust", "growth", "beat", "exceed", "record", "momentum", "accelerat"]
        cautious_count = sum(1 for w in cautious_words if w in analysis_lower)
        positive_count = sum(1 for w in positive_words if w in analysis_lower)
        if cautious_count > positive_count + 2:
            return "CAUTIOUS"
        if positive_count > cautious_count + 2:
            return "POSITIVE"
        return "NEUTRAL"

    def _generate_findings(self, guidance: dict, analysis: str) -> list:
        findings = []
        if not guidance.get("guidance_provided"):
            findings.append(self.make_finding(
                FindingType.WARNING,
                "Management did not provide explicit forward guidance",
                "Absence of quantitative guidance reduces visibility into future earnings trajectory.",
                "Guidance keywords not detected in transcript analysis.",
                risk_level=RiskClassification.LOW,
                confidence=0.55,
            ))
        return findings
