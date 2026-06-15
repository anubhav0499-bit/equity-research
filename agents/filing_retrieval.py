"""
Agent 02 — Filing Retrieval Agent
Retrieves official filings from Tier 1 sources (SEC EDGAR, NSE, BSE, IR websites).
Priority: Tier 1 > Tier 2 > Tier 3 > Tier 4.
"""

from __future__ import annotations
import re
import time
import httpx
from pathlib import Path
from typing import Optional
from loguru import logger

from .base_agent import BaseAgent
from ..models.research import AgentOutput, AgentStatus, RiskClassification
from ..orchestrator.state import ResearchState
from ..core.config import ACQUISITION_CONFIG


DOCUMENT_TYPES = {
    "annual_report": ["10-K", "20-F", "Annual Report", "Annual_Report"],
    "quarterly":     ["10-Q", "Quarterly Results", "Results"],
    "transcript":    ["Earnings Call", "Conference Call", "Transcript"],
    "governance":    ["DEF 14A", "Proxy", "Corporate Governance"],
    "investor_pres": ["Investor Presentation", "Analyst Day"],
}


class FilingRetrievalAgent(BaseAgent):
    AGENT_ID = "02_filing_retrieval"
    AGENT_NAME = "Filing Retrieval Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        profile = state.company_profile or {}
        ticker = profile.get("ticker", state.ticker)
        exchange = profile.get("exchange", "OTHER")
        sec_cik = profile.get("sec_cik")
        country = profile.get("country", "US")
        years = ACQUISITION_CONFIG.years_history

        acquired: list[dict] = []
        findings = []

        # ── Tier 1A: SEC EDGAR (US companies) ────────────────────
        if exchange in ("NYSE", "NASDAQ") or country == "US":
            edgar_docs = self._fetch_sec_edgar(ticker, sec_cik, years)
            acquired.extend(edgar_docs)
            self.audit.log(self.AGENT_ID, self.AGENT_NAME, "SEC_EDGAR_FETCH",
                          f"SEC EDGAR: {len(edgar_docs)} documents retrieved")

        # ── Tier 1B: NSE/BSE (Indian companies) ──────────────────
        if exchange in ("NSE", "BSE") or country == "IN":
            india_docs = self._fetch_india_filings(ticker, profile.get("bse_code"), years)
            acquired.extend(india_docs)
            self.audit.log(self.AGENT_ID, self.AGENT_NAME, "INDIA_FILING_FETCH",
                          f"India exchanges: {len(india_docs)} documents retrieved")

        # ── Tier 1C: IR Website ───────────────────────────────────
        ir_url = profile.get("ir_url")
        if ir_url:
            ir_docs = self._scrape_ir_website(ir_url, ticker, years)
            acquired.extend(ir_docs)

        # ── Download files ────────────────────────────────────────
        downloaded = self._download_documents(acquired)

        # ── Assess coverage ───────────────────────────────────────
        annual_count = sum(1 for d in downloaded if d.get("type") == "annual_report")
        if annual_count == 0:
            findings.append(self.red_flag(
                title="No annual reports retrieved from primary sources",
                detail="Filing retrieval could not obtain annual reports from official exchange sources.",
                evidence=f"Checked SEC EDGAR: {exchange in ('NYSE', 'NASDAQ')}, NSE/BSE: {exchange in ('NSE', 'BSE')}, IR website: {bool(ir_url)}",
                risk_level=RiskClassification.HIGH,
                confidence=0.95,
            ))
        elif annual_count < 3:
            findings.append(self.make_finding(
                finding_type=__import__("..models.research", fromlist=["FindingType"]).FindingType.WARNING,
                title=f"Limited annual report coverage: only {annual_count} years retrieved",
                detail="Full 5-year historical analysis may be incomplete.",
                evidence=f"Retrieved {annual_count} annual reports; target is 5+.",
                risk_level=RiskClassification.MEDIUM,
                confidence=0.85,
            ))

        risk_score = 15.0 if annual_count >= 5 else (30.0 if annual_count >= 3 else 50.0)

        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=(
                f"Retrieved {len(downloaded)} documents for {ticker}: "
                f"{annual_count} annual reports, "
                f"{sum(1 for d in downloaded if d.get('type') == 'quarterly')} quarterly reports, "
                f"{sum(1 for d in downloaded if d.get('type') == 'transcript')} transcripts."
            ),
            findings=findings,
            risk_score=risk_score,
            risk_classification=RiskClassification.LOW if risk_score < 30 else RiskClassification.MEDIUM,
            payload={
                "documents_retrieved": len(downloaded),
                "annual_reports": annual_count,
                "document_manifest": downloaded[:50],
            },
            sources_used=[d.get("source_url", "") for d in downloaded if d.get("source_url")],
        )

    # ── SEC EDGAR ─────────────────────────────────────────────────

    def _fetch_sec_edgar(self, ticker: str, cik: Optional[str], years: int) -> list[dict]:
        docs = []
        try:
            if not cik:
                cik = self._resolve_sec_cik(ticker)
            if not cik:
                return docs

            url = f"{ACQUISITION_CONFIG.sec_submissions_url}/CIK{cik.zfill(10)}.json"
            resp = self._get(url, headers={"User-Agent": ACQUISITION_CONFIG.sec_user_agent})
            if not resp:
                return docs

            data = resp.json()
            filings = data.get("filings", {}).get("recent", {})
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])
            primary_docs = filings.get("primaryDocument", [])

            from datetime import datetime, timedelta
            cutoff = datetime.now() - timedelta(days=365 * years)

            for i, (form, date_str, acc, doc) in enumerate(zip(forms, dates, accessions, primary_docs)):
                try:
                    filing_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if filing_date < cutoff:
                        continue
                    doc_type = self._classify_form(form)
                    if not doc_type:
                        continue
                    acc_no = acc.replace("-", "")
                    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no}/{doc}"
                    docs.append({
                        "source": "SEC_EDGAR",
                        "type": doc_type,
                        "form": form,
                        "date": date_str,
                        "source_url": url,
                        "accession": acc,
                        "filename": doc,
                        "tier": 1,
                    })
                    if len(docs) >= years * 4:
                        break
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"SEC EDGAR fetch failed: {e}")
        return docs

    def _resolve_sec_cik(self, ticker: str) -> Optional[str]:
        try:
            url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2020-01-01&forms=10-K"
            resp = self._get(url, headers={"User-Agent": ACQUISITION_CONFIG.sec_user_agent})
            if resp and resp.status_code == 200:
                hits = resp.json().get("hits", {}).get("hits", [])
                if hits:
                    return hits[0].get("_source", {}).get("entity_id", "")
        except Exception:
            pass
        try:
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?company=&CIK={ticker}&type=10-K&dateb=&owner=include&count=5&search_text=&action=getcompany&output=atom"
            resp = self._get(url)
            if resp:
                match = re.search(r"CIK=(\d+)", resp.text)
                if match:
                    return match.group(1)
        except Exception:
            pass
        return None

    # ── India Filings ─────────────────────────────────────────────

    def _fetch_india_filings(self, ticker: str, bse_code: Optional[str], years: int) -> list[dict]:
        docs = []
        try:
            session = httpx.Client(
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Referer": "https://www.nseindia.com",
                    "Accept": "application/json",
                },
                timeout=30,
                follow_redirects=True,
            )
            session.get("https://www.nseindia.com", timeout=10)
            time.sleep(1)

            url = f"https://www.nseindia.com/api/annual-reports?index=equities&symbol={ticker.upper()}"
            resp = session.get(url)
            if resp.status_code == 200:
                data = resp.json()
                for item in (data if isinstance(data, list) else []):
                    docs.append({
                        "source": "NSE",
                        "type": "annual_report",
                        "date": item.get("fromDate", ""),
                        "source_url": item.get("fileName", ""),
                        "tier": 1,
                    })
        except Exception as e:
            logger.debug(f"NSE filings fetch: {e}")

        if bse_code:
            try:
                url = f"https://api.bseindia.com/BseIndiaAPI/api/AnnualReport/w?scripcd={bse_code}"
                resp = self._get(url)
                if resp and resp.status_code == 200:
                    items = resp.json() if isinstance(resp.json(), list) else []
                    for item in items[:years]:
                        docs.append({
                            "source": "BSE",
                            "type": "annual_report",
                            "date": item.get("REPORT_DT", ""),
                            "source_url": item.get("PDFLINKANN", ""),
                            "tier": 1,
                        })
            except Exception as e:
                logger.debug(f"BSE filings fetch: {e}")
        return docs

    # ── IR Website ────────────────────────────────────────────────

    def _scrape_ir_website(self, ir_url: str, ticker: str, years: int) -> list[dict]:
        docs = []
        try:
            resp = self._get(ir_url)
            if not resp:
                return docs
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            for link in soup.find_all("a", href=True):
                href = link["href"]
                text = (link.get_text() or "").lower()
                if any(kw in text for kw in ["annual report", "10-k", "20-f", "investor"]):
                    if href.endswith(".pdf") or "pdf" in href:
                        full_url = href if href.startswith("http") else ir_url.rstrip("/") + "/" + href.lstrip("/")
                        docs.append({
                            "source": "IR_WEBSITE",
                            "type": "annual_report",
                            "source_url": full_url,
                            "tier": 1,
                        })
        except Exception as e:
            logger.debug(f"IR website scrape: {e}")
        return docs[:5]

    # ── Download ──────────────────────────────────────────────────

    def _download_documents(self, doc_list: list[dict]) -> list[dict]:
        downloaded = []
        for doc in doc_list:
            url = doc.get("source_url", "")
            if not url or not url.startswith("http"):
                doc["downloaded"] = False
                downloaded.append(doc)
                continue
            try:
                resp = self._get(url, timeout=30)
                if resp and resp.status_code == 200 and len(resp.content) > 1000:
                    ext = "pdf" if "pdf" in url.lower() else "html"
                    fname = f"{doc.get('source', 'doc')}_{doc.get('date', 'nd')}_{doc.get('type', 'filing')}.{ext}"
                    self.storage.save_bytes(resp.content, fname, "Raw_Filings")
                    doc["downloaded"] = True
                    doc["local_filename"] = fname
                    doc["size_bytes"] = len(resp.content)
                    self.audit.log_data_point(
                        self.AGENT_ID, "document_downloaded", fname, url, fname
                    )
                else:
                    doc["downloaded"] = False
                downloaded.append(doc)
                time.sleep(1.0 / ACQUISITION_CONFIG.rate_limit_rps)
            except Exception as e:
                logger.debug(f"Download failed for {url}: {e}")
                doc["downloaded"] = False
                downloaded.append(doc)
        return downloaded

    def _classify_form(self, form: str) -> Optional[str]:
        form = form.upper()
        if form in ("10-K", "10-K/A", "20-F"):
            return "annual_report"
        if form in ("10-Q", "10-Q/A"):
            return "quarterly"
        if "DEF 14A" in form or form == "PROXY":
            return "governance"
        return None

    def _get(self, url: str, headers: dict = None, timeout: int = 30) -> Optional[httpx.Response]:
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                return client.get(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
        except Exception as e:
            logger.debug(f"GET {url}: {e}")
            return None
