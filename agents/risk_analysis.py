"""
Agent 09 — Risk Analysis Agent
Evaluates company-specific, sector, regulatory, ESG, and macro risks.
"""

from __future__ import annotations
from .base_agent import BaseAgent
from ..models.research import AgentOutput, AgentStatus, RiskClassification, FindingType
from ..orchestrator.state import ResearchState


RISK_SYSTEM = """You are a risk officer at an institutional asset manager.
Identify and quantify risks for the given company. Cover:
1. Business/competitive risks
2. Financial leverage and liquidity risks
3. Regulatory and compliance risks
4. ESG risks (environmental, social, governance)
5. Management and governance risks
6. Country / political risks
7. Technology disruption risk
8. Concentration risk (customer, supplier, geography)

For each risk:
- Assign severity: LOW / MEDIUM / HIGH / CRITICAL
- Assign probability: LOW / MEDIUM / HIGH
- State evidence from available data
- Suggest mitigation factor"""


class RiskAnalysisAgent(BaseAgent):
    AGENT_ID = "09_risk_analysis"
    AGENT_NAME = "Risk Analysis Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        profile = state.company_profile or {}
        ticker = profile.get("ticker", state.ticker)
        sector = profile.get("sector", "")
        country = profile.get("country", "US")

        findings = []
        details: dict = {}

        # Get context from earlier agents
        inter_context = state.get_inter_agent_context(exclude_agent=self.AGENT_ID, max_chars=2000)
        market_data = {}
        md_output = state.agent_outputs.get("04_market_data")
        if md_output:
            market_data = md_output.payload.get("market_data", {})

        history_raw = state.financial_history or {}
        fin = self._latest_fin(history_raw)

        # ── 1. Leverage risk ──────────────────────────────────────
        ltd = fin.get("long_term_debt", 0) or 0
        ebitda = fin.get("ebitda", 0) or 0
        net_debt = fin.get("net_debt", 0) or 0
        icr = fin.get("ebit", 0) / max(fin.get("interest_expense", 1) or 1, 1)

        if ebitda > 0:
            nd_ebitda = net_debt / ebitda
            if nd_ebitda > 4:
                findings.append(self.red_flag(
                    f"High leverage: Net Debt/EBITDA = {nd_ebitda:.1f}x",
                    "Net debt exceeds 4x EBITDA — refinancing and covenant risk elevated.",
                    evidence=f"Net Debt={net_debt:.0f}M, EBITDA={ebitda:.0f}M",
                    risk_level=RiskClassification.CRITICAL if nd_ebitda > 6 else RiskClassification.HIGH,
                    confidence=0.90,
                ))
            elif nd_ebitda > 2.5:
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    f"Moderate leverage: Net Debt/EBITDA = {nd_ebitda:.1f}x",
                    "Leverage above median for most sectors.",
                    evidence=f"Net Debt/EBITDA = {nd_ebitda:.1f}x",
                    risk_level=RiskClassification.MEDIUM, confidence=0.80,
                ))

        if icr < 1.5:
            findings.append(self.red_flag(
                f"Interest coverage critical: {icr:.1f}x",
                "Interest coverage below 1.5x — high debt service risk.",
                evidence=f"EBIT={fin.get('ebit')}, Interest={fin.get('interest_expense')}",
                risk_level=RiskClassification.CRITICAL, confidence=0.90,
            ))

        # ── 2. Country / Political risk ───────────────────────────
        high_risk_countries = {"NG", "VE", "ZW", "MM", "KP", "YE", "LY"}
        medium_risk_countries = {"IN", "BR", "ZA", "TR", "AR", "EG", "PK"}
        if country in high_risk_countries:
            findings.append(self.red_flag(
                f"High political risk: {country}",
                "Operating jurisdiction has elevated political and regulatory risk.",
                evidence=f"Country: {country}",
                risk_level=RiskClassification.HIGH, confidence=0.80,
            ))
        elif country in medium_risk_countries:
            findings.append(self.make_finding(
                FindingType.WARNING,
                f"Emerging market risk: {country}",
                "EM jurisdictions carry currency, regulatory, and political risk premium.",
                evidence=f"Country: {country}",
                risk_level=RiskClassification.MEDIUM, confidence=0.75,
            ))

        # ── 3. Short interest / market sentiment ─────────────────
        short_pct = market_data.get("short_pct_float")
        if short_pct and short_pct > 0.15:
            findings.append(self.red_flag(
                f"High short interest: {short_pct:.1%} of float",
                "Elevated short interest may indicate institutional negative sentiment or potential squeeze risk.",
                evidence=f"Short % of float: {short_pct:.1%}",
                risk_level=RiskClassification.HIGH, confidence=0.80,
            ))

        # ── 4. LLM risk assessment ────────────────────────────────
        llm_risks = self.llm_analyze(
            RISK_SYSTEM,
            f"Company: {ticker} | Sector: {sector} | Country: {country}\n\n"
            f"Financial context:\n"
            f"Revenue={fin.get('revenue')}, EBITDA={ebitda}, Net Debt={net_debt}, "
            f"ICR={icr:.1f}x, Net Debt/EBITDA={net_debt/ebitda:.1f}x if {ebitda}>0 else N/A\n\n"
            f"Prior analysis findings:\n{inter_context}\n\n"
            "Identify and quantify the 5 most critical risks for institutional investors.",
            max_tokens=2000,
        )
        details["llm_risk_assessment"] = llm_risks

        # Parse LLM for additional findings
        findings.extend(self._parse_llm_risks(llm_risks, ticker))

        critical_count = sum(1 for f in findings if f.risk_level == RiskClassification.CRITICAL)
        high_count = sum(1 for f in findings if f.risk_level == RiskClassification.HIGH)
        risk_score = min(95.0, 20 + critical_count * 20 + high_count * 10)
        risk_class = (
            RiskClassification.CRITICAL if risk_score >= 75 else
            RiskClassification.HIGH if risk_score >= 55 else
            RiskClassification.MEDIUM if risk_score >= 35 else
            RiskClassification.LOW
        )

        self.storage.save_json(details, "risk_analysis.json", "Agent_Outputs")

        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=(
                f"Risk assessment for {ticker}: {risk_score:.0f}/100 ({risk_class.value}). "
                f"Critical risks: {critical_count}, High risks: {high_count}. "
                f"Key concerns: leverage={net_debt/ebitda:.1f}x EBITDA (if applicable), "
                f"country={country}."
            ),
            findings=findings,
            risk_score=risk_score,
            risk_classification=risk_class,
            payload=details,
        )

    def _latest_fin(self, history: dict) -> dict:
        if not isinstance(history, dict):
            return {}
        years = sorted([k for k in history.keys() if len(k) == 4 and k.isdigit()], reverse=True)
        if not years:
            return {}
        d = history.get(years[0], {})
        if not isinstance(d, dict):
            return {}
        is_d = d.get("income_statements") or {}
        bs_d = d.get("balance_sheets") or {}
        cf_d = d.get("cash_flows") or {}
        return {**is_d, **bs_d, **cf_d}

    def _parse_llm_risks(self, text: str, ticker: str) -> list:
        findings = []
        risk_keywords = ["risk", "threat", "concern", "critical", "high risk", "significant"]
        for line in text.split("\n"):
            line = line.strip()
            if len(line) < 30:
                continue
            lower = line.lower()
            if any(kw in lower for kw in ["critical", "severe", "very high risk"]):
                findings.append(self.red_flag(
                    title=line[:120],
                    detail="LLM-identified risk from narrative analysis.",
                    evidence=line[:300],
                    risk_level=RiskClassification.HIGH, confidence=0.55,
                ))
        return findings[:3]  # Cap at 3 LLM-derived findings
