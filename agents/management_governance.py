"""
Agent 18 — Management & Governance Agent
Assesses promoter quality, executive credibility, board independence,
capital allocation track record, related-party transactions, and compensation.
"""

from __future__ import annotations
from .base_agent import BaseAgent
from ..models.research import AgentOutput, AgentStatus, RiskClassification, FindingType
from ..orchestrator.state import ResearchState
from ..core.research_philosophy import AGENT_SPECS


MGMT_SYSTEM = """You are a governance analyst and forensic investigator specialising in corporate governance
and management quality assessment for institutional investors.

Assess management quality and governance using:
- Promoter/founder background and track record
- Executive team depth and tenure
- Board independence and committee composition
- Capital allocation decisions (M&A, capex, dividends, buybacks)
- Related party transaction history
- Compensation vs. performance alignment
- Disclosure quality and transparency

Return ONLY a valid JSON object:
{
  "governance_score": int,               // 0-100
  "management_credibility_score": int,   // 0-100
  "capital_allocation_score": int,       // 0-100
  "promoter_holding_pct": float,
  "promoter_pledging_pct": float,
  "board_independence_pct": float,
  "independent_directors_count": int,
  "total_directors": int,
  "ceo_tenure_years": float,
  "key_governance_strengths": ["list"],
  "key_governance_concerns": ["list"],
  "related_party_risk": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "related_party_details": str,
  "capital_allocation_assessment": str,
  "compensation_aligned_with_performance": bool,
  "management_credibility_assessment": str,
  "key_governance_red_flags": ["list"]
}"""

_SPEC = AGENT_SPECS["04_management_governance"]


class ManagementGovernanceAgent(BaseAgent):
    AGENT_ID = "18_management_governance"
    AGENT_NAME = "Management & Governance Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        profile = state.company_profile or {}
        company = profile.get("name", state.company_name)
        ticker = profile.get("ticker", state.ticker)
        country = profile.get("country", "")
        sector = profile.get("sector", "")

        findings = []
        details: dict = {}

        # ── Financial context for capital allocation assessment ────
        history_raw = state.financial_history or {}
        fin = self._latest_fin(history_raw)

        # ── Earnings call context for management credibility ──────
        transcript_ctx = ""
        tc_out = state.agent_outputs.get("12_transcript_retrieval")
        if tc_out:
            transcript_ctx = tc_out.payload.get("transcript_summary", "")[:600]

        # ── RAG: governance disclosures ───────────────────────────
        governance_chunks = self.rag_query(
            f"{company} board composition related party transactions promoter holding governance",
            state, top_k=4,
        )
        governance_context = "\n".join(governance_chunks[:3]) if governance_chunks else ""

        user_prompt = f"""Company: {company} ({ticker})
Sector: {sector} | Country: {country}

Financial data (for capital allocation context):
- Revenue: {fin.get('revenue', 'N/A')}
- Net Income: {fin.get('net_income', 'N/A')}
- Capex: {fin.get('capital_expenditures', 'N/A')}
- Dividends paid: {fin.get('dividends_paid', 'N/A')}
- FCF: {fin.get('free_cash_flow', 'N/A')}

Governance disclosures from filings:
{governance_context[:800]}

Earnings call management commentary:
{transcript_ctx}

Responsibilities for this analysis:
{chr(10).join(f"- {r}" for r in _SPEC["responsibilities"])}

Deliverables:
{chr(10).join(f"- {d}" for d in _SPEC["deliverables"])}

Assess management quality and governance. Flag any related-party risks,
promoter pledging, board independence gaps, or compensation misalignment.
Return ONLY the JSON object specified."""

        result = self.llm_analyze(MGMT_SYSTEM, user_prompt, json_mode=True)

        if isinstance(result, dict) and "error" not in result:
            details.update(result)

            gov_score = result.get("governance_score", 50)
            mgmt_score = result.get("management_credibility_score", 50)
            cap_score = result.get("capital_allocation_score", 50)
            details["composite_governance_score"] = round((gov_score + mgmt_score + cap_score) / 3, 1)

            # ── Promoter pledging risk ─────────────────────────────
            pledging_pct = result.get("promoter_pledging_pct", 0.0) or 0.0
            if pledging_pct > 50:
                findings.append(self.red_flag(
                    f"Critical promoter pledging: {pledging_pct:.1f}%",
                    "Over 50% promoter shares pledged — significant margin call risk if stock falls.",
                    evidence=f"Pledging = {pledging_pct:.1f}%",
                    risk_level=RiskClassification.CRITICAL,
                    confidence=0.90,
                ))
            elif pledging_pct > 25:
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    f"Elevated promoter pledging: {pledging_pct:.1f}%",
                    "Significant promoter shares pledged — monitor for margin call events.",
                    evidence=f"Pledging = {pledging_pct:.1f}%",
                    risk_level=RiskClassification.HIGH,
                    confidence=0.85,
                ))

            # ── Board independence ─────────────────────────────────
            board_ind_pct = result.get("board_independence_pct", 50.0) or 50.0
            if board_ind_pct < 33:
                findings.append(self.red_flag(
                    f"Low board independence: {board_ind_pct:.0f}%",
                    "Below minimum recommended threshold of 33% independent directors.",
                    evidence=f"Independent directors: {result.get('independent_directors_count', 'N/A')} "
                             f"of {result.get('total_directors', 'N/A')} total",
                    risk_level=RiskClassification.HIGH,
                    confidence=0.90,
                ))
            elif board_ind_pct >= 50:
                findings.append(self.green_flag(
                    f"Strong board independence: {board_ind_pct:.0f}%",
                    "Board majority independent — good governance structure.",
                    evidence=f"Board independence: {board_ind_pct:.0f}%",
                    confidence=0.85,
                ))

            # ── Related party risk ────────────────────────────────
            rpt_risk = result.get("related_party_risk", "LOW")
            rpt_detail = result.get("related_party_details", "")
            if rpt_risk in ("HIGH", "CRITICAL"):
                findings.append(self.red_flag(
                    f"Related party transaction risk: {rpt_risk}",
                    rpt_detail or "Material related-party transactions detected requiring scrutiny.",
                    evidence=rpt_detail[:300] if rpt_detail else "RPT risk flagged by governance analysis",
                    risk_level=RiskClassification.CRITICAL if rpt_risk == "CRITICAL" else RiskClassification.HIGH,
                    confidence=0.80,
                ))

            # ── Governance red flags ───────────────────────────────
            for flag in result.get("key_governance_red_flags", [])[:3]:
                findings.append(self.red_flag(
                    f"Governance red flag: {flag[:80]}",
                    flag,
                    evidence="Identified via governance analysis of filings and disclosures",
                    risk_level=RiskClassification.HIGH,
                    confidence=0.75,
                ))

            # ── Governance strengths ──────────────────────────────
            for strength in result.get("key_governance_strengths", [])[:2]:
                findings.append(self.green_flag(
                    f"Governance strength: {strength[:80]}",
                    strength,
                    evidence="Identified via governance analysis",
                    confidence=0.70,
                ))

            # ── Compensation alignment ────────────────────────────
            if result.get("compensation_aligned_with_performance") is False:
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    "Executive compensation not aligned with shareholder returns",
                    "Compensation structure rewards management irrespective of performance outcomes.",
                    evidence="Compensation analysis vs. financial KPIs",
                    risk_level=RiskClassification.MEDIUM,
                    confidence=0.70,
                ))

            # ── Risk score from composite governance score ─────────
            comp = details["composite_governance_score"]
            if comp < 30:
                risk_score = 80.0
            elif comp < 50:
                risk_score = 60.0
            elif comp < 70:
                risk_score = 35.0
            else:
                risk_score = 15.0

            # Boost risk for critical findings
            critical_count = sum(1 for f in findings if f.risk_level == RiskClassification.CRITICAL)
            risk_score = min(100.0, risk_score + critical_count * 10)

        else:
            risk_score = 50.0
            details["error"] = str(result)

        summary = (
            f"Governance={details.get('governance_score', 'N/A')}/100 | "
            f"Mgmt Credibility={details.get('management_credibility_score', 'N/A')}/100 | "
            f"Capital Allocation={details.get('capital_allocation_score', 'N/A')}/100 | "
            f"RPT Risk={details.get('related_party_risk', 'N/A')} | "
            f"Board Independence={details.get('board_independence_pct', 'N/A')}%"
        )

        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=summary,
            findings=findings,
            risk_score=risk_score,
            payload=details,
        )
