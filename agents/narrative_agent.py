"""
Agent 10 — Narrative Generation Agent
Synthesizes all prior agent outputs into the institutional research narrative.
Generates each report section as structured text.
"""

from __future__ import annotations
from typing import Optional
from .base_agent import BaseAgent
from ..models.research import AgentOutput, AgentStatus, RiskClassification
from ..models.report import SectionType
from ..orchestrator.state import ResearchState


def _payload(state: ResearchState, agent_id: str, key: str, default=None):
    """Safely extract a payload key from an agent output, returning default if missing."""
    out = state.agent_outputs.get(agent_id)
    if out is None:
        return default
    return out.payload.get(key, default)


NARRATIVE_SYSTEM = """You are a senior equity research analyst at a top-tier institutional asset manager.
Write in clear, precise, institutional language suitable for professional investors.
Every claim must be backed by data. Use specific numbers, percentages, and fiscal years.
Avoid hedging language like "may" or "could" when evidence is clear.
Maintain consistent tone: analytical, factual, and authoritative."""


class NarrativeGenerationAgent(BaseAgent):
    AGENT_ID = "10_narrative"
    AGENT_NAME = "Narrative Generation Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        profile = state.company_profile or {}
        ticker = profile.get("ticker", state.ticker)
        sector = profile.get("sector", "")
        company_name = state.company_name

        context = state.get_inter_agent_context(exclude_agent=self.AGENT_ID, max_chars=3000)
        market_data = _payload(state, "04_market_data", "market_data", {})
        valuation_data = _payload(state, "08_valuation", "valuation_summary", {})

        sections: dict[str, str] = {}

        # ── Executive Summary ──────────────────────────────────────
        sections[SectionType.EXECUTIVE_SUMMARY.value] = self.llm_analyze(
            NARRATIVE_SYSTEM,
            f"""Write an institutional-grade executive summary for {company_name} ({ticker}).
Sector: {sector}

Prior research findings:
{context}

Valuation: {valuation_data.get('base_price', 'N/A')} (base case target)
Current price: {market_data.get('current_price', 'N/A')}

The executive summary must:
1. Open with investment rating and target price
2. Summarise investment thesis in 2-3 sentences
3. List 3 key bull catalysts
4. List 3 key risk factors
5. Present financial snapshot (revenue, EBITDA margin, EPS, PE)
6. Close with conviction statement

Write 400-600 words. Institutional language only.""",
            max_tokens=1200,
        )

        # ── Investment Thesis ─────────────────────────────────────
        sections[SectionType.INVESTMENT_THESIS.value] = self.llm_analyze(
            NARRATIVE_SYSTEM,
            f"""Write a detailed investment thesis for {company_name} ({ticker}).
Sector: {sector}

Research context:
{context}

The investment thesis must cover:
1. Core thesis statement (2-3 lines)
2. Business model quality assessment
3. Competitive moat analysis (5 forces framework)
4. 3 key earnings drivers with quantified estimates
5. Management quality and capital allocation track record
6. ESG considerations relevant to institutional investors
7. Bull / Bear scenario summary
8. Catalysts for re-rating

Write 600-900 words.""",
            max_tokens=1800,
        )

        # ── Accounting Quality Assessment ─────────────────────────
        forensic_data = _payload(state, "06_forensic_accounting", "details", {})
        sections[SectionType.ACCOUNTING_QUALITY.value] = self.llm_analyze(
            NARRATIVE_SYSTEM,
            f"""Write an accounting quality section for {company_name}.

Forensic scores:
- Beneish M-Score: {forensic_data.get('beneish', {}).get('m_score', 'N/A')} ({forensic_data.get('beneish', {}).get('classification', 'N/A')})
- Piotroski F-Score: {forensic_data.get('piotroski', {}).get('f_score', 'N/A')}/9
- Altman Z-Score: {forensic_data.get('altman', {}).get('z_score', 'N/A')}

Accounting quality findings from prior agents:
{self._accounting_findings(state)}

Cover:
1. Revenue recognition quality
2. Earnings quality (accruals, cash conversion)
3. Balance sheet quality
4. Forensic score interpretations
5. Overall accounting risk rating with justification

Write 400-600 words.""",
            max_tokens=1200,
        )

        # ── Risk Analysis ─────────────────────────────────────────
        risk_data = _payload(state, "09_risk_analysis", "llm_risk_assessment", "")
        sections[SectionType.RISK_ANALYSIS.value] = self.llm_analyze(
            NARRATIVE_SYSTEM,
            f"""Write a risk analysis section for {company_name} suitable for institutional investors.

Risk findings summary:
{risk_data[:2000]}

Structure:
1. Risk summary table (risk category, severity, probability)
2. Top 3 business risks — detailed
3. Financial risks (leverage, liquidity, refinancing)
4. Regulatory and compliance risks
5. ESG risks
6. Country / geopolitical risks
7. Mitigating factors

Write 500-700 words.""",
            max_tokens=1500,
        )

        # ── Scenario Analysis Narrative ───────────────────────────
        sections[SectionType.SCENARIO_ANALYSIS.value] = self.llm_analyze(
            NARRATIVE_SYSTEM,
            f"""Write a scenario analysis section for {company_name}.

Valuation data:
Bear case: {valuation_data.get('bear_price', 'N/A')}
Base case: {valuation_data.get('base_price', 'N/A')} (upside: {valuation_data.get('upside_pct', 'N/A')}%)
Bull case: {valuation_data.get('bull_price', 'N/A')}
WACC: {valuation_data.get('wacc_inputs', {}).get('wacc', 'N/A')}%

For each scenario (Bear, Base, Bull):
1. Key assumption set (revenue growth, EBITDA margin, WACC)
2. Trigger conditions that would lead to this scenario
3. Implied valuation
4. Probability weighting

Write 400-600 words.""",
            max_tokens=1200,
        )

        total_words = sum(len(s.split()) for s in sections.values())

        self.storage.save_json(sections, "narrative_sections.json", "Agent_Outputs")

        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=(
                f"Narrative generation complete for {company_name}. "
                f"Generated {len(sections)} sections, {total_words} words total."
            ),
            findings=[],
            risk_score=20.0,
            risk_classification=RiskClassification.LOW,
            payload={"sections": sections, "word_count": total_words},
        )

    def _accounting_findings(self, state: ResearchState) -> str:
        lines = []
        for aid in ["05_accounting_quality", "06_forensic_accounting"]:
            output = state.agent_outputs.get(aid)
            if output:
                for f in output.red_flags[:3]:
                    lines.append(f"- [{f.risk_level.value}] {f.title}: {f.evidence[:200]}")
        return "\n".join(lines) or "No critical accounting findings identified."
