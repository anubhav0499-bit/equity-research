"""
Agent 10 — Narrative Generation Agent (Thesis Construction Agent)
Synthesizes all prior agent outputs into the institutional research narrative.
Acts as the CIO's Thesis Construction layer: generates Bull/Base/Bear cases,
variant perception analysis (what market believes vs. what we believe), and the
full 20-section report narrative.

Thesis Construction follows the 11th step of the research sequence:
  - Variant perception: market consensus vs. our differentiated view
  - Why consensus is likely wrong
  - Bull/Base/Bear cases with target prices and key assumptions
  - Catalysts and key risks
"""

from __future__ import annotations
from typing import Optional
from .base_agent import BaseAgent
from ..models.research import (
    AgentOutput, AgentStatus, RiskClassification,
    ThesisComponent, ThesisCase,
)
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

        # ── Thesis Construction (Variant Perception) ─────────────
        thesis = self._build_thesis_component(state, company_name, ticker, sector, context, valuation_data)
        state.thesis = thesis

        # Generate the investment thesis narrative section from the ThesisComponent
        sections[SectionType.INVESTMENT_THESIS.value] = self.llm_analyze(
            NARRATIVE_SYSTEM,
            f"""Write a detailed investment thesis for {company_name} ({ticker}).
Sector: {sector}

Variant Perception Framework:
- Consensus View: {thesis.consensus_view}
- Our View: {thesis.our_view}
- Why consensus is wrong: {thesis.why_consensus_is_wrong}

Research context from all agents:
{context}

Thesis Cases:
- BULL case ({thesis.bull_case.probability*100 if thesis.bull_case and thesis.bull_case.probability else 'N/A'}% probability):
  Target: {thesis.bull_case.target_price if thesis.bull_case else 'N/A'} | {thesis.bull_case.narrative[:300] if thesis.bull_case else ''}
- BASE case ({thesis.base_case.probability*100 if thesis.base_case and thesis.base_case.probability else 'N/A'}% probability):
  Target: {thesis.base_case.target_price if thesis.base_case else 'N/A'} | {thesis.base_case.narrative[:300] if thesis.base_case else ''}
- BEAR case ({thesis.bear_case.probability*100 if thesis.bear_case and thesis.bear_case.probability else 'N/A'}% probability):
  Target: {thesis.bear_case.target_price if thesis.bear_case else 'N/A'} | {thesis.bear_case.narrative[:300] if thesis.bear_case else ''}

Key Catalysts: {', '.join(thesis.catalysts[:4])}
Key Risks: {', '.join(thesis.key_risks[:4])}

The investment thesis must cover:
1. Core thesis statement with variant perception (what we see that the market misses)
2. Business model quality assessment
3. Competitive moat analysis (Porter Five Forces)
4. 3 key earnings drivers with quantified estimates
5. Management quality and capital allocation track record
6. ESG considerations relevant to institutional investors
7. Bull / Base / Bear case summary with probability weights
8. Catalysts for re-rating and conditions for thesis invalidation

Write 700-1000 words. Institutional language only.""",
            max_tokens=2000,
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

        total_words = sum(len(s.split()) for s in sections.values() if isinstance(s, str))

        self.storage.save_json(sections, "narrative_sections.json", "Agent_Outputs")

        thesis_payload = {}
        if state.thesis:
            thesis_payload = {
                "variant_perception": state.thesis.variant_perception,
                "consensus_view": state.thesis.consensus_view,
                "our_view": state.thesis.our_view,
                "why_consensus_is_wrong": state.thesis.why_consensus_is_wrong,
                "catalysts": state.thesis.catalysts,
                "key_risks": state.thesis.key_risks,
                "bull_case": state.thesis.bull_case.model_dump() if state.thesis.bull_case else None,
                "base_case": state.thesis.base_case.model_dump() if state.thesis.base_case else None,
                "bear_case": state.thesis.bear_case.model_dump() if state.thesis.bear_case else None,
            }

        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=(
                f"Narrative + thesis construction complete for {company_name}. "
                f"{len(sections)} sections, {total_words} words. "
                f"Variant perception: {state.thesis.variant_perception[:80] if state.thesis else 'N/A'}"
            ),
            findings=[],
            risk_score=20.0,
            risk_classification=RiskClassification.LOW,
            payload={"sections": sections, "word_count": total_words, "thesis": thesis_payload},
        )

    def _accounting_findings(self, state: ResearchState) -> str:
        lines = []
        for aid in ["05_accounting_quality", "06_forensic_accounting"]:
            output = state.agent_outputs.get(aid)
            if output:
                for f in output.red_flags[:3]:
                    lines.append(f"- [{f.risk_level.value}] {f.title}: {f.evidence[:200]}")
        return "\n".join(lines) or "No critical accounting findings identified."

    def _build_thesis_component(
        self,
        state: ResearchState,
        company_name: str,
        ticker: str,
        sector: str,
        context: str,
        valuation_data: dict,
    ) -> ThesisComponent:
        """
        Build the investment thesis via structured LLM call.
        Extracts variant perception, scenarios, catalysts, and key risks.
        """
        bear_price = valuation_data.get("bear_price")
        base_price = valuation_data.get("base_price")
        bull_price = valuation_data.get("bull_price")
        upside_pct = valuation_data.get("upside_pct", 0.0)

        prompt = f"""You are a CIO constructing the investment thesis for {company_name} ({ticker}).
Sector: {sector}
Valuation: Bear={bear_price} | Base={base_price} (upside {upside_pct}%) | Bull={bull_price}

Agent synthesis:
{context[:2500]}

Return ONLY a valid JSON object with this exact structure:
{{
  "variant_perception": "one sentence — what differentiates our view from consensus",
  "consensus_view": "what the market currently believes about this company",
  "our_view": "our differentiated view based on research",
  "why_consensus_is_wrong": "specific reason the consensus misses this",
  "catalysts": ["catalyst 1", "catalyst 2", "catalyst 3", "catalyst 4"],
  "key_risks": ["risk 1", "risk 2", "risk 3", "risk 4"],
  "bull_case": {{
    "scenario": "BULL",
    "narrative": "bull case narrative (2-3 sentences)",
    "target_price": {bull_price or 0},
    "return_potential_pct": number,
    "key_assumptions": ["assumption 1", "assumption 2", "assumption 3"],
    "probability": 0.25
  }},
  "base_case": {{
    "scenario": "BASE",
    "narrative": "base case narrative (2-3 sentences)",
    "target_price": {base_price or 0},
    "return_potential_pct": {upside_pct},
    "key_assumptions": ["assumption 1", "assumption 2", "assumption 3"],
    "probability": 0.50
  }},
  "bear_case": {{
    "scenario": "BEAR",
    "narrative": "bear case narrative (2-3 sentences)",
    "target_price": {bear_price or 0},
    "return_potential_pct": number,
    "key_assumptions": ["assumption 1", "assumption 2", "assumption 3"],
    "probability": 0.25
  }}
}}"""

        result = self.llm_analyze(
            "You are a CIO constructing investment theses. Return only valid JSON.",
            prompt,
            json_mode=True,
        )

        if not isinstance(result, dict) or "error" in result:
            return ThesisComponent(
                variant_perception="Thesis construction requires valuation completion.",
                consensus_view="Market view not yet determined.",
                our_view="Research synthesis incomplete.",
            )

        def _make_case(data: Optional[dict]) -> Optional[ThesisCase]:
            if not data:
                return None
            try:
                return ThesisCase(
                    scenario=data.get("scenario", "BASE"),
                    narrative=data.get("narrative", ""),
                    target_price=data.get("target_price"),
                    return_potential_pct=data.get("return_potential_pct"),
                    key_assumptions=data.get("key_assumptions", []),
                    probability=data.get("probability"),
                )
            except Exception:
                return None

        return ThesisComponent(
            variant_perception=result.get("variant_perception", ""),
            consensus_view=result.get("consensus_view", ""),
            our_view=result.get("our_view", ""),
            why_consensus_is_wrong=result.get("why_consensus_is_wrong", ""),
            catalysts=result.get("catalysts", []),
            key_risks=result.get("key_risks", []),
            bull_case=_make_case(result.get("bull_case")),
            base_case=_make_case(result.get("base_case")),
            bear_case=_make_case(result.get("bear_case")),
        )
