"""
Agent 17 — Industry Intelligence Agent
Analyses TAM, Porter Five Forces, competitive intensity, entry barriers,
substitute threats, and regulatory landscape for the target company's industry.
"""

from __future__ import annotations
from .base_agent import BaseAgent
from ..models.research import AgentOutput, AgentStatus, RiskClassification, FindingType
from ..orchestrator.state import ResearchState
from ..core.research_philosophy import AGENT_SPECS


INDUSTRY_SYSTEM = """You are a senior industry analyst with coverage across global and Indian markets.
Your task is to assess the industry attractiveness and competitive dynamics of a company's primary sector.

Always structure your output as valid JSON with these keys:
{
  "industry_name": str,
  "tam_estimate_usd_bn": float,
  "industry_growth_rate_pct": float,
  "industry_attractiveness_score": int,  // 0-100, higher = more attractive
  "porter_five_forces": {
    "competitive_rivalry": {"score": int, "assessment": str},         // 1=low, 5=high intensity
    "supplier_power": {"score": int, "assessment": str},
    "buyer_power": {"score": int, "assessment": str},
    "threat_of_substitutes": {"score": int, "assessment": str},
    "threat_of_new_entrants": {"score": int, "assessment": str}
  },
  "market_structure": str,           // "Monopoly" / "Oligopoly" / "Competitive" / "Fragmented"
  "entry_barriers": ["list of barriers"],
  "key_structural_opportunities": ["list"],
  "key_structural_threats": ["list"],
  "regulatory_environment": str,     // "Benign" / "Moderate" / "Stringent"
  "regulatory_risks": ["list"],
  "competitive_position_of_company": str,
  "growth_forecast_narrative": str
}"""

_SPEC = AGENT_SPECS["02_industry_intelligence"]


class IndustryIntelligenceAgent(BaseAgent):
    AGENT_ID = "17_industry_intelligence"
    AGENT_NAME = "Industry Intelligence Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        profile = state.company_profile or {}
        sector = profile.get("sector", "")
        industry = profile.get("industry", sector)
        company = profile.get("name", state.company_name)
        country = profile.get("country", "")
        ticker = profile.get("ticker", state.ticker)

        findings = []
        details: dict = {}

        # ── Gather context from earlier agents ────────────────────
        macro_context = ""
        md_out = state.agent_outputs.get("04_market_data")
        if md_out:
            macro_context = md_out.payload.get("macro_context", "")[:500]

        user_prompt = f"""Company: {company} ({ticker})
Sector: {sector}
Industry: {industry}
Country / HQ: {country}
Macro context: {macro_context}

Responsibilities:
{chr(10).join(f"- {r}" for r in _SPEC["responsibilities"])}

Deliverables required:
{chr(10).join(f"- {d}" for d in _SPEC["deliverables"])}

Analyse the industry using Porter Five Forces, TAM sizing, competitive intensity,
entry barriers, substitute threats, and regulatory landscape.
Return ONLY the JSON object specified."""

        result = self.llm_analyze(INDUSTRY_SYSTEM, user_prompt, json_mode=True)

        if isinstance(result, dict) and "error" not in result:
            details["industry_analysis"] = result
            details["industry_attractiveness_score"] = result.get("industry_attractiveness_score", 50)
            details["industry_name"] = result.get("industry_name", industry)
            details["porter_five_forces"] = result.get("porter_five_forces", {})
            details["tam_estimate_usd_bn"] = result.get("tam_estimate_usd_bn")
            details["industry_growth_rate_pct"] = result.get("industry_growth_rate_pct")
            details["competitive_position"] = result.get("competitive_position_of_company", "")

            # ── Porter Five Forces findings ───────────────────────
            forces = result.get("porter_five_forces", {})
            for force_name, force_data in forces.items():
                score = force_data.get("score", 3) if isinstance(force_data, dict) else 3
                assessment = force_data.get("assessment", "") if isinstance(force_data, dict) else ""
                if score >= 4:
                    findings.append(self.red_flag(
                        f"High {force_name.replace('_', ' ').title()}: {score}/5",
                        assessment,
                        evidence=f"Porter score: {score}/5",
                        risk_level=RiskClassification.HIGH if score == 5 else RiskClassification.MEDIUM,
                        confidence=0.75,
                    ))
                elif score <= 2:
                    findings.append(self.green_flag(
                        f"Favourable {force_name.replace('_', ' ').title()}: {score}/5",
                        assessment,
                        evidence=f"Porter score: {score}/5",
                        confidence=0.75,
                    ))

            # ── Entry barrier findings ─────────────────────────────
            barriers = result.get("entry_barriers", [])
            if len(barriers) >= 3:
                findings.append(self.green_flag(
                    f"Strong entry barriers ({len(barriers)} identified)",
                    f"High entry barriers protect incumbents: {', '.join(barriers[:3])}",
                    evidence="; ".join(barriers[:3]),
                    confidence=0.70,
                ))
            elif len(barriers) <= 1:
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    "Low entry barriers — disruptive risk elevated",
                    "Minimal barriers to entry increase competitive threat from new entrants.",
                    evidence="; ".join(barriers) if barriers else "No significant barriers identified",
                    risk_level=RiskClassification.MEDIUM,
                    confidence=0.65,
                ))

            # ── Regulatory risk findings ───────────────────────────
            reg_risks = result.get("regulatory_risks", [])
            if reg_risks:
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    f"Regulatory risks identified ({len(reg_risks)})",
                    "; ".join(reg_risks[:3]),
                    evidence="; ".join(reg_risks[:3]),
                    risk_level=RiskClassification.MEDIUM,
                    confidence=0.70,
                ))

            # ── Structural threats ─────────────────────────────────
            threats = result.get("key_structural_threats", [])
            for threat in threats[:2]:
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    f"Structural industry threat: {threat[:80]}",
                    threat,
                    evidence=threat,
                    risk_level=RiskClassification.MEDIUM,
                    confidence=0.65,
                ))

            # ── Industry attractiveness score → risk ──────────────
            attractiveness = result.get("industry_attractiveness_score", 50)
            if attractiveness < 30:
                risk_score = 75.0
            elif attractiveness < 50:
                risk_score = 55.0
            elif attractiveness < 70:
                risk_score = 35.0
            else:
                risk_score = 20.0
        else:
            risk_score = 50.0
            details["error"] = str(result)

        summary_parts = [
            f"Industry: {details.get('industry_name', industry)}",
        ]
        if details.get("industry_attractiveness_score") is not None:
            summary_parts.append(f"Attractiveness: {details['industry_attractiveness_score']}/100")
        if details.get("tam_estimate_usd_bn"):
            summary_parts.append(f"TAM: ${details['tam_estimate_usd_bn']:.0f}B")
        if details.get("industry_growth_rate_pct"):
            summary_parts.append(f"Growth: {details['industry_growth_rate_pct']:.1f}%")
        if details.get("competitive_position"):
            summary_parts.append(f"Company position: {details['competitive_position'][:100]}")

        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=" | ".join(summary_parts),
            findings=findings,
            risk_score=risk_score,
            payload=details,
        )
