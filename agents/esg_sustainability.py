"""
Agent 19 — ESG & Sustainability Agent
Evaluates environmental risk, social impact, and governance quality using
Indian (BRSR) and global (ISSB, SASB, GRI, TCFD) frameworks.
"""

from __future__ import annotations
from .base_agent import BaseAgent
from ..models.research import AgentOutput, AgentStatus, RiskClassification, FindingType
from ..orchestrator.state import ResearchState
from ..core.research_philosophy import AGENT_SPECS


ESG_SYSTEM = """You are a senior ESG analyst with expertise in both Indian and global sustainability frameworks.

Frameworks you apply:
Indian:
- BRSR (Business Responsibility and Sustainability Report) — mandatory for top 1000 listed Indian cos

Global:
- ISSB S1 (General Requirements) and S2 (Climate-related Disclosures)
- SASB — sector-specific sustainability accounting standards
- GRI — comprehensive ESG reporting standards
- TCFD — climate risk and opportunity disclosure

Return ONLY a valid JSON object:
{
  "esg_score": int,                 // 0-100, higher = better ESG profile
  "e_score": int,                   // Environmental score 0-100
  "s_score": int,                   // Social score 0-100
  "g_score": int,                   // Governance score 0-100
  "brsr_compliance": "FULL" | "PARTIAL" | "NONE" | "N/A",
  "tcfd_disclosure": "FULL" | "PARTIAL" | "NONE" | "N/A",
  "material_esg_issues": ["list of top material ESG issues for this sector"],
  "climate_risk_category": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "climate_risk_details": str,
  "social_risk_category": "LOW" | "MEDIUM" | "HIGH",
  "social_risk_details": str,
  "esg_strengths": ["list"],
  "esg_concerns": ["list"],
  "sustainability_disclosure_quality": "STRONG" | "ADEQUATE" | "WEAK" | "ABSENT",
  "stranded_asset_risk": bool,
  "supply_chain_esg_risk": "LOW" | "MEDIUM" | "HIGH",
  "esg_improvement_trajectory": "IMPROVING" | "STABLE" | "DETERIORATING" | "UNCLEAR"
}"""

_SPEC = AGENT_SPECS["09_esg_sustainability"]


class ESGSustainabilityAgent(BaseAgent):
    AGENT_ID = "19_esg_sustainability"
    AGENT_NAME = "ESG & Sustainability Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        profile = state.company_profile or {}
        company = profile.get("name", state.company_name)
        ticker = profile.get("ticker", state.ticker)
        sector = profile.get("sector", "")
        country = profile.get("country", "")

        findings = []
        details: dict = {}

        # ── RAG: ESG disclosures ──────────────────────────────────
        esg_chunks = self.rag_query(
            f"{company} ESG sustainability BRSR environmental social governance climate",
            state, top_k=4,
        )
        esg_context = "\n".join(esg_chunks[:3]) if esg_chunks else ""

        # ── Risk context from prior agents ────────────────────────
        risk_ctx = ""
        risk_out = state.agent_outputs.get("09_risk_analysis")
        if risk_out:
            risk_ctx = risk_out.summary[:300]

        user_prompt = f"""Company: {company} ({ticker})
Sector: {sector} | Country: {country}

ESG/Sustainability disclosures from filings:
{esg_context[:1000]}

Risk context:
{risk_ctx}

Frameworks to apply:
Indian: {', '.join(_SPEC['frameworks']['india'])}
Global: {', '.join(_SPEC['frameworks']['global'])}

Responsibilities:
{chr(10).join(f"- {r}" for r in _SPEC["responsibilities"])}

Assess material ESG issues for this sector, evaluate disclosure quality against
BRSR and global frameworks, and identify climate, social, and governance risks.
Return ONLY the JSON object specified."""

        result = self.llm_analyze(ESG_SYSTEM, user_prompt, json_mode=True)

        if isinstance(result, dict) and "error" not in result:
            details.update(result)

            esg_score = result.get("esg_score", 50)
            e_score = result.get("e_score", 50)
            s_score = result.get("s_score", 50)
            g_score = result.get("g_score", 50)

            # ── Climate risk findings ─────────────────────────────
            climate_risk = result.get("climate_risk_category", "LOW")
            climate_detail = result.get("climate_risk_details", "")
            if climate_risk == "CRITICAL":
                findings.append(self.red_flag(
                    "Critical climate risk exposure",
                    climate_detail or "Company faces critical physical or transition climate risks.",
                    evidence="TCFD/ISSB S2 climate risk assessment",
                    risk_level=RiskClassification.CRITICAL,
                    confidence=0.80,
                ))
            elif climate_risk == "HIGH":
                findings.append(self.red_flag(
                    "High climate risk exposure",
                    climate_detail or "Significant climate transition or physical risk.",
                    evidence="Climate risk assessment (TCFD framework)",
                    risk_level=RiskClassification.HIGH,
                    confidence=0.75,
                ))

            # ── Stranded asset risk ───────────────────────────────
            if result.get("stranded_asset_risk"):
                findings.append(self.red_flag(
                    "Stranded asset risk identified",
                    "Company holds assets at risk of early retirement due to energy transition.",
                    evidence="ESG analysis — energy transition scenario",
                    risk_level=RiskClassification.HIGH,
                    confidence=0.70,
                ))

            # ── Disclosure quality ────────────────────────────────
            disclosure_quality = result.get("sustainability_disclosure_quality", "ADEQUATE")
            if disclosure_quality in ("WEAK", "ABSENT"):
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    f"Sustainability disclosure quality: {disclosure_quality}",
                    "Inadequate ESG disclosure limits investor ability to assess material risks.",
                    evidence=f"BRSR compliance: {result.get('brsr_compliance', 'N/A')} | "
                             f"TCFD: {result.get('tcfd_disclosure', 'N/A')}",
                    risk_level=RiskClassification.MEDIUM,
                    confidence=0.85,
                ))
            elif disclosure_quality == "STRONG":
                findings.append(self.green_flag(
                    "Strong sustainability disclosure",
                    "Company provides comprehensive ESG disclosures aligned with major frameworks.",
                    evidence=f"BRSR: {result.get('brsr_compliance', 'N/A')} | "
                             f"TCFD: {result.get('tcfd_disclosure', 'N/A')}",
                    confidence=0.85,
                ))

            # ── Social risk findings ──────────────────────────────
            social_risk = result.get("social_risk_category", "LOW")
            social_detail = result.get("social_risk_details", "")
            if social_risk == "HIGH":
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    "High social risk identified",
                    social_detail or "Material social risks including labour, community, or supply chain concerns.",
                    evidence="Social risk assessment (GRI/SASB framework)",
                    risk_level=RiskClassification.HIGH,
                    confidence=0.70,
                ))

            # ── Supply chain ESG risk ────────────────────────────
            sc_risk = result.get("supply_chain_esg_risk", "LOW")
            if sc_risk == "HIGH":
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    "High supply chain ESG risk",
                    "Elevated ESG risk in upstream supply chain — reputational and operational exposure.",
                    evidence="Supply chain ESG analysis",
                    risk_level=RiskClassification.MEDIUM,
                    confidence=0.65,
                ))

            # ── ESG trajectory ────────────────────────────────────
            trajectory = result.get("esg_improvement_trajectory", "UNCLEAR")
            if trajectory == "IMPROVING":
                findings.append(self.green_flag(
                    "ESG profile improving",
                    "Company shows a positive ESG improvement trajectory.",
                    evidence="Year-over-year ESG disclosure and initiative comparison",
                    confidence=0.70,
                ))
            elif trajectory == "DETERIORATING":
                findings.append(self.red_flag(
                    "ESG profile deteriorating",
                    "ESG metrics or disclosures are trending negatively.",
                    evidence="ESG trajectory analysis",
                    risk_level=RiskClassification.HIGH,
                    confidence=0.70,
                ))

            # ── ESG score → risk mapping ───────────────────────────
            if esg_score >= 70:
                risk_score = 15.0
            elif esg_score >= 50:
                risk_score = 30.0
            elif esg_score >= 30:
                risk_score = 55.0
            else:
                risk_score = 75.0

        else:
            risk_score = 50.0
            details["error"] = str(result)

        summary = (
            f"ESG Score={details.get('esg_score', 'N/A')}/100 "
            f"(E={details.get('e_score', 'N/A')} "
            f"S={details.get('s_score', 'N/A')} "
            f"G={details.get('g_score', 'N/A')}) | "
            f"Climate Risk={details.get('climate_risk_category', 'N/A')} | "
            f"Disclosure={details.get('sustainability_disclosure_quality', 'N/A')}"
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
