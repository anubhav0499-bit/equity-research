"""
Agent 15 — Scenario Analysis Agent
Synthesizes Bear / Base / Bull scenario assumptions, computes scenario-level implied values,
and generates probability-weighted outcomes with key scenario drivers.
Distinct from the Valuation Agent (which builds the models) — this agent interprets and narrates.
"""

from __future__ import annotations
from typing import Optional
from loguru import logger

from .base_agent import BaseAgent
from ..models.research import AgentOutput, AgentStatus, RiskClassification, FindingType
from ..orchestrator.state import ResearchState


SCENARIO_SYSTEM = """You are a sell-side equity research strategist building scenario analysis.
Given Bear / Base / Bull financial forecasts and valuations, produce:
1. Scenario probability weights (must sum to 100%): Bear / Base / Bull
2. For each scenario: the 3 most critical assumptions, what would cause this scenario to materialize, and the probability
3. Probability-weighted target price calculation
4. Key monitorables: what specific metrics/events would shift from Base to Bear or to Bull
5. Sensitivity: which single assumption has the highest impact on valuation?

Format your response as structured analysis with clear section headers. Be quantitative."""


# Default scenario probability weights
DEFAULT_WEIGHTS = {"BEAR": 0.25, "BASE": 0.50, "BULL": 0.25}


class ScenarioAnalysisAgent(BaseAgent):
    AGENT_ID = "15_scenario_analysis"
    AGENT_NAME = "Scenario Analysis Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        profile = state.company_profile or {}
        ticker = profile.get("ticker", state.ticker)
        company_name = profile.get("name", state.company_name)

        findings = []
        details: dict = {
            "scenarios": {},
            "probability_weights": DEFAULT_WEIGHTS.copy(),
            "probability_weighted_target": None,
            "key_monitorables": [],
            "scenario_triggers": {},
            "sensitivity": {},
        }

        # ── Pull scenario data from prior agents ──────────────────
        val_output = state.agent_outputs.get("08_valuation")
        fm_output = state.agent_outputs.get("07_financial_modeling")
        risk_output = state.agent_outputs.get("09_risk_analysis")
        forensic_output = state.agent_outputs.get("06_forensic_accounting")

        val_payload = val_output.payload if val_output else {}
        fm_payload = fm_output.payload if fm_output else {}

        # ── Extract scenario valuations ───────────────────────────
        val_summary = val_payload.get("valuation_summary", {})
        scenarios = val_payload.get("scenarios", {})

        if not scenarios:
            # Try to reconstruct from valuation_summary
            scenarios = self._reconstruct_scenarios(val_summary, val_payload)

        details["scenarios"] = scenarios

        if not scenarios:
            return AgentOutput(
                agent_id=self.AGENT_ID,
                agent_name=self.AGENT_NAME,
                status=AgentStatus.COMPLETED,
                summary="Scenario data not available from upstream valuation agent.",
                findings=[self.make_finding(
                    FindingType.WARNING,
                    "Scenario analysis cannot proceed — no valuation data",
                    "Scenario analysis requires completed valuation. Run Valuation Agent first.",
                    "No scenarios in state.agent_outputs['08_valuation'].payload",
                    risk_level=RiskClassification.MEDIUM, confidence=0.95,
                )],
                risk_score=40.0,
                payload=details,
                sources_used=["08_valuation"],
            )

        # ── Adjust weights based on risk signals ──────────────────
        weights = self._compute_scenario_weights(state, risk_output, forensic_output)
        details["probability_weights"] = weights

        # ── Probability-weighted target price ─────────────────────
        pwtp = self._compute_pwtp(scenarios, weights)
        details["probability_weighted_target"] = pwtp

        # ── Scenario spread analysis ───────────────────────────────
        bear_price = scenarios.get("BEAR", {}).get("implied_price") or scenarios.get("BEAR", {}).get("blended_value")
        base_price = scenarios.get("BASE", {}).get("implied_price") or scenarios.get("BASE", {}).get("blended_value")
        bull_price = scenarios.get("BULL", {}).get("implied_price") or scenarios.get("BULL", {}).get("blended_value")

        if all(v is not None for v in [bear_price, base_price, bull_price]):
            spread = bull_price - bear_price
            bear_to_base_upside = ((base_price - bear_price) / bear_price * 100) if bear_price else None
            details["spread_analysis"] = {
                "bear_price": bear_price,
                "base_price": base_price,
                "bull_price": bull_price,
                "total_spread": spread,
                "bear_to_base_pct": bear_to_base_upside,
                "base_to_bull_pct": ((bull_price - base_price) / base_price * 100) if base_price else None,
            }
            if spread / max(base_price, 0.01) > 1.0:
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    f"Wide scenario spread: Bear ${bear_price:.1f} → Bull ${bull_price:.1f} (+{spread/base_price*100:.0f}% spread)",
                    "High uncertainty in outcomes. Wide Bear-Bull spread indicates significant execution or macro risk.",
                    f"Spread: ${spread:.1f} ({spread/base_price*100:.0f}% of Base case)",
                    risk_level=RiskClassification.MEDIUM, confidence=0.80,
                ))
        else:
            details["spread_analysis"] = {}

        # ── Extract forecast assumptions for scenario comparison ──
        forecasts = fm_payload.get("forecasts", {})
        scenario_assumptions = self._extract_scenario_assumptions(forecasts)
        details["scenario_assumptions"] = scenario_assumptions

        # ── LLM scenario narrative ─────────────────────────────────
        llm_prompt = self._build_llm_prompt(ticker, company_name, details, state)
        llm_analysis = self.llm_analyze(SCENARIO_SYSTEM, llm_prompt, max_tokens=1800)
        details["llm_scenario_analysis"] = llm_analysis

        # ── Parse monitorables from LLM ───────────────────────────
        monitorables = self._extract_monitorables(llm_analysis)
        details["key_monitorables"] = monitorables

        # ── Scenario trigger conditions ────────────────────────────
        details["scenario_triggers"] = self._extract_triggers(llm_analysis)

        # ── Risk: skew towards bear if high overall risk ───────────
        risk_score_from_state = self._compute_scenario_risk(state, weights, scenarios)

        if weights["BEAR"] > 0.35:
            findings.append(self.red_flag(
                f"Elevated bear scenario probability: {weights['BEAR']*100:.0f}%",
                "Risk analysis signals a higher-than-typical probability of the bear scenario materializing.",
                f"Weights: Bear={weights['BEAR']*100:.0f}% Base={weights['BASE']*100:.0f}% Bull={weights['BULL']*100:.0f}%",
                risk_level=RiskClassification.HIGH, confidence=0.75,
            ))
        elif weights["BULL"] > 0.35:
            findings.append(self.green_flag(
                f"Elevated bull scenario probability: {weights['BULL']*100:.0f}%",
                "Multiple positive signals suggest above-average probability of bull case materializing.",
                f"Weights: Bear={weights['BEAR']*100:.0f}% Base={weights['BASE']*100:.0f}% Bull={weights['BULL']*100:.0f}%",
                confidence=0.65,
            ))

        self.storage.save_json(details, "scenario_analysis.json", "Agent_Outputs")

        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=(
                f"Scenario analysis for {ticker}: "
                f"Bear={weights['BEAR']*100:.0f}%/Base={weights['BASE']*100:.0f}%/Bull={weights['BULL']*100:.0f}% weights. "
                f"Probability-weighted target: ${pwtp:.2f}. " if pwtp else ""
                f"Bear ${bear_price:.1f} → Base ${base_price:.1f} → Bull ${bull_price:.1f}."
                if all(v is not None for v in [bear_price, base_price, bull_price]) else
                f"Scenario analysis complete for {ticker}."
            ),
            findings=findings,
            risk_score=risk_score_from_state,
            risk_classification=self._risk_class(risk_score_from_state),
            payload=details,
            sources_used=["08_valuation", "07_financial_modeling", "09_risk_analysis"],
        )

    # ── Helpers ───────────────────────────────────────────────────

    def _reconstruct_scenarios(self, val_summary: dict, val_payload: dict) -> dict:
        """Build minimal scenario dict from flat valuation summary when nested scenarios missing."""
        scenarios = {}
        for scenario in ("BEAR", "BASE", "BULL"):
            key = scenario.lower()
            price_key = f"{key}_price"
            price = val_summary.get(price_key) or val_payload.get(price_key)
            if price:
                scenarios[scenario] = {"implied_price": price, "blended_value": price}
        return scenarios

    def _compute_scenario_weights(self, state: ResearchState, risk_output, forensic_output) -> dict:
        """Adjust probability weights based on risk and forensic scores."""
        weights = DEFAULT_WEIGHTS.copy()
        risk_score = 50.0
        if risk_output:
            risk_score = risk_output.risk_score
        forensic_score = 50.0
        if forensic_output:
            forensic_score = forensic_output.risk_score

        combined_risk = risk_score * 0.6 + forensic_score * 0.4

        if combined_risk > 70:
            # High risk: shift weight to bear
            weights = {"BEAR": 0.40, "BASE": 0.45, "BULL": 0.15}
        elif combined_risk > 55:
            weights = {"BEAR": 0.30, "BASE": 0.50, "BULL": 0.20}
        elif combined_risk < 30:
            # Low risk: slight shift to bull
            weights = {"BEAR": 0.20, "BASE": 0.50, "BULL": 0.30}
        return weights

    def _compute_pwtp(self, scenarios: dict, weights: dict) -> Optional[float]:
        total = 0.0
        weight_sum = 0.0
        for scenario, weight in weights.items():
            data = scenarios.get(scenario, {})
            price = data.get("implied_price") or data.get("blended_value")
            if price is not None:
                total += float(price) * weight
                weight_sum += weight
        if weight_sum == 0:
            return None
        return round(total / weight_sum, 2)

    def _extract_scenario_assumptions(self, forecasts: dict) -> dict:
        """Extract key growth assumptions from each scenario's forecast data."""
        result = {}
        for scenario in ("BEAR", "BASE", "BULL"):
            years = forecasts.get(scenario, [])
            if not years:
                continue
            rev_growths = [y.get("assumptions", {}).get("revenue_growth_pct") for y in years
                           if y.get("assumptions", {}).get("revenue_growth_pct") is not None]
            ebitda_margins = [y.get("assumptions", {}).get("ebitda_margin") for y in years
                              if y.get("assumptions", {}).get("ebitda_margin") is not None]
            result[scenario] = {
                "avg_revenue_growth_pct": round(sum(rev_growths) / len(rev_growths), 1) if rev_growths else None,
                "avg_ebitda_margin_pct": round(sum(ebitda_margins) / len(ebitda_margins), 1) if ebitda_margins else None,
            }
        return result

    def _compute_scenario_risk(self, state: ResearchState, weights: dict, scenarios: dict) -> float:
        base_score = state.overall_risk_score if state.overall_risk_score else 40.0
        bear_weight = weights.get("BEAR", 0.25)
        risk_penalty = (bear_weight - 0.25) * 100
        return min(max(base_score + risk_penalty, 10), 90)

    def _build_llm_prompt(self, ticker: str, company: str, details: dict, state: ResearchState) -> str:
        scenarios = details.get("scenarios", {})
        weights = details.get("probability_weights", {})
        spread = details.get("spread_analysis", {})
        assumptions = details.get("scenario_assumptions", {})
        forecasts_available = bool(state.agent_outputs.get("07_financial_modeling"))
        val_available = bool(state.agent_outputs.get("08_valuation"))

        lines = [
            f"Company: {company} ({ticker})",
            f"Data available: Financial Model: {forecasts_available} | Valuation: {val_available}",
            "",
            "SCENARIO VALUATIONS:",
        ]
        for s in ("BEAR", "BASE", "BULL"):
            sc = scenarios.get(s, {})
            ass = assumptions.get(s, {})
            price = sc.get("implied_price") or sc.get("blended_value")
            lines.append(
                f"  {s}: Price=${price:.2f} | Rev Growth={ass.get('avg_revenue_growth_pct')}% | "
                f"EBITDA Margin={ass.get('avg_ebitda_margin_pct')}%"
                if price else f"  {s}: Price=N/A | {ass}"
            )
        lines += [
            "",
            f"Current probability weights: Bear={weights.get('BEAR', 0)*100:.0f}% "
            f"Base={weights.get('BASE', 0)*100:.0f}% Bull={weights.get('BULL', 0)*100:.0f}%",
            f"Bear-Bull spread: ${spread.get('total_spread', 'N/A')} ({spread.get('bear_to_base_pct', 'N/A')}% bear-to-base)",
            "",
            "Please provide: (1) Recommended probability weights with rationale, "
            "(2) 3 key assumptions per scenario, (3) trigger conditions for scenario shifts, "
            "(4) 3-5 key monitorables for this investment, (5) most impactful single assumption.",
        ]
        return "\n".join(lines)

    def _extract_monitorables(self, analysis: str) -> list[str]:
        """Extract key monitorables from LLM analysis."""
        monitorables = []
        lines = analysis.split("\n")
        in_monitor_section = False
        for line in lines:
            if "monitorable" in line.lower() or "monitor" in line.lower() or "watch" in line.lower():
                in_monitor_section = True
            if in_monitor_section and line.strip().startswith(("-", "•", "*", "1", "2", "3", "4", "5")):
                cleaned = line.strip().lstrip("-•*123456789. ").strip()
                if cleaned and len(cleaned) > 10:
                    monitorables.append(cleaned)
            if len(monitorables) >= 5:
                break
        if not monitorables:
            # Fallback: extract from context
            monitorables = [
                "Revenue growth vs management guidance",
                "EBITDA margin trajectory",
                "Free cash flow conversion",
                "Debt leverage (Net Debt/EBITDA)",
            ]
        return monitorables[:5]

    def _extract_triggers(self, analysis: str) -> dict:
        """Extract scenario shift triggers from LLM analysis."""
        triggers = {"BEAR": [], "BULL": []}
        analysis_lower = analysis.lower()
        bear_start = analysis_lower.find("bear")
        bull_start = analysis_lower.find("bull")
        if bear_start > -1:
            excerpt = analysis[bear_start:bear_start + 500]
            triggers["BEAR"] = [excerpt[:200].strip()]
        if bull_start > -1:
            excerpt = analysis[bull_start:bull_start + 500]
            triggers["BULL"] = [excerpt[:200].strip()]
        return triggers

    def _risk_class(self, score: float) -> RiskClassification:
        if score >= 70: return RiskClassification.HIGH
        if score >= 45: return RiskClassification.MEDIUM
        return RiskClassification.LOW
