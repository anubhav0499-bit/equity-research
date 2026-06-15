"""
Agent 08 — Valuation Agent
Runs DCF, Relative Valuation, Historical Multiples, and SOTP.
Single WACC is computed once and shared across all methodologies and scenarios.
"""

from __future__ import annotations
from typing import Optional
from loguru import logger

from .base_agent import BaseAgent
from ..models.research import AgentOutput, AgentStatus, RiskClassification, FindingType
from ..models.valuation import (
    ValuationSummary, ScenarioValuation, PeerMultiple, Scenario, WACCInputs
)
from ..orchestrator.state import ResearchState
from ..valuation.wacc import compute_wacc
from ..valuation.dcf import build_dcf
from ..valuation.relative import build_relative_valuation
from ..modeling.forecaster import FinancialForecaster, ForecastAssumptions


VALUATION_SYSTEM = """You are a CFA charterholder performing valuation for an institutional investor.
Assess the valuation output: are the assumptions reasonable? Is the business worth its current price?
Focus on: key value drivers, sensitivity to WACC and terminal growth, and conviction level.
Provide a clear buy/hold/sell recommendation with target price and upside/downside."""


class ValuationAgent(BaseAgent):
    AGENT_ID = "08_valuation"
    AGENT_NAME = "Valuation Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        profile = state.company_profile or {}
        ticker = profile.get("ticker", state.ticker)
        sector = profile.get("sector", "")
        country = profile.get("country", "US")
        currency = profile.get("currency", "USD")

        # Get market data
        market_data = {}
        md_output = state.agent_outputs.get("04_market_data")
        if md_output:
            market_data = md_output.payload.get("market_data", {})

        current_price = market_data.get("current_price", 0.0) or 0.0
        shares_outstanding = (market_data.get("shares_outstanding") or 0) / 1e6  # Convert to millions
        market_cap_usd = market_data.get("market_cap_usd", 0) or 0
        market_cap_m = market_cap_usd / 1e6

        # Get financial history
        history_raw = state.financial_history or {}
        fm_output = state.agent_outputs.get("07_financial_modeling")
        forecasts_raw = fm_output.payload.get("forecasts", {}) if fm_output else {}

        # Get financial figures from most recent year
        fin = self._extract_latest_financials(history_raw)
        net_debt = fin.get("net_debt", 0.0)
        revenue = fin.get("revenue", 0.0)
        ebitda = fin.get("ebitda", 0.0)
        net_income = fin.get("net_income", 0.0)
        beta = market_data.get("beta") or self._sector_beta(sector)
        long_term_debt = fin.get("long_term_debt", 0.0)

        findings = []
        details: dict = {}

        # ── Step 1: Compute single WACC ───────────────────────────
        wacc_inputs = compute_wacc(
            country=country,
            sector=sector,
            beta=beta,
            debt_book_value=long_term_debt,
            equity_market_cap=market_cap_m,
            effective_tax_rate=fin.get("effective_tax_rate", 25.0),
            beta_source="yfinance (5Y monthly) or sector average",
        )
        details["wacc"] = wacc_inputs.model_dump(mode="json")

        # ── Step 2: Peer multiples ────────────────────────────────
        peer_data = market_data.get("peer_market_data", [])
        peer_multiples = [
            PeerMultiple(
                company=p.get("name", ""),
                ticker=p.get("ticker", ""),
                ev_ebitda=p.get("ev_ebitda"),
                pe=p.get("pe"),
                pb=p.get("pb"),
                ps=p.get("ps"),
            )
            for p in peer_data if isinstance(p, dict)
        ]

        # ── Step 3: Valuations per scenario ──────────────────────
        scenario_valuations: dict[Scenario, ScenarioValuation] = {}

        for scenario in [Scenario.BEAR, Scenario.BASE, Scenario.BULL]:
            forecast_key = scenario.value
            forecast_list = forecasts_raw.get(forecast_key, [])
            forecast_years = self._parse_forecast_years(forecast_list)

            # DCF
            dcf = None
            if forecast_years:
                tgr = {"BEAR": 1.5, "BASE": 3.0, "BULL": 4.0}.get(forecast_key, 3.0)
                discount_map = {"BEAR": -20.0, "BASE": -10.0, "BULL": 0.0}
                dcf = build_dcf(
                    forecast_years=forecast_years,
                    wacc_inputs=wacc_inputs,
                    scenario=scenario,
                    net_debt=net_debt,
                    shares_outstanding=shares_outstanding,
                    current_price=current_price,
                    terminal_growth_rate=tgr,
                )

            # Relative
            discount_map2 = {Scenario.BEAR: -25.0, Scenario.BASE: -10.0, Scenario.BULL: 5.0}
            rel = build_relative_valuation(
                scenario=scenario,
                wacc_inputs=wacc_inputs,
                peer_multiples=peer_multiples,
                current_ebitda=ebitda,
                current_net_income=net_income,
                current_revenue=revenue,
                net_debt=net_debt,
                shares_outstanding=shares_outstanding,
                current_price=current_price,
                discount_pct=discount_map2.get(scenario, 0.0),
            )

            sv = ScenarioValuation(
                scenario=scenario,
                dcf=dcf,
                relative=rel,
                weight_dcf=0.50 if forecast_years else 0.0,
                weight_relative=0.50 if peer_multiples else 0.0,
                weight_historical=0.0,
                weight_sotp=0.0,
            )
            scenario_valuations[scenario] = sv

        # ── Step 4: Valuation summary ─────────────────────────────
        summary = ValuationSummary(
            ticker=ticker,
            current_price=current_price,
            currency=currency,
            wacc_inputs=wacc_inputs,
            bear_case=scenario_valuations[Scenario.BEAR],
            base_case=scenario_valuations[Scenario.BASE],
            bull_case=scenario_valuations[Scenario.BULL],
        )
        details["valuation_summary"] = summary.model_dump(mode="json")

        # ── Step 5: LLM valuation commentary ─────────────────────
        val_text = self._build_valuation_text(summary, current_price, currency)
        llm_commentary = self.llm_analyze(
            VALUATION_SYSTEM,
            f"Company: {ticker} | Sector: {sector} | WACC: {wacc_inputs.wacc:.1f}%\n\n"
            f"{val_text}\n\n"
            f"Context: {state.get_inter_agent_context(exclude_agent=self.AGENT_ID, max_chars=1000)}\n\n"
            "Provide valuation assessment with recommendation and target price.",
            max_tokens=1500,
        )
        details["llm_valuation_commentary"] = llm_commentary

        # ── Step 6: Findings ──────────────────────────────────────
        findings.extend(self._assess_valuation(summary, current_price))

        self.storage.save_json(details, "valuation.json", "Valuation")

        upside = summary.upside_pct or 0.0
        risk_score = 30.0 if abs(upside) < 15 else (20.0 if upside > 15 else 60.0)

        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=(
                f"Valuation for {ticker}: Current price={current_price} {currency}. "
                f"Base case target={summary.base_price:.2f} (upside: {upside:+.1f}%). "
                f"Bear={summary.bear_price:.2f} | Bull={summary.bull_price:.2f}. "
                f"WACC={wacc_inputs.wacc:.1f}%."
                if summary.base_price and summary.bear_price and summary.bull_price
                else f"Valuation for {ticker}: WACC={wacc_inputs.wacc:.1f}%. "
                     f"Insufficient data for full scenario pricing."
            ),
            findings=findings,
            risk_score=risk_score,
            risk_classification=RiskClassification.LOW if upside > 20 else (
                RiskClassification.MEDIUM if upside > -10 else RiskClassification.HIGH
            ),
            payload=details,
        )

    def _extract_latest_financials(self, history: dict) -> dict:
        if not isinstance(history, dict):
            return {}
        # Try nested structure
        years = sorted([k for k in history.keys() if k.isdigit() or (len(k) == 4)], reverse=True)
        if not years:
            return {}
        latest = history.get(years[0], {})
        if isinstance(latest, dict):
            is_d = latest.get("income_statements") or {}
            bs_d = latest.get("balance_sheets") or {}
            cf_d = latest.get("cash_flows") or {}
            merged = {**is_d, **bs_d, **cf_d}
            return merged
        return {}

    def _parse_forecast_years(self, forecast_list: list) -> list:
        from ..modeling.forecaster import ForecastYear
        result = []
        for item in forecast_list:
            if isinstance(item, dict):
                try:
                    result.append(ForecastYear(**item))
                except Exception:
                    pass
        return result

    def _sector_beta(self, sector: str) -> float:
        betas = {
            "Information Technology": 1.25, "Health Care": 0.85,
            "Consumer Staples": 0.65, "Consumer Discretionary": 1.15,
            "Financials": 1.10, "Energy": 1.05, "Industrials": 1.00,
            "Materials": 1.10, "Real Estate": 0.80, "Utilities": 0.55,
            "Telecom": 0.85,
        }
        return betas.get(sector, 1.0)

    def _build_valuation_text(self, summary: ValuationSummary, price: float, currency: str) -> str:
        lines = [
            f"Current Price: {price:.2f} {currency}",
            f"WACC: {summary.wacc_inputs.wacc:.2f}% | Beta: {summary.wacc_inputs.beta:.2f}",
            f"Bear case target: {summary.bear_price:.2f}" if summary.bear_price else "",
            f"Base case target: {summary.base_price:.2f} (upside: {summary.upside_pct:+.1f}%)" if summary.base_price else "",
            f"Bull case target: {summary.bull_price:.2f}" if summary.bull_price else "",
        ]
        return "\n".join(l for l in lines if l)

    def _assess_valuation(self, summary: ValuationSummary, price: float) -> list:
        findings = []
        upside = summary.upside_pct or 0.0
        if upside > 30:
            findings.append(self.green_flag(
                f"Significant undervaluation: base case upside {upside:+.1f}%",
                "Base case DCF and relative valuation imply material upside.",
                evidence=f"Base={summary.base_price:.2f} vs Current={price:.2f}",
                confidence=0.70,
            ))
        elif upside < -20:
            findings.append(self.red_flag(
                f"Overvaluation risk: base case downside {upside:+.1f}%",
                "Current price exceeds base case intrinsic value estimate.",
                evidence=f"Base={summary.base_price:.2f} vs Current={price:.2f}",
                risk_level=RiskClassification.HIGH, confidence=0.70,
            ))
        return findings
