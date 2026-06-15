"""
Agent 07 — Financial Modeling Agent
Builds the 5-year historical + 5-year forward integrated financial model.
Sector-specific KPIs are generated dynamically.
"""

from __future__ import annotations
from loguru import logger

from .base_agent import BaseAgent
from ..models.research import AgentOutput, AgentStatus, RiskClassification, FindingType
from ..models.financials import FinancialHistory
from ..modeling.financial_model import FinancialModelEngine
from ..modeling.forecaster import FinancialForecaster
from ..orchestrator.state import ResearchState


MODELING_SYSTEM = """You are a sell-side equity research analyst building financial models.
Review the financial model data provided and:
1. Identify key earnings drivers for this company and sector
2. Assess quality and sustainability of historical margins
3. Flag any unusual trends requiring explanation
4. Comment on the appropriateness of forecast assumptions
Be specific and quantitative. Reference actual numbers."""


class FinancialModelingAgent(BaseAgent):
    AGENT_ID = "07_financial_modeling"
    AGENT_NAME = "Financial Modeling Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        profile = state.company_profile or {}
        sector = profile.get("sector", "")
        ticker = profile.get("ticker", state.ticker)
        history_raw = state.financial_history or {}

        findings = []
        details: dict = {}

        # Reconstruct FinancialHistory from serialized state
        history = self._deserialize_history(history_raw, ticker)

        if not history.available_years:
            return AgentOutput(
                agent_id=self.AGENT_ID,
                agent_name=self.AGENT_NAME,
                status=AgentStatus.COMPLETED,
                summary="No financial history available for modeling.",
                risk_score=50.0,
                findings=[self.red_flag(
                    "Cannot build financial model — no financial data",
                    "Financial modeling requires at least 3 years of income statement data.",
                    "Financial history empty",
                    risk_level=RiskClassification.HIGH, confidence=0.99,
                )],
            )

        # ── Historical model ──────────────────────────────────────
        engine = FinancialModelEngine(sector, history)
        kpi_table = engine.build_kpi_table()
        details["historical_kpis"] = kpi_table
        details["sector_kpi_definitions"] = engine.get_sector_kpi_definitions()

        # ── Revenue / Margin trend assessment ────────────────────
        findings.extend(self._assess_kpi_trends(kpi_table, history))

        # ── Forecast ──────────────────────────────────────────────
        forecaster = FinancialForecaster(history, sector)
        all_forecasts = forecaster.forecast_all_scenarios()
        details["forecasts"] = {
            scen: forecaster.to_serializable(years)
            for scen, years in all_forecasts.items()
        }

        # ── LLM modeling commentary ───────────────────────────────
        model_summary = engine.summary_table()
        llm_commentary = self.llm_analyze(
            MODELING_SYSTEM,
            f"Company: {ticker} | Sector: {sector}\n\n"
            f"{model_summary}\n\n"
            f"Base case forecast revenue growth assumptions: "
            + str([f"{fy.assumptions.get('revenue_growth_pct', 0):.1f}%" for fy in all_forecasts['BASE']]) + "\n"
            "Comment on the quality of historical earnings and reasonableness of forecast assumptions.",
            max_tokens=1200,
        )
        details["llm_modeling_commentary"] = llm_commentary

        # Persist
        self.storage.save_json(details, "financial_model.json", "Agent_Outputs")
        self.storage.save_json(details.get("forecasts", {}), "forecasts.json", "Forecasts")

        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=(
                f"Financial model built for {ticker} ({sector}): "
                f"{len(history.available_years)} historical years, "
                f"5-year forecasts in Bear/Base/Bull scenarios. "
                f"Historical revenue CAGR: {self._format_cagr(forecaster._compute_cagr('revenue'))}."
            ),
            findings=findings,
            risk_score=25.0,
            risk_classification=RiskClassification.LOW,
            payload=details,
            sources_used=["yfinance", "financial_history"],
        )

    def _deserialize_history(self, raw: dict, ticker: str) -> FinancialHistory:
        try:
            if isinstance(raw, dict) and "company_ticker" in raw:
                return FinancialHistory(**raw)
        except Exception:
            pass
        # If nested structure (year -> statements dict)
        history = FinancialHistory(company_ticker=ticker, currency="USD")
        from ..models.financials import IncomeStatement, BalanceSheet, CashFlowStatement
        for yr, data in raw.items():
            if not isinstance(data, dict):
                continue
            try:
                is_d = data.get("income_statements")
                if isinstance(is_d, dict) and is_d.get("fiscal_year"):
                    history.income_statements[yr] = IncomeStatement(**is_d)
                elif isinstance(is_d, dict):
                    history.income_statements[yr] = IncomeStatement(fiscal_year=yr, **is_d)
            except Exception:
                pass
            try:
                bs_d = data.get("balance_sheets")
                if isinstance(bs_d, dict):
                    history.balance_sheets[yr] = BalanceSheet(fiscal_year=yr, **(bs_d if "fiscal_year" in bs_d else {**bs_d}))
            except Exception:
                pass
            try:
                cf_d = data.get("cash_flows")
                if isinstance(cf_d, dict):
                    history.cash_flows[yr] = CashFlowStatement(fiscal_year=yr, **(cf_d if "fiscal_year" in cf_d else {**cf_d}))
            except Exception:
                pass
        history.available_years = sorted(set(
            list(history.income_statements.keys()) + list(history.balance_sheets.keys())
        ))
        return history

    def _assess_kpi_trends(self, kpis: dict, history: FinancialHistory) -> list:
        findings = []
        years = sorted(kpis.keys())
        if len(years) < 2:
            return findings

        # Check margin compression
        ebitda_margins = [(yr, kpis[yr].get("ebitda_margin")) for yr in years if kpis[yr].get("ebitda_margin")]
        if len(ebitda_margins) >= 2:
            first_margin = ebitda_margins[0][1]
            last_margin = ebitda_margins[-1][1]
            delta = last_margin - first_margin
            if delta < -5:
                findings.append(self.red_flag(
                    f"EBITDA margin compression: {delta:+.1f}pp over {len(years)} years",
                    "Sustained margin erosion reduces earnings quality and valuation multiples.",
                    evidence=f"EBITDA margins: {[(yr, f'{m:.1f}%') for yr, m in ebitda_margins]}",
                    risk_level=RiskClassification.HIGH, confidence=0.80,
                ))
            elif delta > 5:
                findings.append(self.green_flag(
                    f"Strong margin expansion: +{delta:.1f}pp over {len(years)} years",
                    "Sustained margin improvement indicates operating leverage or pricing power.",
                    evidence=f"EBITDA margins: {[(yr, f'{m:.1f}%') for yr, m in ebitda_margins]}",
                    confidence=0.80,
                ))

        # Revenue growth consistency
        rev_growths = [kpis[yr].get("revenue_growth") for yr in years if kpis[yr].get("revenue_growth") is not None]
        if len(rev_growths) >= 3:
            import statistics
            avg_g = statistics.mean(rev_growths)
            std_g = statistics.stdev(rev_growths) if len(rev_growths) > 1 else 0
            if std_g > 20:
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    f"Highly volatile revenue growth (std dev: {std_g:.1f}%)",
                    "High revenue growth volatility makes forecasting unreliable.",
                    evidence=f"Growth rates: {[f'{g:.1f}%' for g in rev_growths]}",
                    risk_level=RiskClassification.MEDIUM, confidence=0.75,
                ))

        return findings

    @staticmethod
    def _format_cagr(cagr) -> str:
        if cagr is None: return "N/A"
        return f"{cagr*100:.1f}%"
