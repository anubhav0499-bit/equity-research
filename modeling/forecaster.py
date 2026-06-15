"""
Forecasting Engine — generates 5-year financial projections.
Scenario-based: Bear / Base / Bull. All assumptions are numerical and auditable.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from ..models.financials import FinancialHistory, IncomeStatement, BalanceSheet, CashFlowStatement
from ..models.valuation import Scenario


@dataclass
class ForecastAssumptions:
    scenario: Scenario
    revenue_growth_rates: list[float]           # One per forecast year
    ebitda_margin: list[float]                  # Target EBITDA margin per year
    da_rate: list[float]                        # D&A as % of revenue
    tax_rate: float = 25.0
    capex_intensity: list[float] = field(default_factory=list)  # Capex / revenue
    nwc_change_rate: float = 1.5                # NWC change as % of revenue change
    terminal_growth_rate: float = 3.0
    interest_rate_on_debt: float = 7.0


@dataclass
class ForecastYear:
    year: int
    scenario: str
    revenue: float
    ebitda: float
    ebitda_margin: float
    da: float
    ebit: float
    ebit_margin: float
    interest_expense: float
    ebt: float
    tax: float
    net_income: float
    net_margin: float
    cfo: float
    capex: float
    fcf: float
    capex_intensity: float
    shares: float
    eps: float
    assumptions: dict


class FinancialForecaster:
    """
    Generates 5-year financial forecasts under three scenarios.
    All drivers are explicit numerical assumptions — no black-box estimates.
    """

    def __init__(self, history: FinancialHistory, sector: str = ""):
        self.history = history
        self.sector = sector

    def build_base_assumptions(self) -> dict[str, ForecastAssumptions]:
        """Build scenario assumptions anchored to historical trends."""
        rev_cagr = self._compute_cagr("revenue", years=3)
        avg_ebitda_margin = self._avg_margin("ebitda", "revenue")
        avg_capex = self._avg_ratio("capex", "revenue")
        avg_tax = self._avg_tax_rate()

        # Adjust for sector norms
        sector_adj = self._sector_adjustments()
        ebitda_adj = sector_adj.get("ebitda_adj", 0)

        base_rev_growth = max(0.02, (rev_cagr or 0.08))

        assumptions = {
            Scenario.BASE: ForecastAssumptions(
                scenario=Scenario.BASE,
                revenue_growth_rates=[base_rev_growth] * 5,
                ebitda_margin=[(avg_ebitda_margin + ebitda_adj)] * 5,
                da_rate=[self._avg_ratio("depreciation", "revenue")] * 5,
                tax_rate=avg_tax or 25.0,
                capex_intensity=[avg_capex] * 5,
                terminal_growth_rate=3.0,
            ),
            Scenario.BULL: ForecastAssumptions(
                scenario=Scenario.BULL,
                revenue_growth_rates=[min(base_rev_growth * 1.5, 0.35)] * 3 +
                                     [min(base_rev_growth * 1.25, 0.25)] * 2,
                ebitda_margin=[(avg_ebitda_margin + ebitda_adj + 2.0)] * 5,
                da_rate=[self._avg_ratio("depreciation", "revenue")] * 5,
                tax_rate=max(15.0, (avg_tax or 25.0) - 2.0),
                capex_intensity=[avg_capex] * 5,
                terminal_growth_rate=4.0,
            ),
            Scenario.BEAR: ForecastAssumptions(
                scenario=Scenario.BEAR,
                revenue_growth_rates=[max(-0.05, base_rev_growth * 0.4)] * 5,
                ebitda_margin=[max(5.0, avg_ebitda_margin + ebitda_adj - 4.0)] * 5,
                da_rate=[self._avg_ratio("depreciation", "revenue") * 1.1] * 5,
                tax_rate=min(35.0, (avg_tax or 25.0) + 2.0),
                capex_intensity=[avg_capex * 1.2] * 5,
                terminal_growth_rate=1.5,
            ),
        }
        return assumptions

    def forecast(self, assumptions: ForecastAssumptions, base_year: Optional[str] = None) -> list[ForecastYear]:
        """Project financials for 5 years given explicit assumptions."""
        if not base_year:
            base_year = max(self.history.available_years) if self.history.available_years else None
        if not base_year:
            return []

        base_is = self.history.get_income_statement(base_year)
        base_bs = self.history.get_balance_sheet(base_year)
        base_cf = self.history.get_cash_flow(base_year)

        base_revenue = (base_is.revenue if base_is else None) or 1000.0
        base_debt = (base_bs.net_debt if base_bs else None) or 0.0
        base_shares = (base_is.shares_diluted if base_is else None) or 100.0
        base_interest = base_debt * (assumptions.interest_rate_on_debt / 100) if base_debt > 0 else 0.0

        years_out = range(1, 6)
        forecast_start = int(base_year) + 1

        results = []
        prev_revenue = base_revenue
        prev_nwc = (
            (base_bs.working_capital if base_bs else None) or base_revenue * 0.10
        )

        for i, yr_offset in enumerate(years_out):
            year = forecast_start + yr_offset - 1
            g = assumptions.revenue_growth_rates[min(i, len(assumptions.revenue_growth_rates) - 1)]
            em = assumptions.ebitda_margin[min(i, len(assumptions.ebitda_margin) - 1)]
            da_rate = assumptions.da_rate[min(i, len(assumptions.da_rate) - 1)]
            capex_rate = assumptions.capex_intensity[min(i, len(assumptions.capex_intensity) - 1)] if assumptions.capex_intensity else 0.05
            tax_rate = assumptions.tax_rate

            revenue = round(prev_revenue * (1 + g), 2)
            ebitda = round(revenue * em / 100, 2)
            da = round(revenue * da_rate / 100, 2)
            ebit = round(ebitda - da, 2)
            ebt = round(ebit - base_interest, 2)
            tax = round(max(0, ebt * tax_rate / 100), 2)
            net_income = round(ebt - tax, 2)
            capex = round(revenue * capex_rate / 100, 2)
            nwc = round(revenue * 0.10, 2)
            delta_nwc = nwc - prev_nwc
            cfo = round(net_income + da - delta_nwc, 2)
            fcf = round(cfo - capex, 2)
            eps = round(net_income / base_shares, 4) if base_shares else 0.0

            results.append(ForecastYear(
                year=year,
                scenario=assumptions.scenario.value,
                revenue=revenue,
                ebitda=ebitda,
                ebitda_margin=round(em, 2),
                da=da,
                ebit=ebit,
                ebit_margin=round(ebit / revenue * 100, 2) if revenue else 0,
                interest_expense=round(base_interest, 2),
                ebt=ebt,
                tax=tax,
                net_income=net_income,
                net_margin=round(net_income / revenue * 100, 2) if revenue else 0,
                cfo=cfo,
                capex=capex,
                fcf=fcf,
                capex_intensity=round(capex_rate, 2),
                shares=base_shares,
                eps=eps,
                assumptions={
                    "revenue_growth_pct": round(g * 100, 2),
                    "ebitda_margin_pct": round(em, 2),
                    "tax_rate_pct": round(tax_rate, 2),
                    "capex_intensity_pct": round(capex_rate, 2),
                },
            ))
            prev_revenue = revenue
            prev_nwc = nwc

        return results

    def forecast_all_scenarios(self) -> dict[str, list[ForecastYear]]:
        assumptions = self.build_base_assumptions()
        return {
            "BEAR": self.forecast(assumptions[Scenario.BEAR]),
            "BASE": self.forecast(assumptions[Scenario.BASE]),
            "BULL": self.forecast(assumptions[Scenario.BULL]),
        }

    def to_serializable(self, forecast_years: list[ForecastYear]) -> list[dict]:
        return [vars(fy) for fy in forecast_years]

    # ── Historical computation helpers ───────────────────────────

    def _compute_cagr(self, field: str, years: int = 5) -> Optional[float]:
        yr_sorted = sorted(self.history.available_years)
        if len(yr_sorted) < 2:
            return None
        end_yr = yr_sorted[-1]
        start_idx = max(0, len(yr_sorted) - 1 - years)
        start_yr = yr_sorted[start_idx]
        n = int(end_yr) - int(start_yr)
        if n <= 0:
            return None

        def gv(yr):
            is_ = self.history.get_income_statement(yr)
            if is_ and hasattr(is_, field):
                return getattr(is_, field)
            cf_ = self.history.get_cash_flow(yr)
            if cf_ and hasattr(cf_, field):
                return getattr(cf_, field)
            return None

        v0, vn = gv(start_yr), gv(end_yr)
        if v0 and vn and v0 > 0 and vn > 0:
            try:
                return (vn / v0) ** (1 / n) - 1
            except Exception:
                return None
        return None

    def _avg_margin(self, numerator: str, denominator: str) -> float:
        ratios = []
        for yr in self.history.available_years:
            is_ = self.history.get_income_statement(yr)
            if is_:
                n = getattr(is_, numerator, None)
                d = getattr(is_, denominator, None)
                if n and d and d > 0:
                    ratios.append(n / d * 100)
        return round(sum(ratios) / len(ratios), 2) if ratios else 20.0

    def _avg_ratio(self, numerator: str, denominator: str) -> float:
        ratios = []
        for yr in self.history.available_years:
            is_ = self.history.get_income_statement(yr)
            cf_ = self.history.get_cash_flow(yr)
            n, d = None, None
            for obj in [cf_, is_]:
                if obj and hasattr(obj, numerator):
                    n = getattr(obj, numerator)
                    if n:
                        break
            if is_ and hasattr(is_, denominator):
                d = getattr(is_, denominator)
            if n and d and d > 0:
                ratios.append(abs(n) / d * 100)
        return round(sum(ratios) / len(ratios), 2) if ratios else 5.0

    def _avg_tax_rate(self) -> float:
        rates = []
        for yr in self.history.available_years:
            is_ = self.history.get_income_statement(yr)
            if is_ and is_.effective_tax_rate:
                rates.append(is_.effective_tax_rate)
            elif is_ and is_.income_tax and is_.ebt and is_.ebt > 0:
                rates.append(is_.income_tax / is_.ebt * 100)
        return round(sum(rates) / len(rates), 2) if rates else 25.0

    def _sector_adjustments(self) -> dict:
        adj_map = {
            "Information Technology": {"ebitda_adj": 2.0},
            "Financials": {"ebitda_adj": 0.0},
            "Health Care": {"ebitda_adj": 1.0},
            "Consumer Staples": {"ebitda_adj": 0.5},
            "Energy": {"ebitda_adj": -2.0},
        }
        return adj_map.get(self.sector, {"ebitda_adj": 0.0})
