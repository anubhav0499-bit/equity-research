"""
Financial Modeling Engine — builds 5-year historical + 5-year forward financial model.
Generates sector-specific KPIs dynamically. Never uses generic KPI templates.
"""

from __future__ import annotations
import math
from typing import Optional
from loguru import logger

from ..models.financials import (
    IncomeStatement, BalanceSheet, CashFlowStatement, SectorKPI, FinancialHistory
)


SECTOR_KPI_DEFINITIONS: dict[str, dict[str, str]] = {
    "Information Technology": {
        "revenue_per_employee": "Revenue / Headcount (USD)",
        "arr_growth": "Annual Recurring Revenue growth (%)",
        "gross_margin": "Gross profit / Revenue (%)",
        "r_and_d_intensity": "R&D / Revenue (%)",
        "net_revenue_retention": "NRR — expansion revenue from existing customers (%)",
        "rule_of_40": "Revenue growth (%) + EBITDA margin (%) — SaaS health metric",
    },
    "Financials": {
        "nim": "Net Interest Margin (%)",
        "npa_ratio": "Non-Performing Assets / Gross Loans (%)",
        "pcr": "Provision Coverage Ratio (%)",
        "casa_ratio": "CASA / Total Deposits (%)",
        "roe": "Return on Equity (%)",
        "roa": "Return on Assets (%)",
        "roe_tier1": "Tier-1 Capital Ratio (%)",
        "credit_deposit_ratio": "Net Loans / Total Deposits (%)",
    },
    "Health Care": {
        "r_and_d_intensity": "R&D / Revenue (%)",
        "pipeline_coverage": "Pipeline NMEs / Annual Revenue multiple",
        "gross_margin": "Gross margin (%)",
        "patent_cliff_exposure": "Revenue at risk from patent expiry (%)",
        "anda_approvals": "Generic ANDA approvals (count)",
    },
    "Consumer Staples": {
        "volume_growth": "Volume growth ex-price (%)",
        "price_mix": "Price/mix contribution to revenue growth (%)",
        "distribution_reach": "Distribution coverage (towns/outlets)",
        "gross_margin": "Gross margin (%)",
        "ebitda_margin": "EBITDA margin (%)",
        "advertising_intensity": "A&P spend / Revenue (%)",
    },
    "Consumer Discretionary": {
        "same_store_sales_growth": "Like-for-like SSG (%)",
        "gross_margin": "Gross margin (%)",
        "revenue_per_sqft": "Revenue per sq ft / sq meter",
        "inventory_days": "Inventory days outstanding",
        "ebitda_margin": "EBITDA margin (%)",
    },
    "Energy": {
        "production_cost_per_boe": "Production cost per barrel of oil equivalent (USD)",
        "reserve_replacement_ratio": "Reserves added / Production (%)",
        "2p_reserves": "Proven + Probable reserves (mmboe)",
        "refining_margin": "Gross refining margin (USD/bbl)",
        "capex_per_boe": "Capex per BOE produced",
        "ebitda_margin": "EBITDA margin (%)",
    },
    "Industrials": {
        "order_backlog": "Order book / TTM Revenue (x)",
        "capacity_utilisation": "Production capacity utilised (%)",
        "ebitda_margin": "EBITDA margin (%)",
        "working_capital_days": "Working capital as days of revenue",
        "return_on_capital_employed": "EBIT / (Total Assets - Current Liabilities) (%)",
    },
    "Real Estate": {
        "nav_per_share": "Net Asset Value per share",
        "loan_to_value": "Net Debt / Portfolio Value (%)",
        "occupancy_rate": "Occupied / Total leasable area (%)",
        "rental_yield": "Rental income / Portfolio Value (%)",
        "pre_sales": "Pre-sales bookings (units / value)",
        "collections_efficiency": "Collections / Pre-sales (%)",
    },
    "Telecom": {
        "arpu": "Average Revenue Per User (monthly USD)",
        "subscriber_growth": "Subscriber net additions (%)",
        "churn_rate": "Monthly churn (%)",
        "ebitda_margin": "EBITDA margin (%)",
        "capex_intensity": "Capex / Revenue (%)",
        "spectrum_amortisation": "Annual spectrum charge / Revenue (%)",
    },
    "Materials": {
        "production_volume": "Production volume (mt / tonnes)",
        "realisation_per_tonne": "Net realisation per tonne (USD)",
        "ebitda_per_tonne": "EBITDA per tonne (USD)",
        "coking_coal_cost": "Coking coal input cost / tonne",
        "utilisation_rate": "Capacity utilisation (%)",
    },
}

DEFAULT_KPIS = {
    "gross_margin": "Gross profit / Revenue (%)",
    "ebitda_margin": "EBITDA / Revenue (%)",
    "ebit_margin": "EBIT / Revenue (%)",
    "net_margin": "Net income / Revenue (%)",
    "roe": "Net income / Shareholders equity (%)",
    "roa": "Net income / Total assets (%)",
    "asset_turnover": "Revenue / Total assets",
    "interest_coverage": "EBIT / Interest expense (x)",
    "debt_to_equity": "Total debt / Total equity",
    "current_ratio": "Current assets / Current liabilities",
    "fcf_margin": "Free cash flow / Revenue (%)",
    "capex_intensity": "Capex / Revenue (%)",
    "revenue_growth": "YoY revenue growth (%)",
    "ebitda_growth": "YoY EBITDA growth (%)",
    "eps_growth": "YoY EPS growth (%)",
}


class FinancialModelEngine:
    """
    Builds historical financial model and dynamic KPI framework.
    Sector-specific KPIs are generated dynamically — never generic templates.
    """

    def __init__(self, sector: str, history: FinancialHistory):
        self.sector = sector
        self.history = history
        self.kpi_defs = {
            **DEFAULT_KPIS,
            **SECTOR_KPI_DEFINITIONS.get(sector, {}),
        }

    def build_kpi_table(self) -> dict[str, dict[str, float]]:
        """Compute all KPIs for each available year. Returns {year: {kpi_name: value}}."""
        result: dict[str, dict[str, float]] = {}
        years = self.history.years_sorted()

        for year in years:
            is_ = self.history.get_income_statement(year)
            bs_ = self.history.get_balance_sheet(year)
            cf_ = self.history.get_cash_flow(year)
            kpis: dict[str, float] = {}

            if is_:
                if is_.revenue and is_.revenue > 0:
                    if is_.gross_profit:
                        kpis["gross_margin"] = round(is_.gross_profit / is_.revenue * 100, 2)
                    if is_.ebitda:
                        kpis["ebitda_margin"] = round(is_.ebitda / is_.revenue * 100, 2)
                    if is_.ebit:
                        kpis["ebit_margin"] = round(is_.ebit / is_.revenue * 100, 2)
                    if is_.net_income:
                        kpis["net_margin"] = round(is_.net_income / is_.revenue * 100, 2)
                    if is_.rd_expense:
                        kpis["r_and_d_intensity"] = round(is_.rd_expense / is_.revenue * 100, 2)
                    if is_.sga:
                        kpis["advertising_intensity"] = round(is_.sga / is_.revenue * 100, 2)
                if is_.ebit and is_.interest_expense and is_.interest_expense > 0:
                    kpis["interest_coverage"] = round(is_.ebit / is_.interest_expense, 2)

            if bs_:
                if bs_.total_equity and bs_.total_equity > 0 and is_ and is_.net_income:
                    kpis["roe"] = round(is_.net_income / bs_.total_equity * 100, 2)
                if bs_.total_assets and bs_.total_assets > 0:
                    if is_ and is_.net_income:
                        kpis["roa"] = round(is_.net_income / bs_.total_assets * 100, 2)
                    if is_ and is_.revenue:
                        kpis["asset_turnover"] = round(is_.revenue / bs_.total_assets, 3)
                    if is_ and is_.ebit:
                        cl_ = bs_.total_current_liabilities or 0
                        kpis["roce"] = round(is_.ebit / (bs_.total_assets - cl_) * 100, 2)
                if bs_.current_ratio:
                    kpis["current_ratio"] = round(bs_.current_ratio, 2)
                if bs_.debt_to_equity:
                    kpis["debt_to_equity"] = round(bs_.debt_to_equity, 2)
                if bs_.accounts_receivable and is_ and is_.revenue and is_.revenue > 0:
                    kpis["dso"] = round(bs_.accounts_receivable / is_.revenue * 365, 1)
                if bs_.inventory and is_ and is_.cogs and is_.cogs > 0:
                    kpis["inventory_days"] = round(bs_.inventory / is_.cogs * 365, 1)
                if bs_.accounts_payable and is_ and is_.cogs and is_.cogs > 0:
                    kpis["dpo"] = round(bs_.accounts_payable / is_.cogs * 365, 1)

            if cf_:
                if is_ and is_.revenue and is_.revenue > 0:
                    if cf_.fcf:
                        kpis["fcf_margin"] = round(cf_.fcf / is_.revenue * 100, 2)
                    if cf_.capex:
                        kpis["capex_intensity"] = round(abs(cf_.capex) / is_.revenue * 100, 2)
                if cf_.cash_conversion_ratio:
                    kpis["cash_conversion_ratio"] = round(cf_.cash_conversion_ratio, 3)

            # Rule of 40 (SaaS / IT)
            if "gross_margin" in kpis and is_ and is_.revenue_growth_yoy:
                kpis["rule_of_40"] = round((is_.revenue_growth_yoy or 0) + kpis.get("ebitda_margin", 0), 1)

            result[year] = kpis

        # Compute YoY growth rates
        sorted_years = sorted(result.keys())
        for i in range(1, len(sorted_years)):
            yr, prev = sorted_years[i], sorted_years[i-1]
            is_c = self.history.get_income_statement(yr)
            is_p = self.history.get_income_statement(prev)
            if is_c and is_p:
                if is_c.revenue and is_p.revenue and is_p.revenue > 0:
                    result[yr]["revenue_growth"] = round((is_c.revenue / is_p.revenue - 1) * 100, 2)
                if is_c.ebitda and is_p.ebitda and is_p.ebitda > 0:
                    result[yr]["ebitda_growth"] = round((is_c.ebitda / is_p.ebitda - 1) * 100, 2)
                if is_c.eps_basic and is_p.eps_basic and is_p.eps_basic > 0:
                    result[yr]["eps_growth"] = round((is_c.eps_basic / is_p.eps_basic - 1) * 100, 2)

        return result

    def compute_cagr(self, metric: str, years: int = 5) -> Optional[float]:
        """Compute CAGR for a given metric over the specified years."""
        yr_sorted = self.history.years_sorted()
        if len(yr_sorted) < 2:
            return None
        start_yr = yr_sorted[0]
        end_yr = yr_sorted[min(years, len(yr_sorted) - 1)]
        n = yr_sorted.index(end_yr) - yr_sorted.index(start_yr)
        if n <= 0:
            return None

        def get_val(year: str) -> Optional[float]:
            is_ = self.history.get_income_statement(year)
            if is_:
                return getattr(is_, metric, None)
            return None

        v0 = get_val(start_yr)
        vn = get_val(end_yr)
        if v0 and vn and v0 > 0 and vn > 0:
            try:
                return round((vn / v0) ** (1 / n) - 1, 4)
            except Exception:
                pass
        return None

    def get_sector_kpi_definitions(self) -> dict[str, str]:
        return self.kpi_defs

    def summary_table(self) -> str:
        """Human-readable summary for LLM prompts."""
        kpis = self.build_kpi_table()
        lines = [f"Financial Model — {self.sector}"]
        years = sorted(kpis.keys())
        key_metrics = ["revenue_growth", "ebitda_margin", "net_margin", "roe",
                       "debt_to_equity", "fcf_margin", "interest_coverage"]
        for metric in key_metrics:
            row = f"  {metric:<30}"
            for yr in years[-5:]:
                val = kpis.get(yr, {}).get(metric)
                row += f"  {yr}: {val:>7.1f}" if val is not None else f"  {yr}:     N/A"
            lines.append(row)
        return "\n".join(lines)
