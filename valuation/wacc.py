"""
WACC Calculator — computes a single, auditable WACC used across all valuation methods.
Every input is an explicit numerical assumption with source documentation.
"""

from __future__ import annotations
from typing import Optional
from ..models.valuation import WACCInputs
from ..core.config import MODELING_CONFIG


# Default equity risk premiums by country/region (Damodaran 2024)
COUNTRY_ERP: dict[str, float] = {
    "US": 4.60, "GB": 4.80, "DE": 4.85, "FR": 4.85, "JP": 5.00,
    "CA": 4.90, "AU": 5.00, "IN": 7.00, "CN": 7.50, "BR": 8.50,
    "ZA": 8.00, "NG": 11.00, "KE": 11.50,
}

RISK_FREE_RATES: dict[str, float] = {
    "US": 4.30, "GB": 4.20, "IN": 7.05, "DE": 2.60, "JP": 0.90,
    "CA": 3.80, "AU": 4.40, "CN": 2.30,
}

COUNTRY_RISK_PREMIUMS: dict[str, float] = {
    "IN": 1.40, "CN": 1.50, "BR": 3.00, "ZA": 2.50,
    "NG": 5.00, "KE": 5.50, "US": 0.0, "GB": 0.0,
}


def compute_wacc(
    country: str,
    sector: str,
    beta: float,
    debt_book_value: float,
    equity_market_cap: float,
    pre_tax_cost_of_debt: Optional[float] = None,
    effective_tax_rate: float = 25.0,
    small_cap_premium: float = 0.0,
    beta_source: str = "yfinance (levered, 5Y monthly)",
    override_risk_free: Optional[float] = None,
    override_erp: Optional[float] = None,
) -> WACCInputs:
    """
    Compute WACC using CAPM for cost of equity and market-value weights.

    Args:
        country: ISO-2 country code
        sector: GICS sector string
        beta: Levered beta
        debt_book_value: Total debt in millions (book value)
        equity_market_cap: Market cap in millions
        pre_tax_cost_of_debt: If None, estimated from country spread
        effective_tax_rate: Effective corporate tax rate (%)
        small_cap_premium: Additional premium for small/illiquid companies (%)
        beta_source: Documentation string for beta source
        override_risk_free: Override risk-free rate
        override_erp: Override equity risk premium
    """
    country = (country or "US").upper()

    rf = override_risk_free or RISK_FREE_RATES.get(country, 4.30)
    erp = override_erp or COUNTRY_ERP.get(country, 5.50)
    crp = COUNTRY_RISK_PREMIUMS.get(country, 0.0)

    # Cost of equity: Ke = Rf + β(ERP + CRP) + SCP
    cost_of_equity = round(rf + beta * (erp + crp) + small_cap_premium, 4)

    # Cost of debt
    if pre_tax_cost_of_debt is None:
        # Estimate from country + sector
        base_spread = {"Financials": 1.5, "Real Estate": 2.0, "Energy": 2.5, "Materials": 2.0}
        spread = base_spread.get(sector, 2.0)
        pre_tax_cost_of_debt = round(rf + spread, 2)

    after_tax_cod = round(pre_tax_cost_of_debt * (1 - effective_tax_rate / 100), 4)

    # Capital structure weights (market value basis)
    total_capital = equity_market_cap + debt_book_value
    if total_capital <= 0:
        equity_weight = 0.8
        debt_weight = 0.2
    else:
        equity_weight = round(equity_market_cap / total_capital, 4)
        debt_weight = round(debt_book_value / total_capital, 4)
        # Ensure they sum exactly to 1.0
        diff = round(equity_weight + debt_weight - 1.0, 6)
        equity_weight = round(equity_weight - diff, 4)

    wacc = round(equity_weight * cost_of_equity + debt_weight * after_tax_cod, 4)
    wacc = max(MODELING_CONFIG.wacc_floor, min(MODELING_CONFIG.wacc_ceiling, wacc))

    return WACCInputs(
        risk_free_rate=rf,
        equity_risk_premium=erp,
        beta=beta,
        beta_source=beta_source,
        cost_of_equity=cost_of_equity,
        pre_tax_cost_of_debt=pre_tax_cost_of_debt,
        tax_rate=effective_tax_rate,
        after_tax_cost_of_debt=after_tax_cod,
        debt_weight=debt_weight,
        equity_weight=equity_weight,
        wacc=wacc,
        country_risk_premium=crp,
        small_cap_premium=small_cap_premium,
    )
