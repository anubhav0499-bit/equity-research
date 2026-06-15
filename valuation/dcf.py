"""
DCF Valuation Engine — Discounted Cash Flow model using FCFF approach.
One WACC shared across all scenarios.
"""

from __future__ import annotations
from ..models.valuation import DCFModel, DCFYear, WACCInputs, Scenario
from ..modeling.forecaster import ForecastYear


def build_dcf(
    forecast_years: list[ForecastYear],
    wacc_inputs: WACCInputs,
    scenario: Scenario,
    net_debt: float,
    shares_outstanding: float,
    current_price: float = 0.0,
    terminal_growth_rate: float = 3.0,
) -> DCFModel:
    """
    Build a DCF model from explicit forecast years.

    FCFF = EBIT(1-t) + D&A - ΔNWC - Capex
    Terminal Value = FCFFn * (1+g) / (WACC - g)   [Gordon Growth]
    Equity Value = PV(FCFFs) + PV(TV) - Net Debt
    """
    wacc = wacc_inputs.wacc / 100  # Convert to decimal
    tgr = terminal_growth_rate / 100

    if wacc <= tgr:
        # Prevent division by zero / negative denominator
        tgr = wacc * 0.5

    dcf_years: list[DCFYear] = []
    sum_pv_fcff = 0.0

    for i, fy in enumerate(forecast_years):
        t = i + 1
        tax_rate = fy.assumptions.get("tax_rate_pct", 25.0) / 100
        nopat = fy.ebit * (1 - tax_rate)
        discount_factor = 1 / ((1 + wacc) ** t)
        pv_fcff = fy.fcf * discount_factor
        sum_pv_fcff += pv_fcff

        dcf_year = DCFYear(
            year=fy.year,
            scenario=scenario,
            revenue=fy.revenue,
            ebit_margin=fy.ebit_margin,
            ebit=fy.ebit,
            tax_rate=tax_rate * 100,
            nopat=round(nopat, 2),
            depreciation=fy.da,
            capex=fy.capex,
            change_in_nwc=0.0,  # Included in FCF calc
            fcff=fy.fcf,
            discount_factor=round(discount_factor, 6),
            pv_fcff=round(pv_fcff, 2),
        )
        dcf_years.append(dcf_year)

    # Terminal Value
    last_fcff = dcf_years[-1].fcff if dcf_years else 0.0
    terminal_value = last_fcff * (1 + tgr) / (wacc - tgr)
    n = len(dcf_years)
    pv_terminal_value = terminal_value / ((1 + wacc) ** n)

    enterprise_value = round(sum_pv_fcff + pv_terminal_value, 2)
    equity_value = enterprise_value - net_debt
    intrinsic_value = round(equity_value / shares_outstanding, 2) if shares_outstanding > 0 else 0.0

    return DCFModel(
        scenario=scenario,
        wacc_inputs=wacc_inputs,
        terminal_growth_rate=terminal_growth_rate,
        terminal_value=round(terminal_value, 2),
        pv_terminal_value=round(pv_terminal_value, 2),
        sum_pv_fcff=round(sum_pv_fcff, 2),
        enterprise_value=enterprise_value,
        net_debt=net_debt,
        equity_value=round(equity_value, 2),
        shares_outstanding=shares_outstanding,
        intrinsic_value_per_share=intrinsic_value,
        current_price=current_price if current_price > 0 else None,
        forecast_years=dcf_years,
        key_assumptions={
            "wacc_pct": wacc_inputs.wacc,
            "terminal_growth_pct": terminal_growth_rate,
            "terminal_value_pct_of_ev": round(pv_terminal_value / enterprise_value * 100, 1) if enterprise_value > 0 else 0,
            "cost_of_equity_pct": wacc_inputs.cost_of_equity,
            "beta": wacc_inputs.beta,
            "net_debt": net_debt,
        },
    )
