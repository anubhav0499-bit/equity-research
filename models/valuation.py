"""
Valuation schemas — DCF, Relative Valuation, Historical Multiples, SOTP.
A single WACC is shared across all methodologies.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class Scenario(str, Enum):
    BEAR = "BEAR"
    BASE = "BASE"
    BULL = "BULL"


class ValuationMethod(str, Enum):
    DCF = "DCF"
    RELATIVE = "RELATIVE"
    HISTORICAL_MULTIPLES = "HISTORICAL_MULTIPLES"
    SOTP = "SOTP"
    DIVIDEND_DISCOUNT = "DDM"
    ASSET_BASED = "ASSET_BASED"


class WACCInputs(BaseModel):
    """
    All assumptions are numerical and auditable.
    Single WACC used across all valuation methods.
    """
    risk_free_rate: float = Field(..., description="10Y government bond yield (%)")
    equity_risk_premium: float = Field(..., description="Market equity risk premium (%)")
    beta: float = Field(..., gt=0)
    beta_source: str = Field(..., description="Source of beta estimate")
    cost_of_equity: float = Field(..., description="Re = Rf + β × ERP (%)")
    pre_tax_cost_of_debt: float = Field(..., description="Weighted average borrowing cost (%)")
    tax_rate: float = Field(..., ge=0, le=50, description="Effective corporate tax rate (%)")
    after_tax_cost_of_debt: float = Field(..., description="Kd × (1 - T) (%)")
    debt_weight: float = Field(..., ge=0, le=1, description="D / (D + E) at market values")
    equity_weight: float = Field(..., ge=0, le=1, description="E / (D + E) at market values")
    wacc: float = Field(..., description="WACC = We × Ke + Wd × Kd(1-T) (%)")
    country_risk_premium: float = Field(0.0, description="Additional CRP for EM companies (%)")
    small_cap_premium: float = Field(0.0, description="Illiquidity / size premium (%)")

    @model_validator(mode="after")
    def validate_weights(self) -> "WACCInputs":
        total = round(self.debt_weight + self.equity_weight, 4)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Debt + equity weights must sum to 1.0, got {total}")
        return self

    model_config = {"frozen": True}


class DCFYear(BaseModel):
    year: int
    scenario: Scenario
    revenue: float
    ebit_margin: float
    ebit: float
    tax_rate: float
    nopat: float
    depreciation: float
    capex: float
    change_in_nwc: float
    fcff: float
    discount_factor: float
    pv_fcff: float


class DCFModel(BaseModel):
    scenario: Scenario
    wacc_inputs: WACCInputs
    terminal_growth_rate: float = Field(..., ge=-2, le=10)
    terminal_value: float
    pv_terminal_value: float
    sum_pv_fcff: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    shares_outstanding: float
    intrinsic_value_per_share: float
    current_price: Optional[float] = None
    upside_downside_pct: Optional[float] = None
    forecast_years: list[DCFYear] = Field(default_factory=list)
    key_assumptions: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def compute_upside(self) -> "DCFModel":
        if self.current_price and self.current_price > 0:
            pct = (self.intrinsic_value_per_share / self.current_price - 1) * 100
            object.__setattr__(self, "upside_downside_pct", round(pct, 1))
        return self


class PeerMultiple(BaseModel):
    company: str
    ticker: str
    ev_ebitda: Optional[float] = None
    ev_ebit: Optional[float] = None
    pe: Optional[float] = None
    pb: Optional[float] = None
    ps: Optional[float] = None
    fcf_yield: Optional[float] = None
    source: Optional[str] = None


class RelativeValuation(BaseModel):
    scenario: Scenario
    wacc_inputs: WACCInputs
    peer_group: list[PeerMultiple] = Field(default_factory=list)
    target_ev_ebitda: Optional[float] = None
    target_pe: Optional[float] = None
    target_pb: Optional[float] = None
    target_ps: Optional[float] = None

    # Applied multiples
    applied_ev_ebitda: Optional[float] = None
    implied_ev_from_ebitda: Optional[float] = None
    applied_pe: Optional[float] = None
    implied_equity_from_pe: Optional[float] = None

    net_debt: float = 0.0
    shares_outstanding: float
    value_per_share_ev_ebitda: Optional[float] = None
    value_per_share_pe: Optional[float] = None
    blended_value_per_share: Optional[float] = None

    current_price: Optional[float] = None
    upside_downside_pct: Optional[float] = None
    methodology_note: str = ""


class HistoricalMultipleYear(BaseModel):
    year: str
    ev_ebitda: Optional[float] = None
    pe: Optional[float] = None
    pb: Optional[float] = None
    ps: Optional[float] = None


class HistoricalMultiples(BaseModel):
    scenario: Scenario
    history: list[HistoricalMultipleYear] = Field(default_factory=list)
    mean_ev_ebitda: Optional[float] = None
    mean_pe: Optional[float] = None
    mean_pb: Optional[float] = None
    applied_ev_ebitda: Optional[float] = None
    applied_pe: Optional[float] = None
    net_debt: float = 0.0
    shares_outstanding: float
    value_per_share_ev_ebitda: Optional[float] = None
    value_per_share_pe: Optional[float] = None
    blended_value_per_share: Optional[float] = None
    current_price: Optional[float] = None
    upside_downside_pct: Optional[float] = None


class SOTPSegment(BaseModel):
    name: str
    description: str
    revenue: float
    ebitda: float
    ebitda_margin: float
    valuation_method: str
    applied_multiple: float
    enterprise_value: float
    notes: str = ""


class SumOfPartsModel(BaseModel):
    scenario: Scenario
    wacc_inputs: WACCInputs
    segments: list[SOTPSegment] = Field(default_factory=list)
    corporate_overhead_deduction: float = 0.0
    total_enterprise_value: float
    net_debt: float = 0.0
    minority_interest: float = 0.0
    equity_value: float
    shares_outstanding: float
    value_per_share: float
    current_price: Optional[float] = None
    upside_downside_pct: Optional[float] = None


class ScenarioValuation(BaseModel):
    scenario: Scenario
    dcf: Optional[DCFModel] = None
    relative: Optional[RelativeValuation] = None
    historical: Optional[HistoricalMultiples] = None
    sotp: Optional[SumOfPartsModel] = None
    blended_value_per_share: Optional[float] = None
    weight_dcf: float = Field(0.40, ge=0, le=1)
    weight_relative: float = Field(0.35, ge=0, le=1)
    weight_historical: float = Field(0.15, ge=0, le=1)
    weight_sotp: float = Field(0.10, ge=0, le=1)

    @model_validator(mode="after")
    def compute_blended(self) -> "ScenarioValuation":
        values, weights = [], []
        if self.dcf and self.dcf.intrinsic_value_per_share:
            values.append(self.dcf.intrinsic_value_per_share)
            weights.append(self.weight_dcf)
        if self.relative and self.relative.blended_value_per_share:
            values.append(self.relative.blended_value_per_share)
            weights.append(self.weight_relative)
        if self.historical and self.historical.blended_value_per_share:
            values.append(self.historical.blended_value_per_share)
            weights.append(self.weight_historical)
        if self.sotp and self.sotp.value_per_share:
            values.append(self.sotp.value_per_share)
            weights.append(self.weight_sotp)
        if values and sum(weights) > 0:
            blended = sum(v * w for v, w in zip(values, weights)) / sum(weights)
            object.__setattr__(self, "blended_value_per_share", round(blended, 2))
        return self


class ValuationSummary(BaseModel):
    """Final valuation output — one per research run, covering all three scenarios."""
    ticker: str
    current_price: float
    currency: str
    wacc_inputs: WACCInputs  # Single WACC across all scenarios and methods

    bear_case: ScenarioValuation
    base_case: ScenarioValuation
    bull_case: ScenarioValuation

    bear_price: Optional[float] = None
    base_price: Optional[float] = None
    bull_price: Optional[float] = None

    recommendation: str = ""
    target_price: Optional[float] = None
    upside_pct: Optional[float] = None
    conviction: str = ""

    @model_validator(mode="after")
    def extract_prices(self) -> "ValuationSummary":
        if self.bear_case.blended_value_per_share:
            object.__setattr__(self, "bear_price", self.bear_case.blended_value_per_share)
        if self.base_case.blended_value_per_share:
            object.__setattr__(self, "base_price", self.base_case.blended_value_per_share)
        if self.bull_case.blended_value_per_share:
            object.__setattr__(self, "bull_price", self.bull_case.blended_value_per_share)
        if self.base_price and self.current_price > 0:
            upside = (self.base_price / self.current_price - 1) * 100
            object.__setattr__(self, "upside_pct", round(upside, 1))
            if self.target_price is None:
                object.__setattr__(self, "target_price", self.base_price)
        return self


class ScenarioSet(BaseModel):
    """Scenario analysis container — macro + company-specific levers."""
    scenario: Scenario
    revenue_growth_assumption: float
    ebitda_margin_assumption: float
    wacc_delta: float = 0.0
    terminal_growth_rate: float = 3.0
    macro_assumptions: dict[str, float] = Field(default_factory=dict)
    company_assumptions: dict[str, float] = Field(default_factory=dict)
    narrative: str = ""
    implied_value_per_share: Optional[float] = None
