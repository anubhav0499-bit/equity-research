"""
Financial statement schemas — all monetary values in company reporting currency
(millions unless noted). Every field is Optional to handle partial data gracefully.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field, model_validator


class DataQuality(str, Enum):
    AUDITED = "AUDITED"
    UNAUDITED = "UNAUDITED"
    ESTIMATED = "ESTIMATED"
    SCRAPED = "SCRAPED"
    UNKNOWN = "UNKNOWN"


class FinancialSource(BaseModel):
    url: str
    document_type: str
    fiscal_year: str
    page_number: Optional[int] = None
    extracted_at: str
    confidence: float = Field(1.0, ge=0.0, le=1.0)

    model_config = {"frozen": True}


class IncomeStatement(BaseModel):
    fiscal_year: str
    period_end: Optional[str] = None
    currency: str = "USD"
    unit: str = "millions"
    quality: DataQuality = DataQuality.UNKNOWN
    source: Optional[FinancialSource] = None

    # Revenue
    revenue: Optional[float] = None
    other_income: Optional[float] = None
    total_income: Optional[float] = None

    # Costs
    cogs: Optional[float] = None
    gross_profit: Optional[float] = None
    gross_margin: Optional[float] = None

    # Opex
    sga: Optional[float] = None
    rd_expense: Optional[float] = None
    other_opex: Optional[float] = None
    total_opex: Optional[float] = None

    # Operating
    ebitda: Optional[float] = None
    ebitda_margin: Optional[float] = None
    depreciation: Optional[float] = None
    amortization: Optional[float] = None
    ebit: Optional[float] = None
    ebit_margin: Optional[float] = None

    # Below EBIT
    interest_expense: Optional[float] = None
    interest_income: Optional[float] = None
    other_non_operating: Optional[float] = None
    ebt: Optional[float] = None

    # Tax
    income_tax: Optional[float] = None
    effective_tax_rate: Optional[float] = None

    # Earnings
    net_income: Optional[float] = None
    net_margin: Optional[float] = None
    minority_interest: Optional[float] = None
    pat: Optional[float] = None  # Profit after tax (Indian terminology)

    # Per share
    eps_basic: Optional[float] = None
    eps_diluted: Optional[float] = None
    dps: Optional[float] = None
    shares_basic: Optional[float] = None
    shares_diluted: Optional[float] = None

    # Derived (computed post-init)
    revenue_growth_yoy: Optional[float] = None
    net_income_growth_yoy: Optional[float] = None

    @model_validator(mode="after")
    def compute_margins(self) -> "IncomeStatement":
        if self.revenue and self.revenue > 0:
            if self.gross_profit and self.gross_margin is None:
                object.__setattr__(self, "gross_margin", self.gross_profit / self.revenue * 100)
            if self.ebitda and self.ebitda_margin is None:
                object.__setattr__(self, "ebitda_margin", self.ebitda / self.revenue * 100)
            if self.ebit and self.ebit_margin is None:
                object.__setattr__(self, "ebit_margin", self.ebit / self.revenue * 100)
            if self.net_income and self.net_margin is None:
                object.__setattr__(self, "net_margin", self.net_income / self.revenue * 100)
        return self


class BalanceSheet(BaseModel):
    fiscal_year: str
    period_end: Optional[str] = None
    currency: str = "USD"
    unit: str = "millions"
    quality: DataQuality = DataQuality.UNKNOWN
    source: Optional[FinancialSource] = None

    # Current assets
    cash_and_equivalents: Optional[float] = None
    short_term_investments: Optional[float] = None
    accounts_receivable: Optional[float] = None
    inventory: Optional[float] = None
    other_current_assets: Optional[float] = None
    total_current_assets: Optional[float] = None

    # Non-current assets
    gross_ppe: Optional[float] = None
    accumulated_depreciation: Optional[float] = None
    net_ppe: Optional[float] = None
    goodwill: Optional[float] = None
    intangible_assets: Optional[float] = None
    long_term_investments: Optional[float] = None
    deferred_tax_assets: Optional[float] = None
    other_non_current_assets: Optional[float] = None
    total_non_current_assets: Optional[float] = None
    total_assets: Optional[float] = None

    # Current liabilities
    accounts_payable: Optional[float] = None
    short_term_debt: Optional[float] = None
    current_portion_lt_debt: Optional[float] = None
    accrued_liabilities: Optional[float] = None
    deferred_revenue_current: Optional[float] = None
    other_current_liabilities: Optional[float] = None
    total_current_liabilities: Optional[float] = None

    # Non-current liabilities
    long_term_debt: Optional[float] = None
    deferred_tax_liabilities: Optional[float] = None
    other_non_current_liabilities: Optional[float] = None
    total_non_current_liabilities: Optional[float] = None
    total_liabilities: Optional[float] = None

    # Equity
    common_stock: Optional[float] = None
    additional_paid_in_capital: Optional[float] = None
    retained_earnings: Optional[float] = None
    other_equity: Optional[float] = None
    minority_interest_equity: Optional[float] = None
    total_equity: Optional[float] = None
    total_liabilities_and_equity: Optional[float] = None

    # Derived
    working_capital: Optional[float] = None
    net_debt: Optional[float] = None
    book_value_per_share: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None

    @model_validator(mode="after")
    def compute_derived(self) -> "BalanceSheet":
        if self.total_current_assets and self.total_current_liabilities:
            wc = self.total_current_assets - self.total_current_liabilities
            object.__setattr__(self, "working_capital", wc)
            cr = self.total_current_assets / self.total_current_liabilities
            object.__setattr__(self, "current_ratio", cr)
        total_debt = (self.short_term_debt or 0) + (self.current_portion_lt_debt or 0) + (self.long_term_debt or 0)
        cash = self.cash_and_equivalents or 0
        if total_debt or cash:
            object.__setattr__(self, "net_debt", total_debt - cash)
        if self.total_equity and self.total_equity > 0 and total_debt:
            object.__setattr__(self, "debt_to_equity", total_debt / self.total_equity)
        return self


class CashFlowStatement(BaseModel):
    fiscal_year: str
    period_end: Optional[str] = None
    currency: str = "USD"
    unit: str = "millions"
    quality: DataQuality = DataQuality.UNKNOWN
    source: Optional[FinancialSource] = None

    # Operating
    net_income_cfs: Optional[float] = None
    depreciation_amortization: Optional[float] = None
    stock_based_compensation: Optional[float] = None
    changes_in_working_capital: Optional[float] = None
    accounts_receivable_change: Optional[float] = None
    inventory_change: Optional[float] = None
    accounts_payable_change: Optional[float] = None
    other_operating_changes: Optional[float] = None
    cfo: Optional[float] = None  # Cash flow from operations

    # Investing
    capex: Optional[float] = None
    acquisitions: Optional[float] = None
    asset_sales: Optional[float] = None
    investment_purchases: Optional[float] = None
    investment_sales: Optional[float] = None
    other_investing: Optional[float] = None
    cfi: Optional[float] = None  # Cash flow from investing

    # Financing
    debt_issuance: Optional[float] = None
    debt_repayment: Optional[float] = None
    equity_issuance: Optional[float] = None
    share_buybacks: Optional[float] = None
    dividends_paid: Optional[float] = None
    other_financing: Optional[float] = None
    cff: Optional[float] = None  # Cash flow from financing

    # Summary
    net_change_in_cash: Optional[float] = None
    beginning_cash: Optional[float] = None
    ending_cash: Optional[float] = None

    # Derived
    fcf: Optional[float] = None          # Free cash flow = CFO - Capex
    fcf_margin: Optional[float] = None
    fcf_to_net_income: Optional[float] = None
    capex_to_revenue: Optional[float] = None
    cash_conversion_ratio: Optional[float] = None  # CFO / Net Income

    @model_validator(mode="after")
    def compute_derived(self) -> "CashFlowStatement":
        if self.cfo is not None and self.capex is not None:
            object.__setattr__(self, "fcf", self.cfo - abs(self.capex))
        if self.cfo is not None and self.net_income_cfs and self.net_income_cfs != 0:
            object.__setattr__(self, "cash_conversion_ratio", self.cfo / self.net_income_cfs)
        return self


class SectorKPI(BaseModel):
    """Dynamic sector-specific KPIs — never use generic templates."""
    fiscal_year: str
    sector: str
    kpis: dict[str, float] = Field(default_factory=dict)
    kpi_definitions: dict[str, str] = Field(default_factory=dict)
    sources: dict[str, str] = Field(default_factory=dict)


class FinancialHistory(BaseModel):
    """Complete financial record for a company across all periods."""
    company_ticker: str
    currency: str
    unit: str = "millions"
    income_statements: dict[str, IncomeStatement] = Field(default_factory=dict)
    balance_sheets: dict[str, BalanceSheet] = Field(default_factory=dict)
    cash_flows: dict[str, CashFlowStatement] = Field(default_factory=dict)
    sector_kpis: dict[str, SectorKPI] = Field(default_factory=dict)
    available_years: list[str] = Field(default_factory=list)
    data_completeness_pct: float = Field(0.0, ge=0.0, le=100.0)

    def years_sorted(self, ascending: bool = True) -> list[str]:
        return sorted(self.available_years, reverse=not ascending)

    def get_income_statement(self, year: str) -> Optional[IncomeStatement]:
        return self.income_statements.get(year)

    def get_balance_sheet(self, year: str) -> Optional[BalanceSheet]:
        return self.balance_sheets.get(year)

    def get_cash_flow(self, year: str) -> Optional[CashFlowStatement]:
        return self.cash_flows.get(year)

    def revenue_series(self) -> dict[str, Optional[float]]:
        return {yr: self.income_statements[yr].revenue for yr in self.years_sorted() if yr in self.income_statements}

    def net_income_series(self) -> dict[str, Optional[float]]:
        return {yr: self.income_statements[yr].net_income for yr in self.years_sorted() if yr in self.income_statements}
