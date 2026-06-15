"""
Company Profile — master configuration object consumed by all downstream agents.
Produced by CompanyProfilingAgent; never mutated after creation.
"""

from __future__ import annotations
from datetime import date
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class Exchange(str, Enum):
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    NSE = "NSE"
    BSE = "BSE"
    LSE = "LSE"
    TSX = "TSX"
    ASX = "ASX"
    HKEX = "HKEX"
    SGX = "SGX"
    EURONEXT = "EURONEXT"
    OTHER = "OTHER"


class Currency(str, Enum):
    USD = "USD"
    INR = "INR"
    GBP = "GBP"
    EUR = "EUR"
    CAD = "CAD"
    AUD = "AUD"
    HKD = "HKD"
    SGD = "SGD"
    JPY = "JPY"
    CNY = "CNY"


class FiscalYearEnd(str, Enum):
    MARCH = "MARCH"       # India FY
    DECEMBER = "DECEMBER"
    JUNE = "JUNE"
    SEPTEMBER = "SEPTEMBER"


class BusinessModel(str, Enum):
    B2B = "B2B"
    B2C = "B2C"
    B2B2C = "B2B2C"
    MARKETPLACE = "MARKETPLACE"
    SUBSCRIPTION = "SUBSCRIPTION"
    TRANSACTION = "TRANSACTION"
    ASSET_HEAVY = "ASSET_HEAVY"
    ASSET_LIGHT = "ASSET_LIGHT"
    FINANCIAL_SERVICES = "FINANCIAL_SERVICES"
    MIXED = "MIXED"


class ExchangeInfo(BaseModel):
    exchange: Exchange
    primary: bool = True
    listed_since: Optional[date] = None
    lot_size: Optional[int] = None

    model_config = {"frozen": True}


class CompanyProfile(BaseModel):
    """
    Master company descriptor. All downstream agents receive this as their
    primary configuration object. All fields are validated on construction.
    """
    # Identity
    name: str = Field(..., min_length=1, description="Official registered name")
    ticker: str = Field(..., min_length=1, description="Primary exchange ticker")
    isin: Optional[str] = Field(None, pattern=r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
    cin: Optional[str] = None  # India CIN

    # Listing
    exchange: Exchange
    all_exchanges: list[ExchangeInfo] = Field(default_factory=list)
    country: str = Field(..., description="Country of incorporation (ISO 3166-1 alpha-2)")
    currency: Currency

    # Calendar
    fiscal_year_end: FiscalYearEnd
    ipo_date: Optional[date] = None

    # Classification
    sector: str = Field(..., description="GICS sector or equivalent")
    industry: str = Field(..., description="GICS industry or equivalent")
    sub_industry: Optional[str] = None
    business_model: BusinessModel = BusinessModel.MIXED
    gics_code: Optional[str] = None

    # Market data (point-in-time snapshot from profiling)
    market_cap_usd: Optional[float] = Field(None, description="Market cap in USD at time of profiling")
    shares_outstanding: Optional[float] = None
    free_float_pct: Optional[float] = Field(None, ge=0, le=100)

    # IR / Filing sources
    ir_url: Optional[str] = None
    sec_cik: Optional[str] = None       # SEC EDGAR CIK
    bse_code: Optional[str] = None
    nse_symbol: Optional[str] = None
    lei: Optional[str] = None           # Legal Entity Identifier

    # Ownership
    promoter_holding_pct: Optional[float] = Field(None, ge=0, le=100)
    institutional_holding_pct: Optional[float] = Field(None, ge=0, le=100)

    # Audit
    auditor: Optional[str] = None
    auditor_firm_type: Optional[str] = None  # Big4 / Mid-tier / Small

    # Metadata
    resolved_at: Optional[str] = None   # ISO datetime string
    resolution_source: Optional[str] = None
    resolution_confidence: float = Field(1.0, ge=0.0, le=1.0)

    model_config = {"frozen": True}

    @field_validator("ticker")
    @classmethod
    def ticker_upper(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("country")
    @classmethod
    def country_upper(cls, v: str) -> str:
        return v.upper().strip()

    def display(self) -> str:
        return (
            f"{self.name} ({self.ticker}) | {self.exchange.value} | "
            f"{self.country} | {self.currency.value} | FY: {self.fiscal_year_end.value}"
        )
