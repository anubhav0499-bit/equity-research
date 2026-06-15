"""
Report configuration and section schemas for institutional research reports.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field, model_validator


class SectionType(str, Enum):
    COVER = "cover"
    EXECUTIVE_SUMMARY = "executive_summary"
    INVESTMENT_THESIS = "investment_thesis"
    ACCOUNTING_QUALITY = "accounting_quality"
    FINANCIAL_STATEMENTS = "financial_statements"
    FORECASTS = "forecasts"
    VALUATION = "valuation"
    RISK_ANALYSIS = "risk_analysis"
    SCENARIO_ANALYSIS = "scenario_analysis"
    CERTIFICATION = "certification"
    DISCLAIMER = "disclaimer"


class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    WATERFALL = "waterfall"
    SCATTER = "scatter"
    TABLE = "table"
    GAUGE = "gauge"
    HEATMAP = "heatmap"


class ReportChart(BaseModel):
    title: str
    chart_type: ChartType
    data: dict[str, Any]
    section: SectionType
    caption: Optional[str] = None


class ReportSection(BaseModel):
    section_type: SectionType
    title: str
    content: str
    word_count: int = 0
    charts: list[ReportChart] = Field(default_factory=list)
    tables: list[dict] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    order: int = 0

    @model_validator(mode="after")
    def compute_word_count(self) -> "ReportSection":
        if not self.word_count and self.content:
            object.__setattr__(self, "word_count", len(self.content.split()))
        return self


class ReportConfig(BaseModel):
    firm_name: str = "Equity Intelligence Research"
    firm_tagline: str = "Institutional-Grade Independent Research"
    analyst_name: str = "AI Research Platform"
    report_date: Optional[str] = None

    # Branding
    primary_color: str = "#1a237e"
    accent_color: str = "#b71c1c"
    positive_color: str = "#1b5e20"
    warning_color: str = "#e65100"

    # Fonts
    heading_font: str = "Calibri"
    body_font: str = "Calibri"
    heading_size: int = 14
    body_size: int = 11

    # Thresholds
    min_word_count: int = 15000
    target_word_count: int = 25000

    # Output
    output_dir: str = "reports"


class InstitutionalReport(BaseModel):
    """
    Complete institutional research report — all sections validated before output.
    """
    company_name: str
    ticker: str
    exchange: str
    currency: str
    report_date: str
    analyst: str
    config: ReportConfig = Field(default_factory=ReportConfig)

    sections: dict[SectionType, ReportSection] = Field(default_factory=dict)
    charts: list[ReportChart] = Field(default_factory=list)

    # Metadata
    total_word_count: int = 0
    source_count: int = 0
    validation_passed: bool = False
    output_path: Optional[str] = None

    # Key outputs (for cover page / executive summary)
    investment_rating: str = ""
    target_price: Optional[float] = None
    current_price: Optional[float] = None
    upside_pct: Optional[float] = None
    risk_rating: str = ""

    def required_sections(self) -> list[SectionType]:
        return [
            SectionType.COVER, SectionType.EXECUTIVE_SUMMARY,
            SectionType.INVESTMENT_THESIS, SectionType.ACCOUNTING_QUALITY,
            SectionType.FINANCIAL_STATEMENTS, SectionType.FORECASTS,
            SectionType.VALUATION, SectionType.RISK_ANALYSIS,
            SectionType.SCENARIO_ANALYSIS, SectionType.CERTIFICATION,
            SectionType.DISCLAIMER,
        ]

    def missing_sections(self) -> list[SectionType]:
        required = self.required_sections()
        return [s for s in required if s not in self.sections]

    def word_count(self) -> int:
        return sum(s.word_count for s in self.sections.values())

    def is_complete(self) -> bool:
        return (
            len(self.missing_sections()) == 0
            and self.word_count() >= self.config.min_word_count
            and self.validation_passed
        )
