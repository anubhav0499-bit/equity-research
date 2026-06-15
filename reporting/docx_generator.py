"""
DOCX Report Generator — produces institutional research report in Microsoft Word format.
Sections: Cover | Executive Summary | Investment Thesis | Accounting Quality |
          Financial Statements | Forecasts | Valuation | Risk | Scenario | Certification | Disclaimer
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger

from ..models.research import ResearchState
from ..models.report import SectionType
from ..storage.storage_manager import StorageManager
from ..core.config import REPORT_CONFIG


def generate_docx_report(state: ResearchState, storage: StorageManager) -> Optional[Path]:
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor, Inches, Cm
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
    except ImportError:
        logger.error("python-docx not installed. Run: pip install python-docx")
        return None

    doc = Document()
    _set_margins(doc)

    profile = state.company_profile or {}
    company_name = state.company_name
    ticker = state.ticker
    exchange = profile.get("exchange", "")
    currency = profile.get("currency", "USD")
    report_date = datetime.now().strftime("%d %B %Y")

    val = state.valuation_summary or {}
    current_price = (state.agent_outputs.get("04_market_data") or _dummy()).payload.get(
        "market_data", {}).get("current_price")
    target_price = state.target_price
    rating = state.investment_rating
    risk_score = state.overall_risk_score

    # ─────────────────────────────────────────────────────────────
    # 1. COVER PAGE
    # ─────────────────────────────────────────────────────────────
    _add_cover_page(doc, company_name, ticker, exchange, currency, report_date,
                    current_price, target_price, rating, risk_score, profile)

    doc.add_page_break()

    # ─────────────────────────────────────────────────────────────
    # 2. TABLE OF CONTENTS PLACEHOLDER
    # ─────────────────────────────────────────────────────────────
    _add_heading(doc, "Table of Contents", level=1)
    toc_items = [
        ("1.", "Executive Summary"),
        ("2.", "Investment Thesis"),
        ("3.", "Accounting Quality Assessment"),
        ("4.", "Financial Statements"),
        ("5.", "Financial Forecasts"),
        ("6.", "Valuation"),
        ("7.", "Risk Analysis"),
        ("8.", "Scenario Analysis"),
        ("9.", "Certification"),
        ("10.", "Regulatory Disclaimer"),
    ]
    for num, title in toc_items:
        p = doc.add_paragraph()
        p.add_run(f"{num} {title}").font.size = Pt(11)
    doc.add_page_break()

    # ─────────────────────────────────────────────────────────────
    # 3. SECTIONS
    # ─────────────────────────────────────────────────────────────
    sections = state.report_sections

    section_map = {
        "1. Executive Summary": sections.get(SectionType.EXECUTIVE_SUMMARY.value, ""),
        "2. Investment Thesis": sections.get(SectionType.INVESTMENT_THESIS.value, ""),
        "3. Accounting Quality Assessment": sections.get(SectionType.ACCOUNTING_QUALITY.value, ""),
        "4. Financial Statements": _build_financial_statements_text(state),
        "5. Financial Forecasts": _build_forecast_text(state),
        "6. Valuation": sections.get(SectionType.VALUATION.value, "") or _build_valuation_text(state),
        "7. Risk Analysis": sections.get(SectionType.RISK_ANALYSIS.value, ""),
        "8. Scenario Analysis": sections.get(SectionType.SCENARIO_ANALYSIS.value, ""),
    }

    for section_title, content in section_map.items():
        _add_heading(doc, section_title, level=1)
        if content:
            for para in content.split("\n\n"):
                para = para.strip()
                if not para:
                    continue
                if para.startswith("## ") or para.startswith("# "):
                    _add_heading(doc, para.lstrip("#").strip(), level=2)
                elif para.startswith("**") and para.endswith("**"):
                    p = doc.add_paragraph()
                    run = p.add_run(para.strip("**"))
                    run.bold = True
                    run.font.size = Pt(11)
                else:
                    p = doc.add_paragraph(para)
                    p.style.font.size = Pt(11)
        else:
            doc.add_paragraph(f"[Section pending: {section_title}]")
        doc.add_paragraph("")

    # ─────────────────────────────────────────────────────────────
    # 9. Key Findings Table
    # ─────────────────────────────────────────────────────────────
    _add_findings_table(doc, state)

    # ─────────────────────────────────────────────────────────────
    # 10. CERTIFICATION
    # ─────────────────────────────────────────────────────────────
    _add_heading(doc, "9. Certification", level=1)
    cert_text = (
        f"We, the research team at {REPORT_CONFIG.firm_name}, certify that the views expressed in this "
        f"report accurately reflect our independent analysis of {company_name} as of {report_date}. "
        f"No part of our compensation was, is, or will be related to the specific recommendations or views expressed herein. "
        f"This research was conducted autonomously by the {REPORT_CONFIG.firm_name} AI Research Platform "
        f"using publicly available information only."
    )
    doc.add_paragraph(cert_text)
    doc.add_paragraph(f"Report Date: {report_date}")
    doc.add_paragraph(f"Platform: {REPORT_CONFIG.firm_name} v1.0")
    doc.add_paragraph(f"Run ID: {state.run_id}")

    # ─────────────────────────────────────────────────────────────
    # 11. DISCLAIMER
    # ─────────────────────────────────────────────────────────────
    _add_heading(doc, "10. Regulatory Disclaimer", level=1)
    disclaimer = (
        f"IMPORTANT NOTICE AND DISCLAIMER\n\n"
        f"This research report has been prepared by {REPORT_CONFIG.firm_name} ('the Firm') for informational "
        f"purposes only and does not constitute investment advice, a solicitation, or an offer to buy or sell "
        f"any security. The report is based on publicly available information believed to be reliable, but no "
        f"representation or warranty, express or implied, is made as to its accuracy or completeness.\n\n"
        f"This report is intended solely for institutional investors and qualified persons as defined under applicable "
        f"securities regulations. Past performance is not indicative of future results. All investments carry risk; "
        f"investors may lose some or all of their invested capital.\n\n"
        f"The Firm may hold positions in securities mentioned herein. This report was generated by an autonomous AI "
        f"research platform. Readers should conduct their own due diligence and seek independent financial advice before "
        f"making any investment decisions.\n\n"
        f"MiFID II Disclosure: This document is a non-independent research communication.\n"
        f"SEBI Disclosure: This report complies with SEBI Research Analyst Regulations (2014) where applicable.\n"
        f"CFA Institute: This report follows CFA Institute Standards of Professional Conduct.\n\n"
        f"© {datetime.now().year} {REPORT_CONFIG.firm_name}. All rights reserved."
    )
    for para in disclaimer.split("\n\n"):
        p = doc.add_paragraph(para)
        p.style.font.size = Pt(9)

    # ─────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────
    filename = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_equity_research.docx"
    report_path = storage.reports / filename
    doc.save(str(report_path))
    logger.info(f"DOCX report saved: {report_path}")
    return report_path


# ── Helpers ─────────────────────────────────────────────────────

def _dummy():
    from ..models.research import AgentOutput
    return AgentOutput(agent_id="x", agent_name="x")


def _set_margins(doc):
    from docx.shared import Cm
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)


def _add_heading(doc, text: str, level: int = 1):
    from docx.shared import Pt, RGBColor
    h = doc.add_heading(text, level=level)
    h.style.font.size = Pt(14 if level == 1 else 12)
    h.style.font.color.rgb = RGBColor(0x1a, 0x23, 0x7e)


def _add_cover_page(doc, company_name, ticker, exchange, currency, report_date,
                     current_price, target_price, rating, risk_score, profile):
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(REPORT_CONFIG.firm_name.upper())
    run.bold = True
    run.font.size = Pt(18)
    run.font.color.rgb = RGBColor(0x1a, 0x23, 0x7e)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(REPORT_CONFIG.firm_tagline).font.size = Pt(12)

    doc.add_paragraph("")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("EQUITY RESEARCH REPORT")
    run.bold = True
    run.font.size = Pt(14)

    doc.add_paragraph("")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(company_name)
    run.bold = True
    run.font.size = Pt(20)
    run.font.color.rgb = RGBColor(0x1a, 0x23, 0x7e)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(f"({ticker} | {exchange})").font.size = Pt(13)

    doc.add_paragraph("")
    table = doc.add_table(rows=5, cols=2)
    table.style = "Table Grid"
    data = [
        ("Current Price", f"{current_price:.2f} {currency}" if current_price else "N/A"),
        ("Target Price (12M)", f"{target_price:.2f} {currency}" if target_price else "N/A"),
        ("Investment Rating", rating or "N/A"),
        ("Risk Score", f"{risk_score:.0f}/100" if risk_score else "N/A"),
        ("Sector / Industry", f"{profile.get('sector', 'N/A')} / {profile.get('industry', 'N/A')}"),
    ]
    for i, (label, value) in enumerate(data):
        row = table.rows[i]
        row.cells[0].text = label
        row.cells[1].text = str(value)
        for cell in row.cells:
            for para in cell.paragraphs:
                if para.runs:
                    para.runs[0].font.size = Pt(11)

    doc.add_paragraph("")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(f"Report Date: {report_date}").font.size = Pt(10)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(f"For Institutional Investors Only").font.size = Pt(10)


def _build_financial_statements_text(state: ResearchState) -> str:
    history = state.financial_history or {}
    if not isinstance(history, dict):
        return "Financial statements not available."
    lines = ["Key Financial Metrics (in millions)\n"]
    years = sorted([k for k in history.keys() if len(k) == 4], reverse=True)[:5]
    metrics = ["revenue", "ebitda", "ebit", "net_income", "cfo", "total_assets", "net_debt"]
    for metric in metrics:
        row = f"{metric.replace('_', ' ').title():<25}"
        for yr in years:
            d = history.get(yr, {})
            is_d = d.get("income_statements") or {}
            bs_d = d.get("balance_sheets") or {}
            cf_d = d.get("cash_flows") or {}
            val = is_d.get(metric) or bs_d.get(metric) or cf_d.get(metric)
            row += f"  {yr}: {val:>10.1f}" if val is not None else f"  {yr}:        N/A"
        lines.append(row)
    return "\n".join(lines)


def _build_forecast_text(state: ResearchState) -> str:
    fm_out = state.agent_outputs.get("07_financial_modeling")
    if not fm_out:
        return "Financial forecasts not available."
    forecasts = fm_out.payload.get("forecasts", {})
    if not forecasts:
        return "Financial forecasts not available."
    lines = ["Base Case Financial Forecast (5 Years)\n"]
    base = forecasts.get("BASE", [])
    for fy in base:
        if isinstance(fy, dict):
            lines.append(
                f"{fy.get('year')}: Revenue={fy.get('revenue', 0):.0f}, "
                f"EBITDA={fy.get('ebitda', 0):.0f} ({fy.get('ebitda_margin', 0):.1f}%), "
                f"Net Income={fy.get('net_income', 0):.0f}, "
                f"FCF={fy.get('fcf', 0):.0f}, EPS={fy.get('eps', 0):.2f}"
            )
    return "\n".join(lines)


def _build_valuation_text(state: ResearchState) -> str:
    val = state.valuation_summary or {}
    if not val:
        return "Valuation not available."
    wacc = val.get("wacc_inputs", {}).get("wacc", "N/A")
    return (
        f"Valuation Summary\n\n"
        f"WACC: {wacc}%\n"
        f"Bear Case: {val.get('bear_price', 'N/A')}\n"
        f"Base Case: {val.get('base_price', 'N/A')} (Upside: {val.get('upside_pct', 'N/A')}%)\n"
        f"Bull Case: {val.get('bull_price', 'N/A')}\n\n"
        f"Methodology: 50% DCF (FCFF), 50% Peer Relative Valuation (EV/EBITDA, P/E).\n"
        f"Single WACC applied consistently across all methodologies and scenarios.\n"
    )


def _add_findings_table(doc, state: ResearchState):
    from docx.shared import Pt, RGBColor
    _add_heading(doc, "Key Research Findings", level=1)
    from ..models.research import RiskClassification, FindingType
    red_flags = [f for f in state.all_findings if f.finding_type == FindingType.RED_FLAG][:10]
    green_flags = [f for f in state.all_findings if f.finding_type == FindingType.GREEN_FLAG][:5]

    if red_flags:
        _add_heading(doc, "Risk Flags", level=2)
        table = doc.add_table(rows=1 + len(red_flags), cols=3)
        table.style = "Table Grid"
        headers = ["Agent", "Severity", "Finding"]
        for i, h in enumerate(headers):
            cell = table.rows[0].cells[i]
            cell.text = h
            cell.paragraphs[0].runs[0].bold = True
        for j, f in enumerate(red_flags):
            row = table.rows[j + 1]
            row.cells[0].text = f.agent_name[:25]
            row.cells[1].text = f.risk_level.value
            row.cells[2].text = f.title[:100]
        doc.add_paragraph("")

    if green_flags:
        _add_heading(doc, "Positive Indicators", level=2)
        for f in green_flags:
            p = doc.add_paragraph(f"✓ {f.title}", style="List Bullet")
            p.runs[0].font.size = Pt(11)
