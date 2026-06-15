"""
DOCX Report Generator — institutional research format (Motilal Oswal style).

Layout (mirrors MOFSL TCS / SBI template structure):
  Page 1   Cover page: 2-col (key metrics + CMP/TP/rating + investment thesis)
  Page 2+  Results summary → Management commentary
           Quarterly/annual performance table
           Valuation & scenario analysis
           Forensic accounting & quality
           Financials & valuations (IS + BS + CF + Ratios)  ← MOFSL pp. 10-11 equivalent
           Key research findings
           Investment rating legend
           Regulatory disclaimer
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from loguru import logger

from ..models.research import ResearchState, FindingType
from ..models.report import SectionType
from ..storage.storage_manager import StorageManager
from ..core.config import REPORT_CONFIG

# ── Colour palette (RGB tuples → hex strings) ──────────────────────────────
_NAV  = "1F497D"   # deep navy  — main headings
_TEAL = "17375E"   # dark blue  — section header bars
_LBLU = "2F75B6"   # mid blue   — sub-section bars
_LBKG = "BDD7EE"   # pale blue  — table header fill
_GRN  = "00B050"   # green      — BUY
_RED  = "C00000"   # red        — SELL
_ORG  = "FF9200"   # orange     — HOLD/NEUTRAL
_LGY  = "F2F2F2"   # light grey — alternate rows
_WHT  = "FFFFFF"
_BLK  = "000000"

_RATING_HEX: dict[str, str] = {
    "BUY": _GRN, "STRONG BUY": _GRN, "OUTPERFORM": _GRN, "ACCUMULATE": _GRN,
    "HOLD": _ORG, "NEUTRAL": _ORG,
    "UNDERPERFORM": _RED, "REDUCE": _RED, "SELL": _RED, "STRONG SELL": _RED,
}
_NA = "N/A"


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def generate_docx_report(state: ResearchState, storage: StorageManager) -> Optional[Path]:
    try:
        from docx import Document
        from docx.shared import Pt, Inches, Cm
    except ImportError:
        logger.error("python-docx not installed. Run: pip install python-docx")
        return None

    doc = Document()
    _set_margins(doc)

    profile     = state.company_profile or {}
    company     = state.company_name
    ticker      = state.ticker
    exchange    = profile.get("exchange", "")
    currency    = profile.get("currency", "INR")
    sector      = profile.get("sector", "")
    industry    = profile.get("industry", "")
    report_date = datetime.now().strftime("%d %B %Y")

    mktdata  = _payload(state, "04_market_data", "market_data", {})
    val      = state.valuation_summary or {}
    cmp_px   = mktdata.get("current_price")
    tgt_px   = state.target_price
    rating   = state.investment_rating or "UNDER REVIEW"
    risk_sc  = state.overall_risk_score

    upside = None
    if cmp_px and tgt_px and cmp_px > 0:
        upside = round((tgt_px - cmp_px) / cmp_px * 100, 1)

    fin_data = _collect_financial_data(state)

    # ── 1. Cover ──────────────────────────────────────────────────────────
    _add_cover(doc, company, ticker, exchange, currency, sector, industry,
               report_date, cmp_px, tgt_px, rating, upside, risk_sc,
               profile, mktdata, val, fin_data, state)

    doc.add_page_break()

    # ── 2. Results summary & management commentary ────────────────────────
    _add_results_section(doc, state, company, ticker, report_date)

    # ── 3. Quarterly / annual performance table ───────────────────────────
    _add_performance_table(doc, state, currency)

    # ── 4. Valuation & scenario analysis ─────────────────────────────────
    _add_valuation_section(doc, state, company, currency, cmp_px, tgt_px, upside, rating, val)

    # ── 5. Forensic accounting & accounting quality ───────────────────────
    _add_forensic_section(doc, state, company)

    # ── 6. Financials & valuations (MOFSL pp. 10-11 style) ───────────────
    _add_financials_section(doc, fin_data, currency, company)

    # ── 7. Key research findings ──────────────────────────────────────────
    _add_findings_section(doc, state)

    # ── 8. Certification ──────────────────────────────────────────────────
    _add_certification(doc, company, ticker, report_date, state)

    # ── 9. Rating legend + disclaimer ─────────────────────────────────────
    _add_rating_legend(doc)
    _add_disclaimer(doc, company, report_date)

    # ── Save ──────────────────────────────────────────────────────────────
    filename = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_equity_research.docx"
    out_path = storage.reports / filename
    doc.save(str(out_path))
    logger.info(f"DOCX report saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# SECTION BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def _add_cover(doc, company, ticker, exchange, currency, sector, industry,
               report_date, cmp_px, tgt_px, rating, upside, risk_sc,
               profile, mktdata, val, fin_data, state):
    from docx.shared import Pt, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    # ── Header line (date | report type | ticker) ──────────────────────
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run(f"{report_date}   |   Initiating Coverage   |   Sector: {sector or 'Equity'}")
    run.font.size = Pt(8)
    _color_run(run, _TEAL)

    # ── Company name (large, navy) ────────────────────────────────────
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run(company.upper())
    run.bold = True
    run.font.size = Pt(22)
    _color_run(run, _NAV)

    # ── Two-column layout via table ───────────────────────────────────
    cover = doc.add_table(rows=1, cols=2)
    cover.style = "Table Grid"
    _remove_table_borders(cover)

    lw = Cm(7.0)   # left column width
    rw = Cm(9.0)   # right column width
    cover.columns[0].width = lw
    cover.columns[1].width = rw

    lc = cover.rows[0].cells[0]
    rc = cover.rows[0].cells[1]
    lc.width = lw
    rc.width = rw

    # ── LEFT COLUMN ───────────────────────────────────────────────────
    _left_column(lc, ticker, exchange, currency, profile, mktdata, fin_data, state)

    # ── RIGHT COLUMN ──────────────────────────────────────────────────
    _right_column(rc, company, ticker, currency, cmp_px, tgt_px, rating,
                  upside, risk_sc, val, state)

    # ── Analyst line ──────────────────────────────────────────────────
    doc.add_paragraph("")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(
        f"Equity Intelligence Agent (AI-assisted research)   |   "
        f"Platform: {REPORT_CONFIG.firm_name}   |   Run: {state.run_id}"
    )
    run.font.size = Pt(7)
    _color_run(run, "595959")

    p2 = doc.add_paragraph()
    run2 = p2.add_run(
        "Investors are advised to refer through the important disclosures made at the last page of this report."
    )
    run2.font.size = Pt(7)
    run2.italic = True


def _left_column(cell, ticker, exchange, currency, profile, mktdata, fin_data, state):
    from docx.shared import Pt

    def _lp(text="", bold=False, size=8, color=_BLK):
        p = cell.add_paragraph()
        if text:
            r = p.add_run(text)
            r.bold = bold
            r.font.size = Pt(size)
            _color_run(r, color)
        return p

    # Change indicators
    _lp("Estimate change   ►", size=7, color="595959")
    _lp("TP change         ►", size=7, color="595959")
    _lp("Rating change     ►", size=7, color="595959")
    _lp()

    # Key metrics box
    shares = profile.get("shares_outstanding") or mktdata.get("shares_outstanding")
    mktcap = profile.get("market_cap") or mktdata.get("market_cap")
    hi52   = mktdata.get("week_52_high") or profile.get("52w_high")
    lo52   = mktdata.get("week_52_low")  or profile.get("52w_low")
    beta   = mktdata.get("beta") or profile.get("beta")
    ff     = profile.get("free_float_pct") or mktdata.get("free_float_pct")

    metrics = [
        ("Bloomberg", f"{ticker} IN" if exchange in ("NSE", "BSE") else ticker),
        ("Equity Shares (m)", f"{shares/1e6:.0f}" if shares else _NA),
        (f"M.Cap ({currency} b)", f"{mktcap/1e9:.1f}" if mktcap else _NA),
        ("52-Week Range", f"{_fmt(hi52)} / {_fmt(lo52)}"),
        ("Beta", f"{beta:.2f}" if beta else _NA),
        ("Free Float (%)", f"{ff:.1f}" if ff else _NA),
    ]

    t = _mini_table(cell, metrics, hdr_color=_LBLU, hdr_text="Key Data", key_width=Pt(70))
    cell.add_paragraph()

    # Financials & Valuations mini-table
    _add_fin_mini_table(cell, fin_data, currency)
    cell.add_paragraph()

    # Shareholding pattern
    _add_shareholding_table(cell, state)


def _right_column(cell, company, ticker, currency, cmp_px, tgt_px, rating,
                  upside, risk_sc, val, state):
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    # CMP / TP / Rating banner
    rating_hex = _RATING_HEX.get((rating or "").upper(), _ORG)
    upside_str = f"({upside:+.1f}%)" if upside is not None else ""

    banner_parts = [
        f"CMP: {currency}{_fmt(cmp_px)}",
        f"TP: {currency}{_fmt(tgt_px)} {upside_str}",
        rating.upper() if rating else "UNDER REVIEW",
    ]
    banner_p = cell.add_paragraph()
    for i, part in enumerate(banner_parts):
        r = banner_p.add_run(("   " if i else "") + part + ("   " if i < 2 else ""))
        r.bold = True
        r.font.size = Pt(12 if i < 2 else 13)
        if i == 2:
            _color_run(r, rating_hex)
        else:
            _color_run(r, _NAV)
    _shade_paragraph(banner_p, _LBKG)
    cell.add_paragraph()

    # Headline
    exec_summary = state.report_sections.get(SectionType.EXECUTIVE_SUMMARY.value, "")
    headline = _extract_headline(exec_summary, company)
    p = cell.add_paragraph()
    r = p.add_run(headline)
    r.bold = True
    r.font.size = Pt(13)
    _color_run(r, _NAV)
    cell.add_paragraph()

    # Investment thesis summary bar
    _add_cell_section_bar(cell, "Investment Thesis")
    thesis = state.report_sections.get(SectionType.INVESTMENT_THESIS.value, "")
    _add_bullets_to_cell(cell, _extract_bullets(thesis, max_bullets=5), size=9)
    cell.add_paragraph()

    # Our view bar
    _add_cell_section_bar(cell, "Our View")
    risk_txt = state.report_sections.get(SectionType.RISK_ANALYSIS.value, "")
    _add_bullets_to_cell(cell, _extract_bullets(risk_txt, max_bullets=3), size=9)

    # Valuation & view bar
    _add_cell_section_bar(cell, "Valuation & View")
    wacc   = val.get("wacc_inputs", {}).get("wacc", _NA)
    base   = _fmt(val.get("base_price"))
    bear   = _fmt(val.get("bear_price"))
    bull   = _fmt(val.get("bull_price"))
    val_bullets = [
        f"WACC: {wacc}%" if wacc != _NA else "WACC: N/A",
        f"Base case target: {currency}{base}  |  Upside: {f'{upside:+.1f}%' if upside else _NA}",
        f"Bear: {currency}{bear}   Base: {currency}{base}   Bull: {currency}{bull}",
        f"Risk score: {risk_sc:.0f}/100" if risk_sc else "Risk score: N/A",
    ]
    _add_bullets_to_cell(cell, val_bullets, size=9)


def _add_results_section(doc, state, company, ticker, report_date):
    _section_bar(doc, "Results Summary & Key Highlights")

    exec_text = state.report_sections.get(SectionType.EXECUTIVE_SUMMARY.value, "")
    if exec_text:
        _add_content_paragraphs(doc, exec_text, max_chars=1800)
    else:
        doc.add_paragraph(f"Executive summary pending data retrieval for {company}.")

    _section_bar(doc, "Key Highlights from Management Commentary", color=_LBLU)

    transcript = _payload(state, "12_transcript_retrieval", "transcript_analysis", {})
    key_themes = transcript.get("key_themes", [])
    guidance   = transcript.get("guidance_statements", [])
    sentiment  = transcript.get("overall_sentiment", "")

    if sentiment:
        p = doc.add_paragraph()
        r = p.add_run(f"Management Sentiment: {sentiment}")
        r.bold = True
        r.font.size = _pt(10)

    items = key_themes[:6] + guidance[:4]
    if items:
        for item in items:
            p = doc.add_paragraph(str(item), style="List Bullet")
            p.runs[0].font.size = _pt(10)
    else:
        thesis = state.report_sections.get(SectionType.INVESTMENT_THESIS.value, "")
        _add_content_paragraphs(doc, thesis, max_chars=1200)

    _section_bar(doc, "Valuation and View", color=_LBLU)
    val_text = state.report_sections.get(SectionType.VALUATION.value, "")
    if val_text:
        _add_content_paragraphs(doc, val_text, max_chars=1000)
    else:
        _add_content_paragraphs(
            doc,
            _payload(state, "08_valuation", "llm_valuation_narrative", "Valuation data pending."),
            max_chars=1000,
        )


def _add_performance_table(doc, state, currency):
    from docx.shared import Pt

    doc.add_paragraph()
    _section_bar(doc, "Annual Performance Summary")

    fm_out    = state.agent_outputs.get("07_financial_modeling")
    forecasts = (fm_out.payload.get("forecasts", {}) if fm_out else {})
    base_fcast = forecasts.get("BASE", [])
    history   = state.financial_history or {}
    years_hist = sorted([k for k in history.keys() if len(k) == 4])[-4:]

    est_years = [str(f.get("year", f"FY+{i+1}E")) for i, f in enumerate(base_fcast[:3])]
    all_cols  = years_hist + est_years
    if not all_cols:
        doc.add_paragraph("Performance data not yet available.")
        return

    hdr = ["Metric"] + all_cols
    rows: List[List[str]] = []

    def _hval(yr, *keys):
        d = history.get(yr, {})
        for section in ("income_statements", "balance_sheets", "cash_flows"):
            sd = d.get(section) or {}
            for k in keys:
                v = sd.get(k)
                if v is not None:
                    return v
        return None

    def _fval(i, key):
        if i < len(base_fcast):
            return base_fcast[i].get(key)
        return None

    metrics = [
        ("Revenue",          ["revenue"],               "revenue"),
        ("Revenue Growth %", ["revenue_growth_pct"],    "revenue_growth_pct"),
        ("Gross Profit",     ["gross_profit"],           "gross_profit"),
        ("Gross Margin %",   ["gross_margin_pct"],       "gross_margin_pct"),
        ("EBITDA",           ["ebitda"],                 "ebitda"),
        ("EBITDA Margin %",  ["ebitda_margin_pct", "ebitda_margin"], "ebitda_margin"),
        ("EBIT",             ["ebit"],                   "ebit"),
        ("EBIT Margin %",    ["ebit_margin_pct", "ebit_margin"],     "ebit_margin"),
        ("PAT / Net Income", ["net_income", "pat"],      "net_income"),
        ("PAT Margin %",     ["net_margin_pct", "pat_margin_pct"],  "net_margin"),
        ("EPS",              ["eps"],                    "eps"),
        ("Free Cash Flow",   ["fcf", "free_cash_flow"], "fcf"),
    ]

    for label, hist_keys, fcast_key in metrics:
        row = [label]
        for yr in years_hist:
            v = _hval(yr, *hist_keys)
            row.append(_fmt_metric(v, label))
        for i in range(len(est_years)):
            v = _fval(i, fcast_key)
            row.append((_fmt_metric(v, label) + "E") if v is not None else _NA)
        rows.append(row)

    t = doc.add_table(rows=1 + len(rows), cols=len(hdr))
    t.style = "Table Grid"

    for ci, h in enumerate(hdr):
        cell = t.rows[0].cells[ci]
        cell.text = h
        _set_cell_bg(cell, _LBLU)
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = _pt(8)
        _color_run(cell.paragraphs[0].runs[0], _WHT)

    for ri, row in enumerate(rows):
        bg = _LGY if ri % 2 else _WHT
        for ci, val in enumerate(row):
            cell = t.rows[ri + 1].cells[ci]
            cell.text = str(val)
            _set_cell_bg(cell, bg)
            r = cell.paragraphs[0].runs
            if r:
                r[0].font.size = _pt(8)
                if ci == 0:
                    r[0].bold = True

    # Estimate revision table
    doc.add_paragraph()
    _section_bar(doc, "Estimate Revisions (Bear / Base / Bull)", color=_LBLU)
    _add_scenario_table(doc, state, currency)


def _add_scenario_table(doc, state, currency):
    scen_out  = state.agent_outputs.get("15_scenario_analysis")
    scen_pay  = scen_out.payload if scen_out else {}
    scenarios = scen_pay.get("scenarios", {})

    fm_out    = state.agent_outputs.get("07_financial_modeling")
    forecasts = (fm_out.payload.get("forecasts", {}) if fm_out else {})

    labels = ["BEAR", "BASE", "BULL"]
    hdr = ["Metric"] + [s.title() + " Case" for s in labels]
    rows_data = [
        ("Revenue CAGR (5Y)", "revenue_cagr_pct", "%"),
        ("EBITDA Margin (%)", "ebitda_margin",    "%"),
        ("PAT CAGR (5Y)",     "pat_cagr_pct",     "%"),
        ("Target Price",      "price_target",     currency),
        ("Upside/Downside",   "upside_pct",       "%"),
        ("Probability",       "probability_pct",  "%"),
    ]

    t = doc.add_table(rows=1 + len(rows_data), cols=len(hdr))
    t.style = "Table Grid"
    for ci, h in enumerate(hdr):
        c = t.rows[0].cells[ci]
        c.text = h
        _set_cell_bg(c, _TEAL)
        if c.paragraphs[0].runs:
            c.paragraphs[0].runs[0].bold = True
            c.paragraphs[0].runs[0].font.size = _pt(9)
            _color_run(c.paragraphs[0].runs[0], _WHT)

    for ri, (label, key, unit) in enumerate(rows_data):
        t.rows[ri + 1].cells[0].text = label
        for ci, s in enumerate(labels):
            cell = t.rows[ri + 1].cells[ci + 1]
            sd   = scen_pay.get(s.lower(), {}) or (forecasts.get(s, [{}]) or [{}])[0]
            v    = sd.get(key)
            if v is not None:
                cell.text = f"{unit}{v:.1f}" if unit == currency else f"{v:.1f}{unit}"
            else:
                cell.text = _NA
            if cell.paragraphs[0].runs:
                cell.paragraphs[0].runs[0].font.size = _pt(9)

    # Probability-weighted target
    pwtp = scen_pay.get("probability_weighted_target_price")
    if pwtp:
        p = doc.add_paragraph()
        r = p.add_run(f"Probability-Weighted Target Price: {currency}{pwtp:.2f}")
        r.bold = True
        r.font.size = _pt(10)


def _add_valuation_section(doc, state, company, currency, cmp_px, tgt_px, upside, rating, val):
    _section_bar(doc, "Valuation")

    wacc_in = val.get("wacc_inputs", {}) or {}
    rows = [
        ("Valuation Method",     "Weight",   "Implied Value"),
        ("DCF (Base Case)",      "40%",      f"{currency}{_fmt(val.get('dcf_value'))}"),
        ("Peer EV/EBITDA",       "30%",      f"{currency}{_fmt(val.get('relative_value'))}"),
        ("SOTP",                 "20%",      f"{currency}{_fmt(val.get('sotp_value'))}"),
        ("FCF Yield / DDM",      "10%",      f"{currency}{_fmt(val.get('fcf_yield_value'))}"),
        ("Blended Target Price", "100%",     f"{currency}{_fmt(tgt_px)}"),
        ("Current Market Price", "—",        f"{currency}{_fmt(cmp_px)}"),
        ("Upside / (Downside)",  "—",        f"{f'{upside:+.1f}%' if upside else _NA}"),
    ]

    t = doc.add_table(rows=len(rows), cols=3)
    t.style = "Table Grid"
    for ri, row in enumerate(rows):
        bg = _LBLU if ri == 0 else (_LGY if ri % 2 else _WHT)
        for ci, val_txt in enumerate(row):
            c = t.rows[ri].cells[ci]
            c.text = val_txt
            _set_cell_bg(c, bg)
            if c.paragraphs[0].runs:
                r = c.paragraphs[0].runs[0]
                r.font.size = _pt(9)
                if ri == 0:
                    r.bold = True
                    _color_run(r, _WHT)
                if ri in (len(rows) - 3, len(rows) - 2):
                    r.bold = True

    doc.add_paragraph()

    # WACC breakdown
    if wacc_in:
        _section_bar(doc, "WACC Construction", color=_LBLU)
        wacc_rows = [
            ("Risk-free Rate",         f"{_fmt(wacc_in.get('risk_free_rate'))}%"),
            ("Equity Risk Premium",    f"{_fmt(wacc_in.get('equity_risk_premium'))}%"),
            ("Country Risk Premium",   f"{_fmt(wacc_in.get('country_risk_premium'))}%"),
            ("Beta",                   f"{_fmt(wacc_in.get('beta'))}"),
            ("Cost of Equity",         f"{_fmt(wacc_in.get('cost_of_equity'))}%"),
            ("Pre-tax Cost of Debt",   f"{_fmt(wacc_in.get('pre_tax_cost_of_debt'))}%"),
            ("Effective Tax Rate",     f"{_fmt(wacc_in.get('tax_rate'))}%"),
            ("After-tax Cost of Debt", f"{_fmt(wacc_in.get('after_tax_cost_of_debt'))}%"),
            ("Equity Weight",          f"{_fmt(wacc_in.get('equity_weight'), pct=True)}%"),
            ("Debt Weight",            f"{_fmt(wacc_in.get('debt_weight'), pct=True)}%"),
            ("WACC",                   f"{_fmt(wacc_in.get('wacc'))}%"),
        ]
        t2 = doc.add_table(rows=len(wacc_rows), cols=2)
        t2.style = "Table Grid"
        for ri, (k, v) in enumerate(wacc_rows):
            bg = _LGY if ri % 2 else _WHT
            is_last = ri == len(wacc_rows) - 1
            for ci, txt in enumerate((k, v)):
                c = t2.rows[ri].cells[ci]
                c.text = txt
                _set_cell_bg(c, _TEAL if is_last else bg)
                if c.paragraphs[0].runs:
                    r = c.paragraphs[0].runs[0]
                    r.font.size = _pt(9)
                    r.bold = is_last
                    if is_last:
                        _color_run(r, _WHT)


def _add_forensic_section(doc, state, company):
    _section_bar(doc, "Forensic Accounting & Quality Assessment")

    forensic = _payload(state, "06_forensic_accounting", "details", {})
    acctq    = _payload(state, "05_accounting_quality", "quality_metrics", {})

    scores_data = [
        ("Score", "Value", "Threshold", "Interpretation"),
        ("Beneish M-Score",
         str(forensic.get("beneish", {}).get("m_score", _NA)),
         "< −1.78",
         forensic.get("beneish", {}).get("classification", _NA)),
        ("Piotroski F-Score",
         f"{forensic.get('piotroski', {}).get('f_score', _NA)} / 9",
         "≥ 7 = strong, ≤ 2 = weak",
         forensic.get("piotroski", {}).get("interpretation", _NA)),
        ("Altman Z-Score",
         str(forensic.get("altman", {}).get("z_score", _NA)),
         "> 2.6 = safe zone",
         forensic.get("altman", {}).get("zone", _NA)),
        ("Sloan Accrual Ratio",
         str(forensic.get("sloan_accrual", _NA)),
         "< 5% preferred",
         ""),
        ("Cash Conversion (OCF/NI)",
         str(acctq.get("cash_conversion", _NA)),
         "> 0.8× preferred",
         ""),
    ]

    t = doc.add_table(rows=len(scores_data), cols=4)
    t.style = "Table Grid"
    for ri, row in enumerate(scores_data):
        bg = _LBLU if ri == 0 else (_LGY if ri % 2 else _WHT)
        for ci, txt in enumerate(row):
            c = t.rows[ri].cells[ci]
            c.text = str(txt)
            _set_cell_bg(c, bg)
            if c.paragraphs[0].runs:
                r = c.paragraphs[0].runs[0]
                r.font.size = _pt(9)
                if ri == 0:
                    r.bold = True
                    _color_run(r, _WHT)

    doc.add_paragraph()

    # Red flags from forensic
    red_flags = [f for f in state.all_findings if f.finding_type == FindingType.RED_FLAG]
    if red_flags:
        _section_bar(doc, "Forensic Red Flags", color=_RED)
        t2 = doc.add_table(rows=1 + len(red_flags[:12]), cols=4)
        t2.style = "Table Grid"
        for ci, h in enumerate(("#", "Flag", "Severity", "Implication")):
            c = t2.rows[0].cells[ci]
            c.text = h
            _set_cell_bg(c, _TEAL)
            if c.paragraphs[0].runs:
                c.paragraphs[0].runs[0].bold = True
                c.paragraphs[0].runs[0].font.size = _pt(9)
                _color_run(c.paragraphs[0].runs[0], _WHT)
        for ri, f in enumerate(red_flags[:12]):
            t2.rows[ri + 1].cells[0].text = str(ri + 1)
            t2.rows[ri + 1].cells[1].text = f.title[:80]
            t2.rows[ri + 1].cells[2].text = f.risk_level.value
            t2.rows[ri + 1].cells[3].text = (f.evidence or "")[:120]
            for ci in range(4):
                if t2.rows[ri + 1].cells[ci].paragraphs[0].runs:
                    t2.rows[ri + 1].cells[ci].paragraphs[0].runs[0].font.size = _pt(8)

    doc.add_paragraph()
    aq_text = state.report_sections.get(SectionType.ACCOUNTING_QUALITY.value, "")
    if aq_text:
        _section_bar(doc, "Accounting Quality Assessment", color=_LBLU)
        _add_content_paragraphs(doc, aq_text, max_chars=1500)


def _add_financials_section(doc, fin_data: dict, currency: str, company: str):
    """MOFSL pp. 10-11 equivalent — IS, BS, CF, Ratios."""
    _section_bar(doc, f"Financials & Valuations  ({currency})")

    years     = fin_data.get("years", [])
    is_rows   = fin_data.get("income_statement", [])
    bs_rows   = fin_data.get("balance_sheet", [])
    cf_rows   = fin_data.get("cash_flow", [])
    rat_rows  = fin_data.get("ratios", [])

    if not years:
        doc.add_paragraph("Financial data not available for this company.")
        return

    ncols = 1 + len(years)

    def _write_fin_table(label: str, rows: List[List[str]]):
        if not rows:
            return
        _subsection_label(doc, label)
        t = doc.add_table(rows=1 + len(rows), cols=ncols)
        t.style = "Table Grid"
        # Header row
        hdr_cells = t.rows[0].cells
        hdr_cells[0].text = "Y/E"
        for ci, yr in enumerate(years, 1):
            hdr_cells[ci].text = str(yr)
        _set_cell_bg(hdr_cells[0], _NAV)
        for ci in range(ncols):
            _set_cell_bg(hdr_cells[ci], _NAV)
            if hdr_cells[ci].paragraphs[0].runs:
                r = hdr_cells[ci].paragraphs[0].runs[0]
                r.bold = True
                r.font.size = _pt(8)
                _color_run(r, _WHT)
        # Data rows
        for ri, row in enumerate(rows):
            is_bold_row = row[0].startswith("**") or row[0].isupper()
            label_clean = row[0].lstrip("*")
            bg = _LBKG if is_bold_row else (_LGY if ri % 2 else _WHT)
            for ci in range(ncols):
                c = t.rows[ri + 1].cells[ci]
                txt = label_clean if ci == 0 else (row[ci] if ci < len(row) else _NA)
                c.text = str(txt)
                _set_cell_bg(c, bg)
                if c.paragraphs[0].runs:
                    r2 = c.paragraphs[0].runs[0]
                    r2.font.size = _pt(8)
                    r2.bold = is_bold_row
        doc.add_paragraph()

    _write_fin_table("Income Statement", is_rows)
    _write_fin_table("Balance Sheet",    bs_rows)
    _write_fin_table("Cash Flow Statement", cf_rows)
    _write_fin_table("Ratios",           rat_rows)


def _add_findings_section(doc, state):
    _section_bar(doc, "Key Research Findings")

    green_flags = [f for f in state.all_findings if f.finding_type == FindingType.GREEN_FLAG]
    if green_flags:
        _subsection_label(doc, "Positive Indicators")
        for f in green_flags[:8]:
            p = doc.add_paragraph(f"✓  {f.title}", style="List Bullet")
            if p.runs:
                p.runs[0].font.size = _pt(10)
                _color_run(p.runs[0], "00B050")

    risk_text = state.report_sections.get(SectionType.RISK_ANALYSIS.value, "")
    if risk_text:
        _section_bar(doc, "Risk Analysis", color=_LBLU)
        _add_content_paragraphs(doc, risk_text, max_chars=1500)

    scen_text = state.report_sections.get(SectionType.SCENARIO_ANALYSIS.value, "")
    if scen_text:
        _section_bar(doc, "Scenario Analysis", color=_LBLU)
        _add_content_paragraphs(doc, scen_text, max_chars=1200)


def _add_certification(doc, company, ticker, report_date, state):
    _section_bar(doc, "Analyst Certification")
    doc.add_paragraph(
        f"The views expressed in this research report accurately reflect the independent "
        f"analysis of {company} ({ticker}) as of {report_date}. No part of the compensation "
        f"of the research team was, is, or will be directly or indirectly related to the "
        f"specific recommendations or views expressed herein. "
        f"Platform: {REPORT_CONFIG.firm_name} v1.0   |   Run ID: {state.run_id}"
    ).runs[0].font.size = _pt(9)


def _add_rating_legend(doc):
    from docx.shared import Pt
    doc.add_page_break()
    _section_bar(doc, "Explanation of Investment Rating")
    legend = [
        ("Investment Rating", "Expected Return (over 12 months)"),
        ("STRONG BUY",        "> 25%"),
        ("BUY",               ">= 15%"),
        ("ACCUMULATE",        "5% to 15%"),
        ("HOLD / NEUTRAL",    "−5% to 5%"),
        ("REDUCE",            "−5% to −15%"),
        ("SELL",              "< −15%"),
        ("UNDER REVIEW",      "Rating may undergo a change"),
        ("NOT RATED",         "Forward-looking estimates exist; no recommendation assigned"),
    ]
    t = doc.add_table(rows=len(legend), cols=2)
    t.style = "Table Grid"
    for ri, (k, v) in enumerate(legend):
        bg = _TEAL if ri == 0 else _WHT
        for ci, txt in enumerate((k, v)):
            c = t.rows[ri].cells[ci]
            c.text = txt
            _set_cell_bg(c, bg)
            if c.paragraphs[0].runs:
                r = c.paragraphs[0].runs[0]
                r.font.size = _pt(9)
                r.bold = ri == 0
                if ri == 0:
                    _color_run(r, _WHT)


def _add_disclaimer(doc, company, report_date):
    from docx.shared import Pt
    _section_bar(doc, "Important Disclosures & Disclaimer")

    disclaimer_blocks = [
        (
            f"This research report has been prepared by {REPORT_CONFIG.firm_name} "
            f"('the Platform') for informational purposes only and does not constitute "
            f"investment advice, a solicitation, or an offer to buy or sell any security. "
            f"Report date: {report_date}."
        ),
        (
            "This report is generated by an AI-assisted autonomous research platform. "
            "All data has been sourced from publicly available information and has not been "
            "independently verified. No representation or warranty, express or implied, is "
            "made as to the accuracy or completeness of the information contained herein."
        ),
        (
            "This report is intended solely for institutional investors and qualified "
            "persons as defined under applicable securities regulations. Past performance "
            "is not indicative of future results. All investments carry risk; investors "
            "may lose some or all of their invested capital."
        ),
        (
            "MiFID II Disclosure: This document is a non-independent research communication. "
            "SEBI Disclosure: Produced in compliance with SEBI Research Analyst Regulations (2014) "
            "where applicable. CFA Institute: This report follows CFA Institute Standards of "
            "Professional Conduct."
        ),
        (
            f"The Platform and its associates may have financial interest in {company}. "
            f"Recipients should conduct their own due diligence and seek independent financial "
            f"advice before making any investment decisions. "
            f"© {datetime.now().year} {REPORT_CONFIG.firm_name}. All rights reserved."
        ),
        (
            "Registration granted by SEBI, enlistment with Exchange and certification from "
            "NISM in no way guarantee performance of the intermediary or provide any assurance "
            "of returns to investors. Investment in securities market is subject to market risks. "
            "Read all related documents carefully before investing."
        ),
    ]

    for block in disclaimer_blocks:
        p = doc.add_paragraph(block)
        p.runs[0].font.size = _pt(8)
    doc.add_paragraph()


# ═══════════════════════════════════════════════════════════════════════════
# DATA COLLECTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _collect_financial_data(state: ResearchState) -> dict:
    """Organise historical + forecast data into parallel column lists."""
    history    = state.financial_history or {}
    hist_years = sorted([k for k in history.keys() if len(k) == 4])[-5:]

    fm_out     = state.agent_outputs.get("07_financial_modeling")
    forecasts  = (fm_out.payload.get("forecasts", {}) if fm_out else {})
    base_f     = forecasts.get("BASE", [])[:3]

    est_labels = [str(f.get("year", f"FY+{i+1}E")) + "E"
                  for i, f in enumerate(base_f)]
    years = hist_years + est_labels

    def _hv(yr, *keys):
        d = history.get(yr, {})
        for sect in ("income_statements", "balance_sheets", "cash_flows"):
            sd = d.get(sect) or {}
            for k in keys:
                v = sd.get(k)
                if v is not None:
                    return v
        return None

    def _fv(i, key):
        if i < len(base_f):
            return base_f[i].get(key)
        return None

    def _row(label: str, hist_keys, fcast_key, pct=False, currency=True, bold=False):
        prefix = "**" if bold else ""
        cells  = [prefix + label]
        for yr in hist_years:
            v = _hv(yr, *hist_keys)
            cells.append(_fmt_num(v, pct))
        for i in range(len(est_labels)):
            v = _fv(i, fcast_key)
            cells.append(_fmt_num(v, pct) if v is not None else _NA)
        return cells

    is_rows = [
        _row("**Sales",           ["revenue"],               "revenue",              bold=True),
        _row("  Change (%)",       ["revenue_growth_pct"],    "revenue_growth_pct",  pct=True),
        _row("Cost of Services",   ["cogs", "cost_of_goods"], "cogs"),
        _row("**Gross Profit",     ["gross_profit"],          "gross_profit",         bold=True),
        _row("  Gross Margin (%)", ["gross_margin_pct"],      "gross_margin_pct",    pct=True),
        _row("SG&A Expenses",      ["sga", "opex"],           "opex"),
        _row("**EBITDA",           ["ebitda"],                "ebitda",               bold=True),
        _row("  % of Net Sales",   ["ebitda_margin_pct", "ebitda_margin"], "ebitda_margin", pct=True),
        _row("Depreciation",       ["depreciation", "da"],   "depreciation"),
        _row("**EBIT",             ["ebit"],                  "ebit",                 bold=True),
        _row("  % of Net Sales",   ["ebit_margin_pct", "ebit_margin"], "ebit_margin", pct=True),
        _row("Other Income",       ["other_income"],          "other_income"),
        _row("**PBT",              ["pbt", "pre_tax_income"], "pbt",                  bold=True),
        _row("Tax",                ["tax_expense", "tax"],    "tax"),
        _row("  Rate (%)",         ["effective_tax_rate"],    "tax_rate",             pct=True),
        _row("Minority Interest",  ["minority_interest"],     "minority_interest"),
        _row("**Adjusted PAT",     ["net_income", "pat"],     "net_income",           bold=True),
        _row("  Change (%)",       ["net_income_growth_pct"], "net_income_growth_pct", pct=True),
    ]

    bs_rows = [
        _row("**Net Worth",        ["equity", "total_equity", "net_worth"], "equity", bold=True),
        _row("Minority Interest",  ["minority_interest_bs"], "minority_interest"),
        _row("Gross Debt",         ["total_debt", "gross_debt"],            "total_debt"),
        _row("**Capital Employed", ["capital_employed", "total_assets"],    "total_assets", bold=True),
        _row("**Gross Block",      ["gross_block", "ppe_gross"],            "ppe_gross",    bold=True),
        _row("Depreciation",       ["accumulated_depreciation"],            ""),
        _row("**Net Block",        ["net_block", "ppe_net"],                "ppe_net",      bold=True),
        _row("Debtors",            ["accounts_receivable", "trade_receivables"], ""),
        _row("Cash & Bank",        ["cash_and_equivalents", "cash"],        "cash"),
        _row("**Curr. Assets",     ["current_assets", "total_current_assets"], "current_assets", bold=True),
        _row("**Curr. Liab.",      ["current_liabilities"],                 "current_liabilities", bold=True),
        _row("**Net Current Assets",["working_capital", "net_current_assets"], "working_capital", bold=True),
    ]

    cf_rows = [
        _row("CF from Operations", ["cfo", "operating_cash_flow"],   "cfo",    bold=True),
        _row("  Working Capital Δ",["working_capital_change"],        ""),
        _row("**Net Operating CF", ["net_operating_cf", "cfo"],      "cfo",    bold=True),
        _row("Net Capex",          ["capex"],                         "capex"),
        _row("**Free Cash Flow",   ["fcf", "free_cash_flow"],        "fcf",    bold=True),
        _row("Dividend Payments",  ["dividends_paid"],               "dividends"),
        _row("**Net Cash Flow",    ["net_change_in_cash"],            ""),
        _row("Closing Cash Bal.",  ["cash_and_equivalents", "cash"], "cash"),
    ]

    rat_rows = [
        _row("**EPS",              ["eps", "basic_eps"],              "eps",     bold=True),
        _row("Book Value/Share",   ["bvps", "book_value_per_share"],  "bvps"),
        _row("DPS",                ["dps", "dividend_per_share"],     "dps"),
        _row("Payout (%)",         ["payout_ratio_pct"],              "",        pct=True),
        _row("**P/E (x)",          ["pe_ratio"],                      "",        bold=True),
        _row("EV/EBITDA (x)",      ["ev_ebitda"],                     ""),
        _row("EV/Sales (x)",       ["ev_sales"],                      ""),
        _row("P/BV (x)",           ["pb_ratio", "price_to_book"],     ""),
        _row("Div. Yield (%)",     ["dividend_yield_pct"],            "",        pct=True),
        _row("**RoE (%)",          ["roe_pct", "return_on_equity"],   "",        bold=True, pct=True),
        _row("RoCE (%)",           ["roce_pct", "return_on_capital"], "",        pct=True),
        _row("Debtors (Days)",     ["dso", "days_sales_outstanding"], ""),
        _row("Asset Turnover (x)", ["asset_turnover"],                ""),
    ]

    return {
        "years":            years,
        "income_statement": is_rows,
        "balance_sheet":    bs_rows,
        "cash_flow":        cf_rows,
        "ratios":           rat_rows,
    }


def _add_fin_mini_table(cell, fin_data: dict, currency: str):
    """Compact financials & valuation table on cover — like MOFSL left-column block."""
    years = fin_data.get("years", [])[-3:]   # show only last 3 (most recent 1 actual + 2 estimates)
    if not years:
        return

    is_rows = fin_data.get("income_statement", [])
    rat_rows = fin_data.get("ratios", [])

    # Distill key rows
    def _find(rows, label_fragment):
        for r in rows:
            if label_fragment.lower() in r[0].lower():
                # Grab last 3 data cols
                data_cols = r[1:]
                if len(data_cols) >= len(fin_data.get("years", [])):
                    return data_cols[-3:]
                return data_cols[-len(years):]
        return [_NA] * len(years)

    mini = [
        ("Y/E",          years),
        ("Sales",        _find(is_rows, "**Sales")),
        ("EBIT Margin %",_find(is_rows, "% of Net Sales")),
        ("PAT",          _find(is_rows, "**Adjusted PAT")),
        ("EPS",          _find(rat_rows, "**EPS")),
        ("EPS Gr. %",    _find(is_rows, "Change")),
        ("BV/Sh.",       _find(rat_rows, "Book Value")),
        ("RoE %",        _find(rat_rows, "**RoE")),
        ("RoCE %",       _find(rat_rows, "RoCE")),
        ("P/E (x)",      _find(rat_rows, "**P/E")),
        ("EV/EBITDA (x)",_find(rat_rows, "EV/EBITDA")),
    ]

    p = cell.add_paragraph()
    r = p.add_run("Financials & Valuations")
    r.bold = True
    r.font.size = _pt(8)
    _color_run(r, _NAV)
    _shade_paragraph(p, _LBKG)

    t = cell.add_table(rows=len(mini), cols=1 + len(years))
    t.style = "Table Grid"
    for ri, (label, vals) in enumerate(mini):
        bg = _LBLU if ri == 0 else (_LGY if ri % 2 else _WHT)
        t.rows[ri].cells[0].text = label
        _set_cell_bg(t.rows[ri].cells[0], bg)
        if t.rows[ri].cells[0].paragraphs[0].runs:
            t.rows[ri].cells[0].paragraphs[0].runs[0].font.size = _pt(7)
            t.rows[ri].cells[0].paragraphs[0].runs[0].bold = (ri == 0)
            if ri == 0:
                _color_run(t.rows[ri].cells[0].paragraphs[0].runs[0], _WHT)
        for ci, v in enumerate(vals[:len(years)], 1):
            c = t.rows[ri].cells[ci]
            c.text = str(v)
            _set_cell_bg(c, bg)
            if c.paragraphs[0].runs:
                rr = c.paragraphs[0].runs[0]
                rr.font.size = _pt(7)
                rr.bold = (ri == 0)
                if ri == 0:
                    _color_run(rr, _WHT)


def _add_shareholding_table(cell, state: ResearchState):
    """Shareholding pattern block on cover."""
    mktdata = _payload(state, "04_market_data", "market_data", {})
    sholding = mktdata.get("shareholding", {}) or {}
    if not sholding:
        return

    p = cell.add_paragraph()
    r = p.add_run("Shareholding Pattern (%)")
    r.bold = True
    r.font.size = _pt(8)
    _color_run(r, _NAV)
    _shade_paragraph(p, _LBKG)

    promoter = sholding.get("promoter_pct")
    dii      = sholding.get("dii_pct")
    fii      = sholding.get("fii_pct")
    others   = sholding.get("others_pct")

    rows = [
        ("As On", "Current"),
        ("Promoter", f"{promoter:.1f}" if promoter else _NA),
        ("DII",      f"{dii:.1f}"  if dii      else _NA),
        ("FII",      f"{fii:.1f}"  if fii      else _NA),
        ("Others",   f"{others:.1f}" if others  else _NA),
    ]

    t = cell.add_table(rows=len(rows), cols=2)
    t.style = "Table Grid"
    for ri, (k, v) in enumerate(rows):
        bg = _LBLU if ri == 0 else (_LGY if ri % 2 else _WHT)
        for ci, txt in enumerate((k, v)):
            c = t.rows[ri].cells[ci]
            c.text = txt
            _set_cell_bg(c, bg)
            if c.paragraphs[0].runs:
                rr = c.paragraphs[0].runs[0]
                rr.font.size = _pt(7)
                rr.bold = (ri == 0)
                if ri == 0:
                    _color_run(rr, _WHT)


# ═══════════════════════════════════════════════════════════════════════════
# XML / STYLE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _pt(size: int):
    from docx.shared import Pt
    return Pt(size)


def _set_margins(doc):
    from docx.shared import Cm
    for sec in doc.sections:
        sec.top_margin    = Cm(2.0)
        sec.bottom_margin = Cm(2.0)
        sec.left_margin   = Cm(2.0)
        sec.right_margin  = Cm(2.0)


def _set_cell_bg(cell, hex_color: str):
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color.lstrip("#").upper())
    # Remove existing shd if any
    for existing in tcPr.findall(qn("w:shd")):
        tcPr.remove(existing)
    tcPr.append(shd)


def _shade_paragraph(p, hex_color: str):
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color.lstrip("#").upper())
    for existing in pPr.findall(qn("w:shd")):
        pPr.remove(existing)
    pPr.append(shd)


def _color_run(run, hex_color: str):
    from docx.shared import RGBColor
    h = hex_color.lstrip("#")
    run.font.color.rgb = RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _remove_table_borders(table):
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else OxmlElement("w:tblPr")
    tblBorders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "none")
        tblBorders.append(el)
    for existing in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(existing)
    tblPr.append(tblBorders)


def _section_bar(doc, text: str, color: str = _TEAL):
    """Full-width coloured section header bar (MOFSL style)."""
    p = doc.add_paragraph()
    _shade_paragraph(p, color)
    run = p.add_run("  " + text)
    run.bold = True
    run.font.size = _pt(10)
    _color_run(run, _WHT)
    return p


def _add_cell_section_bar(cell, text: str, color: str = _TEAL):
    p = cell.add_paragraph()
    _shade_paragraph(p, color)
    run = p.add_run("  " + text)
    run.bold = True
    run.font.size = _pt(9)
    _color_run(run, _WHT)


def _subsection_label(doc, text: str):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = _pt(10)
    _color_run(run, _NAV)


def _mini_table(cell, rows: list, hdr_color: str = _LBLU,
                hdr_text: str = "", key_width=None) -> Any:
    if hdr_text:
        p = cell.add_paragraph()
        r = p.add_run(hdr_text)
        r.bold = True
        r.font.size = _pt(8)
        _color_run(r, _NAV)
        _shade_paragraph(p, _LBKG)

    t = cell.add_table(rows=len(rows), cols=2)
    t.style = "Table Grid"
    for ri, (k, v) in enumerate(rows):
        bg = _LGY if ri % 2 else _WHT
        for ci, txt in enumerate((k, v)):
            c = t.rows[ri].cells[ci]
            c.text = str(txt)
            _set_cell_bg(c, bg)
            if c.paragraphs[0].runs:
                c.paragraphs[0].runs[0].font.size = _pt(8)
    return t


def _add_bullets_to_cell(cell, items: list, size: int = 9):
    for item in items:
        if item:
            p = cell.add_paragraph()
            run = p.add_run(f"■  {item}")
            run.font.size = _pt(size)


def _add_content_paragraphs(doc, text: str, max_chars: int = 2000):
    if not text:
        return
    text = text[:max_chars]
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("##") or block.startswith("**"):
            p = doc.add_paragraph()
            run = p.add_run(block.lstrip("#*").strip())
            run.bold = True
            run.font.size = _pt(10)
        else:
            p = doc.add_paragraph(block)
            if p.runs:
                p.runs[0].font.size = _pt(10)


# ═══════════════════════════════════════════════════════════════════════════
# FORMATTING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _payload(state: ResearchState, agent_id: str, key: str, default=None):
    out = state.agent_outputs.get(agent_id)
    if out is None:
        return default
    return out.payload.get(key, default)


def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return _NA
    try:
        f = float(v)
        if pct:
            return f"{f:.1f}"
        if abs(f) >= 1_000_000:
            return f"{f/1_000_000:.1f}M"
        if abs(f) >= 1_000:
            return f"{f:,.0f}"
        return f"{f:.2f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_num(v, pct: bool = False) -> str:
    if v is None:
        return _NA
    try:
        f = float(v)
        if pct:
            return f"{f:.1f}"
        if abs(f) >= 1_000:
            return f"{f:,.1f}"
        return f"{f:.1f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_metric(v, label: str) -> str:
    if v is None:
        return _NA
    pct_hints = ("%", "margin", "growth", "rate", "yield", "payout")
    is_pct = any(h in label.lower() for h in pct_hints)
    return _fmt_num(v, pct=is_pct)


def _extract_headline(text: str, company: str) -> str:
    if not text:
        return f"{company}: Equity Research"
    first_line = text.strip().split("\n")[0].strip()
    return first_line[:100] if first_line else f"{company}: Equity Research"


def _extract_bullets(text: str, max_bullets: int = 5) -> List[str]:
    if not text:
        return []
    bullets = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(("- ", "• ", "* ", "■ ")):
            bullets.append(line[2:].strip())
        elif line.startswith(("1.", "2.", "3.", "4.", "5.")):
            bullets.append(line[2:].strip())
        if len(bullets) >= max_bullets:
            break
    if not bullets:
        sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if len(s) > 20]
        bullets = sentences[:max_bullets]
    return [b[:150] for b in bullets if b]
