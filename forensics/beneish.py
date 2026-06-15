"""
Beneish M-Score — 8-variable earnings manipulation model.
Threshold: M > -1.78 suggests possible manipulation.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BeneishResult:
    m_score: float
    dsri: Optional[float]   # Days Sales Receivable Index
    gmi: Optional[float]    # Gross Margin Index
    aqi: Optional[float]    # Asset Quality Index
    sgi: Optional[float]    # Sales Growth Index
    depi: Optional[float]   # Depreciation Index
    sgai: Optional[float]   # SGA Index
    lvgi: Optional[float]   # Leverage Index
    tata: Optional[float]   # Total Accruals to Total Assets
    classification: str
    risk_level: str
    interpretation: str
    data_quality: str


def compute_beneish_m_score(
    current: dict,
    prior: dict,
    manipulation_threshold: float = -1.78,
    high_risk_threshold: float = -1.0,
) -> BeneishResult:
    """
    Compute Beneish M-Score using two consecutive year financials.

    Args:
        current: dict with current year financial data (millions)
        prior: dict with prior year financial data (millions)
        manipulation_threshold: default -1.78 (Beneish 1999)
        high_risk_threshold: above this level = high risk

    Returns:
        BeneishResult with M-score and all 8 variables
    """

    def safe(val) -> Optional[float]:
        try:
            v = float(val)
            return None if (math.isnan(v) or math.isinf(v)) else v
        except (TypeError, ValueError):
            return None

    def g(d: dict, *keys):
        for k in keys:
            v = d.get(k)
            if v is not None:
                return safe(v)
        return None

    # ── Extract current year ──────────────────────────────────────
    rev_c    = g(current, "revenue", "total_revenue")
    ar_c     = g(current, "accounts_receivable", "net_receivables")
    gp_c     = g(current, "gross_profit")
    cogs_c   = g(current, "cogs", "cost_of_revenue")
    assets_c = g(current, "total_assets")
    sga_c    = g(current, "sga", "selling_general_administrative")
    depr_c   = g(current, "depreciation", "depreciation_amortization")
    ltd_c    = g(current, "long_term_debt")
    ni_c     = g(current, "net_income")
    cfo_c    = g(current, "cfo", "cash_from_operations")
    capex_c  = g(current, "capex", "capital_expenditures")
    ca_c     = g(current, "total_current_assets")
    cl_c     = g(current, "total_current_liabilities")
    cash_c   = g(current, "cash_and_equivalents", "cash_equivalents")

    # ── Extract prior year ────────────────────────────────────────
    rev_p    = g(prior, "revenue", "total_revenue")
    ar_p     = g(prior, "accounts_receivable", "net_receivables")
    gp_p     = g(prior, "gross_profit")
    cogs_p   = g(prior, "cogs", "cost_of_revenue")
    assets_p = g(prior, "total_assets")
    sga_p    = g(prior, "sga", "selling_general_administrative")
    depr_p   = g(prior, "depreciation", "depreciation_amortization")
    ltd_p    = g(prior, "long_term_debt")
    ca_p     = g(prior, "total_current_assets")
    cl_p     = g(prior, "total_current_liabilities")
    cash_p   = g(prior, "cash_and_equivalents", "cash_equivalents")

    # ── Compute 8 variables ───────────────────────────────────────

    # 1. DSRI = (AR_t/Rev_t) / (AR_{t-1}/Rev_{t-1})
    dsri = None
    if ar_c and rev_c and ar_p and rev_p and rev_c > 0 and rev_p > 0:
        dsri = (ar_c / rev_c) / (ar_p / rev_p)

    # 2. GMI = GP Margin_{t-1} / GP Margin_t
    gmi = None
    if gp_c and gp_p and rev_c and rev_p and rev_c > 0 and rev_p > 0:
        gm_c = gp_c / rev_c
        gm_p = gp_p / rev_p
        if gm_c > 0:
            gmi = gm_p / gm_c

    # 3. AQI = (1 - (CA_t + NCA_ppe_t) / TA_t) / (1 - (CA_{t-1} + NCA_ppe_{t-1}) / TA_{t-1})
    aqi = None
    if assets_c and assets_p and ca_c and ca_p:
        lta_c = assets_c - ca_c
        lta_p = assets_p - ca_p
        if assets_c > 0 and assets_p > 0:
            aq_c = 1 - (ca_c / assets_c) if assets_c else 0
            aq_p = 1 - (ca_p / assets_p) if assets_p else 0
            if aq_p > 0:
                aqi = aq_c / aq_p

    # 4. SGI = Rev_t / Rev_{t-1}
    sgi = None
    if rev_c and rev_p and rev_p > 0:
        sgi = rev_c / rev_p

    # 5. DEPI = (Dep_{t-1} / (Dep_{t-1} + PPE_{t-1})) / (Dep_t / (Dep_t + PPE_t))
    depi = None
    if depr_c and depr_p:
        ppe_c = g(current, "net_ppe", "gross_ppe") or 0
        ppe_p = g(prior, "net_ppe", "gross_ppe") or 0
        denom_p = depr_p + ppe_p
        denom_c = depr_c + ppe_c
        if denom_c > 0 and denom_p > 0:
            depi = (depr_p / denom_p) / (depr_c / denom_c)

    # 6. SGAI = (SGA_t / Rev_t) / (SGA_{t-1} / Rev_{t-1})
    sgai = None
    if sga_c and sga_p and rev_c and rev_p and rev_c > 0 and rev_p > 0:
        sgai = (sga_c / rev_c) / (sga_p / rev_p)

    # 7. LVGI = ((LTD_t + CL_t) / TA_t) / ((LTD_{t-1} + CL_{t-1}) / TA_{t-1})
    lvgi = None
    if assets_c and assets_p and assets_c > 0 and assets_p > 0:
        debt_c = (ltd_c or 0) + (cl_c or 0)
        debt_p = (ltd_p or 0) + (cl_p or 0)
        if debt_c and debt_p:
            lvgi = (debt_c / assets_c) / (debt_p / assets_p)

    # 8. TATA = (Net Income - CFO) / Total Assets
    tata = None
    if ni_c is not None and cfo_c is not None and assets_c:
        tata = (ni_c - cfo_c) / assets_c

    # ── M-Score formula ───────────────────────────────────────────
    components_available = sum(1 for v in [dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata] if v is not None)
    data_quality = "FULL" if components_available >= 7 else ("PARTIAL" if components_available >= 4 else "INSUFFICIENT")

    if components_available < 4:
        m_score = 0.0  # Cannot compute
    else:
        m_score = (
            -4.840
            + 0.920 * (dsri or 1.0)
            + 0.528 * (gmi or 1.0)
            + 0.404 * (aqi or 1.0)
            + 0.892 * (sgi or 1.0)
            + 0.115 * (depi or 1.0)
            - 0.172 * (sgai or 1.0)
            + 4.679 * (tata or 0.0)
            - 0.327 * (lvgi or 1.0)
        )

    # ── Classification ────────────────────────────────────────────
    if data_quality == "INSUFFICIENT":
        classification = "INSUFFICIENT_DATA"
        risk_level = "UNKNOWN"
        interpretation = "Insufficient data to compute M-Score reliably."
    elif m_score > high_risk_threshold:
        classification = "LIKELY_MANIPULATOR"
        risk_level = "CRITICAL"
        interpretation = (
            f"M-Score of {m_score:.2f} exceeds high-risk threshold ({high_risk_threshold}). "
            "Strong statistical evidence of earnings manipulation."
        )
    elif m_score > manipulation_threshold:
        classification = "POSSIBLE_MANIPULATOR"
        risk_level = "HIGH"
        interpretation = (
            f"M-Score of {m_score:.2f} in the grey zone (between {manipulation_threshold} and {high_risk_threshold}). "
            "Heightened manipulation risk — investigate further."
        )
    else:
        classification = "UNLIKELY_MANIPULATOR"
        risk_level = "LOW"
        interpretation = (
            f"M-Score of {m_score:.2f} below manipulation threshold ({manipulation_threshold}). "
            "No statistical evidence of earnings manipulation from M-Score alone."
        )

    return BeneishResult(
        m_score=round(m_score, 4),
        dsri=round(dsri, 4) if dsri else None,
        gmi=round(gmi, 4) if gmi else None,
        aqi=round(aqi, 4) if aqi else None,
        sgi=round(sgi, 4) if sgi else None,
        depi=round(depi, 4) if depi else None,
        sgai=round(sgai, 4) if sgai else None,
        lvgi=round(lvgi, 4) if lvgi else None,
        tata=round(tata, 4) if tata else None,
        classification=classification,
        risk_level=risk_level,
        interpretation=interpretation,
        data_quality=data_quality,
    )
