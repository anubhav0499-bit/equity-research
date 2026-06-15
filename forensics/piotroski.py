"""
Piotroski F-Score — 9-signal financial strength scoring model.
Score 8-9: Strong. Score 0-2: Weak. Used for financial health + fraud screening.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PiotroskiResult:
    f_score: int
    profitability_score: int  # max 4
    leverage_score: int       # max 3
    efficiency_score: int     # max 2 (operating efficiency)
    signals: dict
    classification: str
    risk_level: str
    interpretation: str


def compute_piotroski_f_score(current: dict, prior: dict) -> PiotroskiResult:
    """
    Compute Piotroski F-Score. Requires two consecutive year balance sheets and income statements.

    Profitability (4 signals):
      F1: ROA > 0
      F2: Operating Cash Flow > 0
      F3: Change in ROA > 0
      F4: Accruals (CFO/Assets > ROA)

    Leverage / Liquidity / Source of Funds (3 signals):
      F5: Decrease in long-term leverage
      F6: Increase in current ratio
      F7: No new shares issued

    Operating Efficiency (2 signals):
      F8: Increase in gross margin
      F9: Increase in asset turnover
    """

    def gv(d: dict, *keys) -> Optional[float]:
        for k in keys:
            v = d.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return None

    # Current year
    ni_c    = gv(current, "net_income")
    ta_c    = gv(current, "total_assets")
    cfo_c   = gv(current, "cfo", "cash_from_operations")
    ltd_c   = gv(current, "long_term_debt")
    ca_c    = gv(current, "total_current_assets")
    cl_c    = gv(current, "total_current_liabilities")
    shares_c = gv(current, "shares_outstanding", "shares_diluted")
    rev_c   = gv(current, "revenue", "total_revenue")
    gp_c    = gv(current, "gross_profit")

    # Prior year
    ni_p    = gv(prior, "net_income")
    ta_p    = gv(prior, "total_assets")
    cfo_p   = gv(prior, "cfo", "cash_from_operations")
    ltd_p   = gv(prior, "long_term_debt")
    ca_p    = gv(prior, "total_current_assets")
    cl_p    = gv(prior, "total_current_liabilities")
    shares_p = gv(prior, "shares_outstanding", "shares_diluted")
    rev_p   = gv(prior, "revenue", "total_revenue")
    gp_p    = gv(prior, "gross_profit")

    signals: dict[str, Optional[int]] = {}

    # ── Profitability ─────────────────────────────────────────────

    # F1: ROA > 0
    roa_c = (ni_c / ta_c) if ni_c and ta_c and ta_c > 0 else None
    signals["F1_roa_positive"] = (1 if roa_c and roa_c > 0 else 0) if roa_c is not None else None

    # F2: CFO > 0
    signals["F2_cfo_positive"] = (1 if cfo_c and cfo_c > 0 else 0) if cfo_c is not None else None

    # F3: Delta ROA > 0
    roa_p = (ni_p / ta_p) if ni_p and ta_p and ta_p > 0 else None
    if roa_c is not None and roa_p is not None:
        signals["F3_roa_improving"] = 1 if roa_c > roa_p else 0
    else:
        signals["F3_roa_improving"] = None

    # F4: Accruals (CFO/Assets > ROA)
    cfo_assets = (cfo_c / ta_c) if cfo_c and ta_c and ta_c > 0 else None
    if cfo_assets is not None and roa_c is not None:
        signals["F4_low_accruals"] = 1 if cfo_assets > roa_c else 0
    else:
        signals["F4_low_accruals"] = None

    # ── Leverage / Liquidity / Funding ────────────────────────────

    # F5: Decrease in leverage (LTD/TotalAssets)
    lev_c = (ltd_c / ta_c) if ltd_c is not None and ta_c and ta_c > 0 else None
    lev_p = (ltd_p / ta_p) if ltd_p is not None and ta_p and ta_p > 0 else None
    if lev_c is not None and lev_p is not None:
        signals["F5_leverage_decrease"] = 1 if lev_c < lev_p else 0
    else:
        signals["F5_leverage_decrease"] = None

    # F6: Current ratio improvement
    cr_c = (ca_c / cl_c) if ca_c and cl_c and cl_c > 0 else None
    cr_p = (ca_p / cl_p) if ca_p and cl_p and cl_p > 0 else None
    if cr_c is not None and cr_p is not None:
        signals["F6_current_ratio_improve"] = 1 if cr_c > cr_p else 0
    else:
        signals["F6_current_ratio_improve"] = None

    # F7: No new dilution
    if shares_c is not None and shares_p is not None:
        signals["F7_no_dilution"] = 1 if shares_c <= shares_p * 1.02 else 0
    else:
        signals["F7_no_dilution"] = None

    # ── Efficiency ────────────────────────────────────────────────

    # F8: Gross margin improvement
    gm_c = (gp_c / rev_c) if gp_c and rev_c and rev_c > 0 else None
    gm_p = (gp_p / rev_p) if gp_p and rev_p and rev_p > 0 else None
    if gm_c is not None and gm_p is not None:
        signals["F8_gross_margin_improve"] = 1 if gm_c > gm_p else 0
    else:
        signals["F8_gross_margin_improve"] = None

    # F9: Asset turnover improvement
    at_c = (rev_c / ta_c) if rev_c and ta_c and ta_c > 0 else None
    at_p = (rev_p / ta_p) if rev_p and ta_p and ta_p > 0 else None
    if at_c is not None and at_p is not None:
        signals["F9_asset_turnover_improve"] = 1 if at_c > at_p else 0
    else:
        signals["F9_asset_turnover_improve"] = None

    # ── Compute scores ────────────────────────────────────────────
    prof_keys = ["F1_roa_positive", "F2_cfo_positive", "F3_roa_improving", "F4_low_accruals"]
    lev_keys  = ["F5_leverage_decrease", "F6_current_ratio_improve", "F7_no_dilution"]
    eff_keys  = ["F8_gross_margin_improve", "F9_asset_turnover_improve"]

    def score_group(keys):
        return sum(signals[k] for k in keys if signals.get(k) is not None)

    prof = score_group(prof_keys)
    lev  = score_group(lev_keys)
    eff  = score_group(eff_keys)
    f_score = prof + lev + eff

    # ── Classification ────────────────────────────────────────────
    if f_score >= 8:
        classification = "STRONG"
        risk_level = "LOW"
        interpretation = f"F-Score {f_score}/9 — high financial strength. Low financial distress risk."
    elif f_score >= 5:
        classification = "MODERATE"
        risk_level = "MEDIUM"
        interpretation = f"F-Score {f_score}/9 — average financial strength."
    elif f_score >= 3:
        classification = "WEAK"
        risk_level = "HIGH"
        interpretation = f"F-Score {f_score}/9 — below-average financial strength. Elevated distress indicators."
    else:
        classification = "VERY_WEAK"
        risk_level = "CRITICAL"
        interpretation = f"F-Score {f_score}/9 — poor financial health on most dimensions. High distress risk."

    return PiotroskiResult(
        f_score=f_score,
        profitability_score=prof,
        leverage_score=lev,
        efficiency_score=eff,
        signals={k: v for k, v in signals.items()},
        classification=classification,
        risk_level=risk_level,
        interpretation=interpretation,
    )
