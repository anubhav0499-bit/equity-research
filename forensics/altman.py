"""
Altman Z-Score — bankruptcy prediction model.
Uses the Emerging Market (EM) variant for non-US companies (Altman 2000).
Original model for US manufacturers also supported.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AltmanResult:
    z_score: float
    x1: Optional[float]  # Working Capital / Total Assets
    x2: Optional[float]  # Retained Earnings / Total Assets
    x3: Optional[float]  # EBIT / Total Assets
    x4: Optional[float]  # Equity / Total Liabilities
    x5: Optional[float]  # Revenue / Total Assets  (not used in EM model)
    model_used: str      # "ORIGINAL" or "EM"
    classification: str
    risk_level: str
    interpretation: str
    safe_zone: float
    distress_zone: float


def compute_altman_z_score(
    financials: dict,
    market_cap: Optional[float] = None,
    use_em_model: bool = True,
    safe_threshold: float = 2.60,
    distress_threshold: float = 1.10,
) -> AltmanResult:
    """
    Compute Altman Z-Score.

    EM Model (use_em_model=True):  Z' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
    Original (US manufacturers):  Z  = 1.20*X1 + 1.40*X2 + 3.30*X3 + 0.60*X4 + 1.00*X5

    Args:
        financials: dict with current year financial data
        market_cap: market capitalisation (same units as financials)
        use_em_model: True for Emerging Market / non-manufacturing variant
    """

    def gv(d: dict, *keys) -> Optional[float]:
        for k in keys:
            v = d.get(k)
            if v is not None:
                try:
                    f = float(v)
                    import math
                    return None if (math.isnan(f) or math.isinf(f)) else f
                except (TypeError, ValueError):
                    pass
        return None

    ca    = gv(financials, "total_current_assets")
    cl    = gv(financials, "total_current_liabilities")
    ta    = gv(financials, "total_assets")
    re    = gv(financials, "retained_earnings")
    ebit  = gv(financials, "ebit", "operating_income")
    te    = gv(financials, "total_equity", "shareholder_equity")
    tl    = gv(financials, "total_liabilities")
    rev   = gv(financials, "revenue", "total_revenue")

    # Working capital
    wc = (ca - cl) if ca and cl else None

    # X1: Working Capital / Total Assets
    x1 = (wc / ta) if wc is not None and ta and ta > 0 else None

    # X2: Retained Earnings / Total Assets
    x2 = (re / ta) if re is not None and ta and ta > 0 else None

    # X3: EBIT / Total Assets
    x3 = (ebit / ta) if ebit is not None and ta and ta > 0 else None

    # X4: Book Value Equity / Total Liabilities (EM uses book value, not market value)
    equity_val = market_cap if (market_cap and not use_em_model) else te
    x4 = (equity_val / tl) if equity_val and tl and tl > 0 else None

    # X5: Revenue / Total Assets (original model only)
    x5 = (rev / ta) if rev and ta and ta > 0 else None

    # ── Compute Z-Score ───────────────────────────────────────────
    available = [v for v in [x1, x2, x3, x4] if v is not None]
    if len(available) < 2:
        z_score = 0.0
        model_used = "INSUFFICIENT_DATA"
    elif use_em_model:
        z_score = (
            6.56 * (x1 or 0)
            + 3.26 * (x2 or 0)
            + 6.72 * (x3 or 0)
            + 1.05 * (x4 or 0)
        )
        model_used = "EM"
    else:
        z_score = (
            1.20 * (x1 or 0)
            + 1.40 * (x2 or 0)
            + 3.30 * (x3 or 0)
            + 0.60 * (x4 or 0)
            + 1.00 * (x5 or 0)
        )
        model_used = "ORIGINAL"
        safe_threshold = 2.99
        distress_threshold = 1.81

    # ── Classification ────────────────────────────────────────────
    if model_used == "INSUFFICIENT_DATA":
        classification = "INSUFFICIENT_DATA"
        risk_level = "UNKNOWN"
        interpretation = "Insufficient financial data to compute Altman Z-Score."
    elif z_score > safe_threshold:
        classification = "SAFE_ZONE"
        risk_level = "LOW"
        interpretation = (
            f"Z-Score of {z_score:.2f} is in the safe zone (>{safe_threshold}). "
            "Low financial distress risk."
        )
    elif z_score > distress_threshold:
        classification = "GREY_ZONE"
        risk_level = "MEDIUM"
        interpretation = (
            f"Z-Score of {z_score:.2f} is in the grey zone ({distress_threshold}–{safe_threshold}). "
            "Moderate distress indicators — monitor closely."
        )
    else:
        classification = "DISTRESS_ZONE"
        risk_level = "CRITICAL"
        interpretation = (
            f"Z-Score of {z_score:.2f} is in the distress zone (<{distress_threshold}). "
            "High probability of financial distress."
        )

    return AltmanResult(
        z_score=round(z_score, 4),
        x1=round(x1, 4) if x1 else None,
        x2=round(x2, 4) if x2 else None,
        x3=round(x3, 4) if x3 else None,
        x4=round(x4, 4) if x4 else None,
        x5=round(x5, 4) if x5 else None,
        model_used=model_used,
        classification=classification,
        risk_level=risk_level,
        interpretation=interpretation,
        safe_zone=safe_threshold,
        distress_zone=distress_threshold,
    )
