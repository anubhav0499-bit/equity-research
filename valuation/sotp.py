"""
Sum-of-the-Parts (SOTP) Valuation Engine
Computes segment-level valuations using individual EV/EBITDA or revenue multiples per segment,
sums to enterprise value, then backs into equity value via net debt.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


@dataclass
class SOTPSegment:
    """A single business segment for SOTP valuation."""
    name: str
    segment_ebitda: Optional[float] = None          # USD millions
    segment_revenue: Optional[float] = None         # USD millions
    ev_ebitda_multiple: Optional[float] = None      # Applied multiple
    ev_revenue_multiple: Optional[float] = None     # Alternative: EV/Revenue
    minority_interest_pct: float = 0.0              # % owned by minorities (0-100)
    description: str = ""


@dataclass
class SOTPResult:
    """Full SOTP computation output."""
    ticker: str
    segments: list[SOTPSegment]
    segment_values: dict[str, float]                # Segment name → EV contribution
    gross_segment_ev: float                         # Sum of all segment EVs
    net_debt: float                                 # Net debt (USD M)
    minority_interests: float                       # Total minority interests deducted
    equity_value: float                             # Gross EV - Net Debt - Minority
    shares_outstanding: Optional[float] = None      # Millions
    implied_price: Optional[float] = None           # Per share
    nav_discount_pct: Optional[float] = None        # If conglomerate discount applied
    notes: list[str] = field(default_factory=list)
    data_quality: str = "FULL"                      # FULL / PARTIAL / INSUFFICIENT


def compute_sotp(
    ticker: str,
    segments: list[SOTPSegment],
    net_debt: float,
    shares_outstanding: Optional[float] = None,
    conglomerate_discount_pct: float = 0.0,
) -> SOTPResult:
    """
    Compute SOTP valuation across all segments.

    Args:
        ticker: Company ticker symbol
        segments: List of SOTPSegment objects with segment financials and multiples
        net_debt: Net debt in USD millions (positive = net debt, negative = net cash)
        shares_outstanding: Shares outstanding in millions
        conglomerate_discount_pct: Optional holding company discount (0-30%)

    Returns:
        SOTPResult with full computation details
    """
    if not segments:
        raise ValueError("SOTP requires at least one business segment")

    segment_values: dict[str, float] = {}
    notes: list[str] = []
    data_quality = "FULL"
    incomplete_segments = 0

    for seg in segments:
        val = _value_segment(seg, notes)
        if val is not None:
            # Apply minority interest discount
            if seg.minority_interest_pct > 0:
                ownership_pct = (100 - seg.minority_interest_pct) / 100
                val = val * ownership_pct
                notes.append(f"{seg.name}: Applied {100 - seg.minority_interest_pct:.0f}% ownership factor "
                              f"(minority interest: {seg.minority_interest_pct:.0f}%)")
            segment_values[seg.name] = round(val, 1)
        else:
            segment_values[seg.name] = 0.0
            incomplete_segments += 1
            notes.append(f"{seg.name}: Insufficient data — segment valued at zero")

    if incomplete_segments > 0:
        data_quality = "PARTIAL" if incomplete_segments < len(segments) else "INSUFFICIENT"

    gross_ev = sum(segment_values.values())

    # Conglomerate/holding company discount
    if conglomerate_discount_pct > 0:
        discount_amount = gross_ev * conglomerate_discount_pct / 100
        gross_ev_post_discount = gross_ev - discount_amount
        notes.append(f"Applied {conglomerate_discount_pct:.0f}% conglomerate discount: "
                     f"${gross_ev:.0f}M → ${gross_ev_post_discount:.0f}M")
        nav_discount_pct = conglomerate_discount_pct
        gross_ev = gross_ev_post_discount
    else:
        nav_discount_pct = None

    # Total minority interests (market value if we have prices; otherwise already applied per-segment above)
    total_minority = 0.0

    equity_value = gross_ev - net_debt - total_minority
    notes.append(f"Equity value = Gross EV ${gross_ev:.0f}M - Net Debt ${net_debt:.0f}M = ${equity_value:.0f}M")

    implied_price = None
    if shares_outstanding and shares_outstanding > 0:
        implied_price = round(equity_value / shares_outstanding, 2)
        notes.append(f"Implied price = ${equity_value:.0f}M / {shares_outstanding:.1f}M shares = ${implied_price:.2f}")

    return SOTPResult(
        ticker=ticker,
        segments=segments,
        segment_values=segment_values,
        gross_segment_ev=round(gross_ev, 1),
        net_debt=round(net_debt, 1),
        minority_interests=round(total_minority, 1),
        equity_value=round(equity_value, 1),
        shares_outstanding=shares_outstanding,
        implied_price=implied_price,
        nav_discount_pct=nav_discount_pct,
        notes=notes,
        data_quality=data_quality,
    )


def _value_segment(seg: SOTPSegment, notes: list[str]) -> Optional[float]:
    """Value a single segment using EV/EBITDA (preferred) or EV/Revenue (fallback)."""
    if seg.ev_ebitda_multiple is not None and seg.segment_ebitda is not None:
        val = seg.segment_ebitda * seg.ev_ebitda_multiple
        notes.append(f"{seg.name}: ${seg.segment_ebitda:.0f}M EBITDA × {seg.ev_ebitda_multiple:.1f}x = ${val:.0f}M EV")
        return val

    if seg.ev_revenue_multiple is not None and seg.segment_revenue is not None:
        val = seg.segment_revenue * seg.ev_revenue_multiple
        notes.append(f"{seg.name}: ${seg.segment_revenue:.0f}M Revenue × {seg.ev_revenue_multiple:.1f}x = ${val:.0f}M EV")
        return val

    return None


def build_sotp_from_peer_multiples(
    ticker: str,
    segments: list[dict],          # [{"name": str, "ebitda": float, "sector": str}, ...]
    peer_multiples: dict,          # From MarketDataAgent: {sector: ev_ebitda}
    net_debt: float,
    shares_outstanding: Optional[float] = None,
    conglomerate_discount_pct: float = 0.0,
) -> SOTPResult:
    """
    Convenience wrapper: build SOTP segments using sector peer multiples.
    Used when no explicit multiples are provided.
    """
    # Sector-to-EV/EBITDA fallback multiples (Damodaran 2024 median approximations)
    SECTOR_MULTIPLES: dict[str, float] = {
        "Technology": 18.0,
        "Consumer Staples": 12.0,
        "Consumer Discretionary": 10.0,
        "Healthcare": 14.0,
        "Financials": 10.0,
        "Energy": 7.0,
        "Industrials": 9.0,
        "Real Estate": 15.0,
        "Telecommunications": 8.0,
        "Materials": 8.0,
        "Utilities": 10.0,
    }

    sotp_segments = []
    for seg_data in segments:
        name = seg_data.get("name", "Unknown")
        ebitda = seg_data.get("ebitda")
        revenue = seg_data.get("revenue")
        sector = seg_data.get("sector", "")
        minority_pct = seg_data.get("minority_interest_pct", 0.0)

        # Use peer multiples from state if available, else fallback to sector defaults
        multiple = (
            peer_multiples.get(sector)
            or peer_multiples.get(name)
            or SECTOR_MULTIPLES.get(sector, 10.0)
        )

        sotp_segments.append(SOTPSegment(
            name=name,
            segment_ebitda=ebitda,
            segment_revenue=revenue,
            ev_ebitda_multiple=multiple if ebitda is not None else None,
            ev_revenue_multiple=None,
            minority_interest_pct=minority_pct,
            description=seg_data.get("description", ""),
        ))

    return compute_sotp(ticker, sotp_segments, net_debt, shares_outstanding, conglomerate_discount_pct)


def sotp_to_dict(result: SOTPResult) -> dict:
    """Serialize SOTPResult to a plain dict for JSON storage."""
    return {
        "ticker": result.ticker,
        "gross_segment_ev": result.gross_segment_ev,
        "net_debt": result.net_debt,
        "minority_interests": result.minority_interests,
        "equity_value": result.equity_value,
        "shares_outstanding": result.shares_outstanding,
        "implied_price": result.implied_price,
        "nav_discount_pct": result.nav_discount_pct,
        "data_quality": result.data_quality,
        "segment_values": result.segment_values,
        "notes": result.notes,
        "segments": [
            {
                "name": s.name,
                "segment_ebitda": s.segment_ebitda,
                "segment_revenue": s.segment_revenue,
                "ev_ebitda_multiple": s.ev_ebitda_multiple,
                "ev_revenue_multiple": s.ev_revenue_multiple,
                "minority_interest_pct": s.minority_interest_pct,
                "implied_ev": result.segment_values.get(s.name),
                "description": s.description,
            }
            for s in result.segments
        ],
    }
