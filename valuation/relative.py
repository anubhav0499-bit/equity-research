"""
Relative Valuation Engine — EV/EBITDA, P/E, P/B, P/S based on peer multiples.
"""

from __future__ import annotations
from typing import Optional
from ..models.valuation import RelativeValuation, PeerMultiple, WACCInputs, Scenario


def build_relative_valuation(
    scenario: Scenario,
    wacc_inputs: WACCInputs,
    peer_multiples: list[PeerMultiple],
    current_ebitda: float,
    current_net_income: float,
    current_revenue: float,
    net_debt: float,
    shares_outstanding: float,
    current_price: float = 0.0,
    discount_pct: float = 0.0,  # Discount / premium to apply to peer multiples
) -> RelativeValuation:
    """
    Value the company using peer group multiples.

    discount_pct: negative = discount, positive = premium (e.g., -15 = 15% discount to peers)
    """
    if not peer_multiples:
        return RelativeValuation(
            scenario=scenario,
            wacc_inputs=wacc_inputs,
            shares_outstanding=shares_outstanding,
            methodology_note="No peer data available",
        )

    # Filter valid multiples
    ev_ebitda_list = [p.ev_ebitda for p in peer_multiples if p.ev_ebitda and 3 < p.ev_ebitda < 100]
    pe_list = [p.pe for p in peer_multiples if p.pe and 3 < p.pe < 200]
    pb_list = [p.pb for p in peer_multiples if p.pb and 0 < p.pb < 50]
    ps_list = [p.ps for p in peer_multiples if p.ps and 0 < p.ps < 50]

    def median(lst: list) -> Optional[float]:
        if not lst: return None
        s = sorted(lst)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    def apply_disc(v: Optional[float]) -> Optional[float]:
        if v is None: return None
        return round(v * (1 + discount_pct / 100), 2)

    target_ev_ebitda = median(ev_ebitda_list)
    target_pe = median(pe_list)
    applied_ev_ebitda = apply_disc(target_ev_ebitda)
    applied_pe = apply_disc(target_pe)

    # EV/EBITDA implied value
    implied_ev = None
    value_ev = None
    if applied_ev_ebitda and current_ebitda and current_ebitda > 0:
        implied_ev = round(applied_ev_ebitda * current_ebitda, 2)
        equity_from_ev = implied_ev - net_debt
        value_ev = round(equity_from_ev / shares_outstanding, 2) if shares_outstanding > 0 else None

    # P/E implied value
    value_pe = None
    if applied_pe and current_net_income and current_net_income > 0:
        total_equity_value = applied_pe * current_net_income
        value_pe = round(total_equity_value / shares_outstanding, 2) if shares_outstanding > 0 else None

    # Blended value
    values = [v for v in [value_ev, value_pe] if v is not None]
    blended = round(sum(values) / len(values), 2) if values else None

    return RelativeValuation(
        scenario=scenario,
        wacc_inputs=wacc_inputs,
        peer_group=peer_multiples,
        target_ev_ebitda=target_ev_ebitda,
        target_pe=target_pe,
        target_pb=median(pb_list),
        target_ps=median(ps_list),
        applied_ev_ebitda=applied_ev_ebitda,
        implied_ev_from_ebitda=implied_ev,
        applied_pe=applied_pe,
        implied_equity_from_pe=round(applied_pe * (current_net_income or 0), 2) if applied_pe else None,
        net_debt=net_debt,
        shares_outstanding=shares_outstanding,
        value_per_share_ev_ebitda=value_ev,
        value_per_share_pe=value_pe,
        blended_value_per_share=blended,
        current_price=current_price if current_price > 0 else None,
        methodology_note=(
            f"Peer median EV/EBITDA={target_ev_ebitda}, P/E={target_pe} | "
            f"Discount/premium={discount_pct:+.0f}% | Peers={len(peer_multiples)}"
        ),
    )
