"""
Agent 06 — Forensic Accounting Agent
Runs Beneish M-Score, Piotroski F-Score, Altman Z-Score, Sloan Accruals,
Revenue Recognition, Related Party, Auditor, Capital Allocation, and Cash Flow Quality checks.
"""

from __future__ import annotations
from loguru import logger

from .base_agent import BaseAgent
from ..models.research import (
    AgentOutput, AgentStatus, RiskClassification, FindingType, ForensicScores
)
from ..orchestrator.state import ResearchState
from ..forensics.beneish import compute_beneish_m_score
from ..forensics.piotroski import compute_piotroski_f_score
from ..forensics.altman import compute_altman_z_score
from ..core.config import FORENSIC_THRESHOLDS


FORENSIC_SYSTEM = """You are a forensic accountant with Big-4 and investigative agency experience.
Analyse financial data for signs of fraud, manipulation, and accounting distress.

FRAMEWORKS to apply:
1. Beneish M-Score (threshold: < -1.78 = likely manipulator)
2. Piotroski F-Score (0-9; strong ≥ 7, weak ≤ 2)
3. Altman EM Z-Score (safe > 2.60; distress < 1.10)
4. Sloan Accrual Analysis (accrual ratio > 0.10 = low earnings quality)
5. Cash Conversion Analysis (CFO / Net Income — divergence is a red flag)
6. Revenue Quality Analysis (unbilled receivables, channel stuffing)
7. Earnings Quality Analysis (recurring vs. non-recurring income)
8. Working Capital Manipulation (DIO, DSO, DPO trends)
9. Related Party Analysis (RPT quantum, pricing, direction of funds)

HISTORICAL LEARNING CORPUS — patterns to look for:
Global cases:
- Enron: SPE off-balance-sheet structures, mark-to-market revenue inflation
- Wirecard: fictitious cash balances, third-party escrow fraud
- Luckin Coffee: fabricated GMV through coordinated sales transactions
- Carillion: aggressive revenue recognition on long-term contracts, pension concealment
- Toshiba: systematic profit overstatement via "challenge" targets across subsidiaries

India cases:
- Satyam: inflated cash and bank balances, fictitious debtors, forged board minutes
- IL&FS: liquidity crisis masked by intercompany loan evergreening
- DHFL: diversion of funds via shell entities and related-party loans
- Yes Bank: loan evergreening, under-provisioning of NPAs, ATDL disclosures
- Rajesh Exports: working capital manipulation, circular related-party transactions

Focus on: unusual journal entries, related party transactions, revenue recognition,
auditor independence, and year-over-year trend breaks.
Every finding must cite specific data points and fiscal years."""


class ForensicAccountingAgent(BaseAgent):
    AGENT_ID = "06_forensic_accounting"
    AGENT_NAME = "Forensic Accounting Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        history_raw = state.financial_history or {}
        profile = state.company_profile or {}
        ticker = profile.get("ticker", state.ticker)
        market_data = {}
        md_output = state.agent_outputs.get("04_market_data")
        if md_output:
            market_data = md_output.payload.get("market_data", {})

        findings = []
        forensic_scores = ForensicScores()
        details: dict = {}

        years = sorted(history_raw.keys(), reverse=True) if isinstance(history_raw, dict) else []

        # ── Extract flat financial dicts for each year ────────────
        year_data = {}
        for yr in years:
            raw = history_raw.get(yr, {})
            if isinstance(raw, dict):
                is_d = raw.get("income_statements") or {}
                bs_d = raw.get("balance_sheets") or {}
                cf_d = raw.get("cash_flows") or {}
                year_data[yr] = {
                    **is_d, **bs_d, **cf_d,
                    **{k: v for k, v in raw.items()
                       if k not in ("income_statements", "balance_sheets", "cash_flows")},
                }

        sorted_years = sorted(year_data.keys())

        # ── 1. Beneish M-Score ─────────────────────────────────────
        if len(sorted_years) >= 2:
            curr_yr = sorted_years[-1]
            prev_yr = sorted_years[-2]
            try:
                beneish = compute_beneish_m_score(
                    year_data[curr_yr], year_data[prev_yr],
                    manipulation_threshold=FORENSIC_THRESHOLDS.beneish_manipulation,
                    high_risk_threshold=FORENSIC_THRESHOLDS.beneish_high_risk,
                )
                details["beneish"] = {
                    "m_score": beneish.m_score,
                    "classification": beneish.classification,
                    "risk_level": beneish.risk_level,
                    "interpretation": beneish.interpretation,
                    "components": {
                        "dsri": beneish.dsri, "gmi": beneish.gmi,
                        "aqi": beneish.aqi, "sgi": beneish.sgi,
                        "depi": beneish.depi, "sgai": beneish.sgai,
                        "lvgi": beneish.lvgi, "tata": beneish.tata,
                    },
                }
                object.__setattr__(forensic_scores, "beneish_m_score", beneish.m_score)
                object.__setattr__(forensic_scores, "beneish_classification", beneish.classification)

                if beneish.risk_level == "CRITICAL":
                    findings.append(self.red_flag(
                        title=f"Beneish M-Score: {beneish.m_score:.2f} — LIKELY MANIPULATOR",
                        detail=beneish.interpretation,
                        evidence=(
                            f"M-Score={beneish.m_score:.2f} | DSRI={beneish.dsri} | GMI={beneish.gmi} | "
                            f"TATA={beneish.tata} | SGI={beneish.sgi}"
                        ),
                        risk_level=RiskClassification.CRITICAL,
                        confidence=0.85,
                        fiscal_year=curr_yr,
                    ))
                elif beneish.risk_level == "HIGH":
                    findings.append(self.red_flag(
                        title=f"Beneish M-Score: {beneish.m_score:.2f} — Elevated manipulation risk",
                        detail=beneish.interpretation,
                        evidence=f"M-Score={beneish.m_score:.2f}, threshold={FORENSIC_THRESHOLDS.beneish_manipulation}",
                        risk_level=RiskClassification.HIGH,
                        confidence=0.75,
                        fiscal_year=curr_yr,
                    ))
                else:
                    findings.append(self.green_flag(
                        f"Beneish M-Score: {beneish.m_score:.2f} — No manipulation signal",
                        beneish.interpretation,
                        evidence=f"M-Score={beneish.m_score:.2f}",
                    ))
            except Exception as e:
                logger.warning(f"Beneish failed: {e}")

        # ── 2. Piotroski F-Score ───────────────────────────────────
        if len(sorted_years) >= 2:
            curr_yr = sorted_years[-1]
            prev_yr = sorted_years[-2]
            try:
                pio = compute_piotroski_f_score(year_data[curr_yr], year_data[prev_yr])
                details["piotroski"] = {
                    "f_score": pio.f_score,
                    "profitability": pio.profitability_score,
                    "leverage": pio.leverage_score,
                    "efficiency": pio.efficiency_score,
                    "signals": pio.signals,
                    "classification": pio.classification,
                    "interpretation": pio.interpretation,
                }
                object.__setattr__(forensic_scores, "piotroski_f_score", pio.f_score)
                object.__setattr__(forensic_scores, "piotroski_classification", pio.classification)

                if pio.risk_level in ("CRITICAL", "HIGH"):
                    findings.append(self.red_flag(
                        title=f"Piotroski F-Score: {pio.f_score}/9 — {pio.classification}",
                        detail=pio.interpretation,
                        evidence=f"F-Score={pio.f_score} | Profit={pio.profitability_score} | Leverage={pio.leverage_score} | Efficiency={pio.efficiency_score}",
                        risk_level=RiskClassification.CRITICAL if pio.f_score <= 2 else RiskClassification.HIGH,
                        confidence=0.85,
                        fiscal_year=curr_yr,
                    ))
                elif pio.f_score >= FORENSIC_THRESHOLDS.piotroski_strong:
                    findings.append(self.green_flag(
                        f"Piotroski F-Score: {pio.f_score}/9 — Strong financial health",
                        pio.interpretation,
                        evidence=f"F-Score={pio.f_score}/9",
                    ))
            except Exception as e:
                logger.warning(f"Piotroski failed: {e}")

        # ── 3. Altman Z-Score ──────────────────────────────────────
        if sorted_years:
            curr_yr = sorted_years[-1]
            market_cap = market_data.get("market_cap_usd")
            # Convert market cap to reporting currency millions
            if market_cap:
                market_cap = market_cap / 1e6
            try:
                country = profile.get("country", "US")
                use_em = country not in ("US", "GB", "CA")
                altman = compute_altman_z_score(
                    year_data[curr_yr], market_cap=market_cap,
                    use_em_model=use_em,
                    safe_threshold=FORENSIC_THRESHOLDS.altman_safe,
                    distress_threshold=FORENSIC_THRESHOLDS.altman_distress,
                )
                details["altman"] = {
                    "z_score": altman.z_score,
                    "x1": altman.x1, "x2": altman.x2, "x3": altman.x3, "x4": altman.x4,
                    "model": altman.model_used,
                    "classification": altman.classification,
                    "interpretation": altman.interpretation,
                }
                object.__setattr__(forensic_scores, "altman_z_score", altman.z_score)
                object.__setattr__(forensic_scores, "altman_classification", altman.classification)

                if altman.classification == "DISTRESS_ZONE":
                    findings.append(self.red_flag(
                        title=f"Altman Z-Score: {altman.z_score:.2f} — FINANCIAL DISTRESS ZONE",
                        detail=altman.interpretation,
                        evidence=f"Z={altman.z_score:.2f} | X1={altman.x1} | X2={altman.x2} | X3={altman.x3}",
                        risk_level=RiskClassification.CRITICAL,
                        confidence=0.85,
                        fiscal_year=curr_yr,
                    ))
                elif altman.classification == "GREY_ZONE":
                    findings.append(self.make_finding(
                        FindingType.WARNING,
                        f"Altman Z-Score: {altman.z_score:.2f} — Grey zone",
                        altman.interpretation,
                        evidence=f"Z={altman.z_score:.2f}",
                        risk_level=RiskClassification.MEDIUM,
                        confidence=0.75,
                        fiscal_year=curr_yr,
                    ))
                else:
                    findings.append(self.green_flag(
                        f"Altman Z-Score: {altman.z_score:.2f} — Safe zone",
                        altman.interpretation,
                        evidence=f"Z={altman.z_score:.2f} (>{altman.safe_zone})",
                    ))
            except Exception as e:
                logger.warning(f"Altman failed: {e}")

        # ── 4. Sloan Accrual ──────────────────────────────────────
        sloan_findings = self._compute_sloan_accruals(year_data, sorted_years)
        findings.extend(sloan_findings)

        # ── 5. Capital Allocation Review ──────────────────────────
        capex_findings = self._assess_capital_allocation(year_data, sorted_years)
        findings.extend(capex_findings)

        # ── 6. LLM Forensic Deep Dive ─────────────────────────────
        summary_text = self._build_forensic_summary(details, year_data, sorted_years)
        llm_forensic = self.llm_analyze(
            FORENSIC_SYSTEM,
            f"Company: {ticker}\nForensic scores:\n{summary_text}\n\n"
            "Provide a forensic accounting assessment with specific red flags and supporting evidence.",
            max_tokens=1800,
        )
        details["llm_forensic_analysis"] = llm_forensic

        # ── Overall forensic risk ─────────────────────────────────
        critical_count = sum(1 for f in findings if f.risk_level == RiskClassification.CRITICAL)
        high_count = sum(1 for f in findings if f.risk_level == RiskClassification.HIGH)
        overall_risk = min(95.0, 20 + critical_count * 20 + high_count * 10)

        risk_class = (
            RiskClassification.CRITICAL if overall_risk >= 75 else
            RiskClassification.HIGH if overall_risk >= 55 else
            RiskClassification.MEDIUM if overall_risk >= 35 else
            RiskClassification.LOW
        )
        object.__setattr__(forensic_scores, "overall_forensic_risk", risk_class)

        self.storage.save_json(details, "forensic_scores.json", "Agent_Outputs")

        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=(
                f"Forensic assessment: risk={overall_risk:.0f}/100 ({risk_class.value}). "
                f"Beneish M-Score={details.get('beneish', {}).get('m_score', 'N/A')}, "
                f"Piotroski F={details.get('piotroski', {}).get('f_score', 'N/A')}/9, "
                f"Altman Z={details.get('altman', {}).get('z_score', 'N/A')}. "
                f"Red flags: {len([f for f in findings if f.finding_type == FindingType.RED_FLAG])}."
            ),
            findings=findings,
            risk_score=overall_risk,
            risk_classification=risk_class,
            payload={
                "forensic_scores": forensic_scores.model_dump(mode="json"),
                "details": details,
            },
        )

    def _compute_sloan_accruals(self, year_data: dict, years: list) -> list:
        findings = []
        accruals = []
        for yr in years:
            d = year_data.get(yr, {})
            ni = d.get("net_income")
            cfo = d.get("cfo")
            ta = d.get("total_assets")
            if ni is not None and cfo is not None and ta and ta > 0:
                sloan = (ni - cfo) / ta
                accruals.append((yr, sloan))
                self.audit.log_data_point(
                    self.AGENT_ID, "sloan_accrual", sloan,
                    "financial_statements", f"computed_from_{yr}",
                    fiscal_year=yr, confidence=0.9,
                )

        if accruals:
            avg = sum(v for _, v in accruals) / len(accruals)
            if avg > 0.10:
                findings.append(self.red_flag(
                    title=f"High Sloan accrual: avg {avg:.2%} of assets",
                    detail="Persistently high accruals indicate earnings manipulation risk (Sloan 1996).",
                    evidence=f"Sloan accruals by year: {[(y, f'{v:.3f}') for y, v in accruals]}",
                    risk_level=RiskClassification.HIGH,
                    confidence=0.80,
                ))
            object.__setattr__(
                __import__("..models.research", fromlist=["ForensicScores"]).ForensicScores(),
                "sloan_accrual", avg,
            ) if False else None

        return findings

    def _assess_capital_allocation(self, year_data: dict, years: list) -> list:
        findings = []
        fcf_list, div_list = [], []
        for yr in years:
            d = year_data.get(yr, {})
            cfo = d.get("cfo")
            capex = d.get("capex")
            div = d.get("dividends_paid")
            ni = d.get("net_income")
            if cfo and capex:
                fcf = cfo - abs(capex)
                fcf_list.append((yr, fcf))
            if div and ni and ni > 0:
                payout = abs(div) / ni
                div_list.append((yr, payout))

        negative_fcf = [yr for yr, fcf in fcf_list if fcf < 0]
        if len(negative_fcf) >= 2:
            findings.append(self.red_flag(
                title=f"Persistently negative free cash flow: {negative_fcf}",
                detail="Multiple years of negative FCF indicate capex exceeds operating cash generation.",
                evidence=f"FCF negative in years: {negative_fcf}",
                risk_level=RiskClassification.HIGH,
                confidence=0.85,
            ))

        high_payout = [(yr, p) for yr, p in div_list if p > 1.0]
        if high_payout:
            findings.append(self.red_flag(
                title="Dividends exceed net income — unsustainable payout",
                detail="Paying dividends in excess of earnings depletes retained earnings and may signal distress.",
                evidence=f"Payout ratios: {[(yr, f'{p:.1%}') for yr, p in high_payout]}",
                risk_level=RiskClassification.HIGH,
                confidence=0.90,
            ))

        return findings

    def _build_forensic_summary(self, details: dict, year_data: dict, years: list) -> str:
        lines = []
        beneish = details.get("beneish", {})
        if beneish:
            lines.append(f"Beneish M-Score: {beneish.get('m_score')} ({beneish.get('classification')})")
        pio = details.get("piotroski", {})
        if pio:
            lines.append(f"Piotroski F-Score: {pio.get('f_score')}/9 ({pio.get('classification')})")
        alt = details.get("altman", {})
        if alt:
            lines.append(f"Altman Z-Score: {alt.get('z_score')} ({alt.get('classification')})")

        for yr in years[-3:]:
            d = year_data.get(yr, {})
            lines.append(
                f"\n{yr}: NI={d.get('net_income')}, CFO={d.get('cfo')}, "
                f"TotalAssets={d.get('total_assets')}, LTD={d.get('long_term_debt')}, "
                f"Revenue={d.get('revenue')}"
            )
        return "\n".join(lines)
