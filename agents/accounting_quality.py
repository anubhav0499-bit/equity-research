"""
Agent 05 — Accounting Quality Agent
Assesses quality of earnings, revenue recognition, accruals, and accounting choices.
"""

from __future__ import annotations
import math
from typing import Optional
from loguru import logger

from .base_agent import BaseAgent
from ..models.research import AgentOutput, AgentStatus, RiskClassification, FindingType
from ..orchestrator.state import ResearchState


SYSTEM_PROMPT = """You are a forensic accounting specialist at an institutional asset manager.
Analyze the provided financial data for accounting quality red flags.
Focus on: revenue recognition timing, accrual patterns, channel stuffing, big bath charges,
cookie jar reserves, aggressive capitalization, and off-balance-sheet obligations.
Be specific, quantitative, and cite fiscal years."""


class AccountingQualityAgent(BaseAgent):
    AGENT_ID = "05_accounting_quality"
    AGENT_NAME = "Accounting Quality Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        history_raw = state.financial_history or {}
        profile = state.company_profile or {}
        ticker = profile.get("ticker", state.ticker)
        sector = profile.get("sector", "")

        findings = []
        quality_scores: list[float] = []
        details: dict = {}

        if not history_raw or not isinstance(history_raw, dict):
            return self._no_data_output()

        # ── 1. Revenue recognition quality ────────────────────────
        rev_score, rev_findings = self._analyze_revenue_recognition(history_raw)
        quality_scores.append(rev_score)
        findings.extend(rev_findings)
        details["revenue_quality_score"] = rev_score

        # ── 2. Accrual analysis ───────────────────────────────────
        acc_score, acc_findings = self._analyze_accruals(history_raw)
        quality_scores.append(acc_score)
        findings.extend(acc_findings)
        details["accrual_score"] = acc_score

        # ── 3. Cash conversion quality ────────────────────────────
        cc_score, cc_findings = self._analyze_cash_conversion(history_raw)
        quality_scores.append(cc_score)
        findings.extend(cc_findings)
        details["cash_conversion_score"] = cc_score

        # ── 4. LLM deep dive ──────────────────────────────────────
        financial_summary = self._build_financial_summary(history_raw)
        llm_analysis = self.llm_analyze(
            SYSTEM_PROMPT,
            f"Company: {ticker}\nSector: {sector}\n\n{financial_summary}\n\n"
            "Identify top 5 accounting quality concerns with specific evidence.",
            max_tokens=1500,
        )
        details["llm_accounting_analysis"] = llm_analysis

        overall_risk = self.risk_from_scores(quality_scores)
        self.storage.save_json(details, "accounting_quality.json", "Agent_Outputs")

        classification = (
            RiskClassification.CRITICAL if overall_risk >= 75 else
            RiskClassification.HIGH if overall_risk >= 55 else
            RiskClassification.MEDIUM if overall_risk >= 35 else
            RiskClassification.LOW
        )

        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=(
                f"Accounting quality risk: {overall_risk:.0f}/100 ({classification.value}). "
                f"Revenue quality: {rev_score:.0f}, Accrual score: {acc_score:.0f}, "
                f"Cash conversion: {cc_score:.0f}. "
                f"Key concerns: {len([f for f in findings if f.finding_type == FindingType.RED_FLAG])} red flags."
            ),
            findings=findings,
            risk_score=overall_risk,
            risk_classification=classification,
            payload=details,
        )

    def _analyze_revenue_recognition(self, history: dict) -> tuple[float, list]:
        findings = []
        years = sorted(history.keys(), reverse=True)
        revenues = {}
        receivables = {}

        for yr in years:
            data = history[yr]
            if isinstance(data, dict):
                rev = (data.get("income_statements") or {}).get("revenue") or data.get("revenue")
                ar = (data.get("balance_sheets") or {}).get("accounts_receivable") or data.get("accounts_receivable")
                if rev: revenues[yr] = rev
                if ar: receivables[yr] = ar

        score = 30.0

        # Check DSO trend (rising DSO = potential channel stuffing)
        yrs = sorted(revenues.keys())
        dso_trend = []
        for i in range(1, len(yrs)):
            yr = yrs[i]
            prev = yrs[i-1]
            if revenues.get(yr) and revenues.get(prev) and receivables.get(yr) and receivables.get(prev):
                dso_curr = receivables[yr] / (revenues[yr] / 365)
                dso_prev = receivables[prev] / (revenues[prev] / 365)
                dso_trend.append(dso_curr - dso_prev)

        if dso_trend:
            avg_dso_change = sum(dso_trend) / len(dso_trend)
            if avg_dso_change > 15:
                score += 30
                findings.append(self.red_flag(
                    title=f"Rising DSO trend: avg +{avg_dso_change:.0f} days/year",
                    detail="Accounts receivable growing faster than revenue — potential channel stuffing or collection issues.",
                    evidence=f"DSO changes by year: {[f'{x:.1f}' for x in dso_trend]}",
                    risk_level=RiskClassification.HIGH,
                    confidence=0.80,
                ))
            elif avg_dso_change > 7:
                score += 15
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    "Moderate DSO increase",
                    f"DSO rising by avg {avg_dso_change:.0f} days/year — monitor closely.",
                    evidence=f"DSO delta: {[f'{x:.1f}' for x in dso_trend]}",
                    risk_level=RiskClassification.MEDIUM, confidence=0.70,
                ))

        # Revenue growth anomaly check
        rev_growths = []
        for i in range(1, len(yrs)):
            yr, prev = yrs[i], yrs[i-1]
            if revenues.get(yr) and revenues.get(prev) and revenues[prev] > 0:
                g = (revenues[yr] / revenues[prev] - 1) * 100
                rev_growths.append((yr, g))
        for yr, g in rev_growths:
            if g > 50:
                score += 20
                findings.append(self.red_flag(
                    title=f"Anomalous revenue surge in {yr}: +{g:.0f}%",
                    detail="Revenue growth exceeding 50% YoY warrants scrutiny for recognition timing or one-off items.",
                    evidence=f"{yr} revenue growth: {g:.1f}%",
                    risk_level=RiskClassification.HIGH, confidence=0.75,
                ))
            elif g < -20:
                score += 10
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    f"Significant revenue decline in {yr}: {g:.0f}%",
                    "Sharp revenue decline may indicate business deterioration or prior-year overstatement.",
                    evidence=f"{yr} revenue growth: {g:.1f}%",
                    risk_level=RiskClassification.MEDIUM, confidence=0.70,
                ))

        return min(score, 90.0), findings

    def _analyze_accruals(self, history: dict) -> tuple[float, list]:
        findings = []
        score = 25.0
        years = sorted(history.keys(), reverse=True)

        accrual_ratios = []
        for yr in years:
            data = history[yr]
            if isinstance(data, dict):
                is_data = data.get("income_statements") or {}
                bs_data = data.get("balance_sheets") or {}
                cf_data = data.get("cash_flows") or {}
                net_income = is_data.get("net_income") or data.get("net_income")
                cfo = cf_data.get("cfo") or data.get("cfo")
                total_assets = bs_data.get("total_assets") or data.get("total_assets")
                if net_income and cfo and total_assets and total_assets > 0:
                    sloan = (net_income - cfo) / total_assets
                    accrual_ratios.append((yr, sloan))

        if accrual_ratios:
            avg_accrual = sum(r for _, r in accrual_ratios) / len(accrual_ratios)
            if avg_accrual > 0.10:
                score += 35
                findings.append(self.red_flag(
                    title=f"High accrual ratio: {avg_accrual:.2%} of assets",
                    detail="Net income significantly exceeds cash flow — potential earnings quality concern (Sloan accrual).",
                    evidence=f"Accrual ratios by year: {[(y, f'{r:.3f}') for y, r in accrual_ratios]}",
                    risk_level=RiskClassification.HIGH, confidence=0.85,
                ))
            elif avg_accrual > 0.05:
                score += 15
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    f"Elevated accrual ratio: {avg_accrual:.2%}",
                    "Moderate earnings-to-cash-flow divergence.",
                    evidence=f"Avg Sloan accrual: {avg_accrual:.3f}",
                    risk_level=RiskClassification.MEDIUM, confidence=0.75,
                ))
            elif avg_accrual < 0:
                findings.append(self.green_flag(
                    "Negative accrual ratio — cash earnings exceed reported earnings",
                    "Conservative accounting: cash flow exceeds net income consistently.",
                    evidence=f"Avg accrual ratio: {avg_accrual:.3f}",
                    confidence=0.80,
                ))

        return min(score, 90.0), findings

    def _analyze_cash_conversion(self, history: dict) -> tuple[float, list]:
        findings = []
        score = 20.0
        years = sorted(history.keys(), reverse=True)
        cc_ratios = []

        for yr in years:
            data = history[yr]
            if isinstance(data, dict):
                is_data = data.get("income_statements") or {}
                cf_data = data.get("cash_flows") or {}
                net_income = is_data.get("net_income") or data.get("net_income")
                cfo = cf_data.get("cfo") or data.get("cfo")
                if net_income and cfo and net_income > 0:
                    cc_ratios.append((yr, cfo / net_income))

        if cc_ratios:
            avg_cc = sum(r for _, r in cc_ratios) / len(cc_ratios)
            if avg_cc < 0.70:
                score += 40
                findings.append(self.red_flag(
                    title=f"Low cash conversion ratio: {avg_cc:.1%}",
                    detail="Significant portion of reported earnings not converting to cash — potential earnings quality concern.",
                    evidence=f"Cash conversion by year: {[(y, f'{r:.2f}') for y, r in cc_ratios]}",
                    risk_level=RiskClassification.HIGH, confidence=0.85,
                ))
            elif avg_cc < 0.85:
                score += 15
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    f"Below-average cash conversion: {avg_cc:.1%}",
                    "Cash earnings conversion below institutional threshold of 85%.",
                    evidence=f"Avg cash conversion: {avg_cc:.2f}",
                    risk_level=RiskClassification.MEDIUM, confidence=0.75,
                ))
            elif avg_cc >= 1.0:
                findings.append(self.green_flag(
                    "Excellent cash conversion: CFO consistently exceeds net income",
                    "High-quality earnings with strong cash backing.",
                    evidence=f"Avg cash conversion ratio: {avg_cc:.2f}",
                    confidence=0.85,
                ))

        return min(score, 90.0), findings

    def _build_financial_summary(self, history: dict) -> str:
        lines = ["Financial Summary (latest 3 years):"]
        years = sorted(history.keys(), reverse=True)[:3]
        for yr in years:
            data = history[yr]
            if isinstance(data, dict):
                is_d = data.get("income_statements") or {}
                bs_d = data.get("balance_sheets") or {}
                cf_d = data.get("cash_flows") or {}
                lines.append(
                    f"\n{yr}: Revenue={is_d.get('revenue') or data.get('revenue')}, "
                    f"EBITDA={is_d.get('ebitda') or data.get('ebitda')}, "
                    f"Net Income={is_d.get('net_income') or data.get('net_income')}, "
                    f"CFO={cf_d.get('cfo') or data.get('cfo')}, "
                    f"Total Assets={bs_d.get('total_assets') or data.get('total_assets')}"
                )
        return "\n".join(lines)

    def _no_data_output(self) -> AgentOutput:
        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary="No financial history available for accounting quality analysis.",
            risk_score=50.0,
            risk_classification=RiskClassification.MEDIUM,
            findings=[self.red_flag(
                "No financial data available",
                "Cannot perform accounting quality assessment without financial statements.",
                "Financial history is empty or not yet extracted.",
                risk_level=RiskClassification.HIGH, confidence=0.99,
            )],
        )
