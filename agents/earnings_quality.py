"""
Agent 14 — Earnings Quality Agent
Dedicated to assessing the quality of reported earnings beyond accounting checks.
Focuses on: earnings persistence, cash conversion, segment attribution,
one-time item frequency, and earnings surprise analysis.
"""

from __future__ import annotations
from typing import Optional
from loguru import logger

from .base_agent import BaseAgent
from ..models.research import AgentOutput, AgentStatus, RiskClassification, FindingType
from ..orchestrator.state import ResearchState


EARNINGS_QUALITY_SYSTEM = """You are a forensic accounting specialist evaluating earnings quality.
Given the financial data, assess:
1. Earnings persistence — are profits driven by core operations or one-time items?
2. Cash conversion ratio — Net Income vs Operating Cash Flow (OCF/NI should be >0.8x)
3. Effective tax rate consistency — unusual drops may mask restructuring or one-timers
4. Segment margin attribution — are high-margin segments growing or declining as % of mix?
5. Operating leverage — does EBIT growth significantly exceed revenue growth?
6. Earnings surprise pattern — consistent beats may indicate sandbagging; consistent misses suggest guidance credibility issues

Score each dimension from 1-5 (5=excellent quality) and provide an overall earnings quality grade: A/B/C/D/F."""


class EarningsQualityAgent(BaseAgent):
    AGENT_ID = "14_earnings_quality"
    AGENT_NAME = "Earnings Quality Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        profile = state.company_profile or {}
        ticker = profile.get("ticker", state.ticker)
        company_name = profile.get("name", state.company_name)
        history_raw = state.financial_history or {}

        findings = []
        details: dict = {
            "cash_conversion": {},
            "tax_rate_analysis": {},
            "earnings_persistence": {},
            "operating_leverage": {},
            "earnings_surprise": {},
            "quality_scores": {},
            "overall_grade": "N/A",
        }

        if not history_raw:
            return AgentOutput(
                agent_id=self.AGENT_ID,
                agent_name=self.AGENT_NAME,
                status=AgentStatus.COMPLETED,
                summary="No financial history available for earnings quality assessment.",
                findings=[self.make_finding(
                    FindingType.WARNING,
                    "Insufficient data for earnings quality assessment",
                    "Financial history is empty — earnings quality cannot be assessed.",
                    "No financial_history in ResearchState.",
                    risk_level=RiskClassification.MEDIUM, confidence=0.95,
                )],
                risk_score=50.0,
                payload=details,
                sources_used=[],
            )

        years = sorted(history_raw.keys()) if isinstance(history_raw, dict) else []

        # ── Extract time-series data ───────────────────────────────
        net_incomes = self._extract_series(history_raw, years, "income_statements", "net_income")
        cfo_series = self._extract_series(history_raw, years, "cash_flows", "operating_cash_flow")
        revenue_series = self._extract_series(history_raw, years, "income_statements", "revenue")
        ebit_series = self._extract_series(history_raw, years, "income_statements", "ebit")
        pretax_income = self._extract_series(history_raw, years, "income_statements", "pretax_income")
        income_tax = self._extract_series(history_raw, years, "income_statements", "income_tax")

        # ── 1. Cash Conversion Ratio ──────────────────────────────
        cash_conversions = {}
        for yr in years:
            ni = net_incomes.get(yr)
            cfo = cfo_series.get(yr)
            if ni and cfo and ni != 0:
                cash_conversions[yr] = round(cfo / ni, 2)

        details["cash_conversion"] = cash_conversions
        if len(cash_conversions) >= 2:
            avg_ccr = sum(cash_conversions.values()) / len(cash_conversions)
            if avg_ccr < 0.6:
                findings.append(self.red_flag(
                    f"Poor cash conversion: {avg_ccr:.2f}x (avg OCF/Net Income)",
                    "Operating cash flow consistently below net income — suggests earnings are not cash-backed. "
                    "May indicate working capital build-up, accrual manipulation, or aggressive revenue recognition.",
                    f"Year-wise CCR: {cash_conversions}",
                    risk_level=RiskClassification.HIGH, confidence=0.85,
                ))
            elif avg_ccr < 0.8:
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    f"Below-par cash conversion: {avg_ccr:.2f}x (avg OCF/Net Income)",
                    "Average cash conversion below 0.8x raises questions about earnings backing.",
                    f"Year-wise CCR: {cash_conversions}",
                    risk_level=RiskClassification.MEDIUM, confidence=0.80,
                ))
            elif avg_ccr > 1.2:
                findings.append(self.green_flag(
                    f"Excellent cash conversion: {avg_ccr:.2f}x",
                    "Operating cash flow exceeds net income, indicating conservative accounting and strong earnings quality.",
                    f"Year-wise CCR: {cash_conversions}",
                    confidence=0.85,
                ))
            details["quality_scores"]["cash_conversion"] = self._score_ccr(avg_ccr)

        # ── 2. Tax Rate Consistency ────────────────────────────────
        effective_tax_rates = {}
        for yr in years:
            pti = pretax_income.get(yr)
            tax = income_tax.get(yr)
            if pti and tax and pti > 0:
                effective_tax_rates[yr] = round(abs(tax) / pti * 100, 1)

        details["tax_rate_analysis"] = effective_tax_rates
        if len(effective_tax_rates) >= 2:
            rates = list(effective_tax_rates.values())
            rate_range = max(rates) - min(rates)
            if rate_range > 15:
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    f"Volatile effective tax rate (range: {rate_range:.1f}pp over {len(rates)} years)",
                    "High tax rate variability may indicate one-time deferred tax benefits, "
                    "restructuring charges, or aggressive tax planning.",
                    f"Tax rates: {effective_tax_rates}",
                    risk_level=RiskClassification.MEDIUM, confidence=0.75,
                ))
            details["quality_scores"]["tax_rate_consistency"] = 5 - min(int(rate_range / 5), 4)

        # ── 3. Earnings Persistence ────────────────────────────────
        ni_volatility = self._compute_volatility(list(net_incomes.values()))
        rev_growth_rates = []
        rev_values = [revenue_series[yr] for yr in sorted(revenue_series.keys())]
        for i in range(1, len(rev_values)):
            if rev_values[i - 1] and rev_values[i - 1] != 0:
                rev_growth_rates.append((rev_values[i] - rev_values[i - 1]) / abs(rev_values[i - 1]) * 100)

        ni_values = [net_incomes[yr] for yr in sorted(net_incomes.keys())]
        profitable_years = sum(1 for v in ni_values if v and v > 0)
        total_years = len(ni_values)

        details["earnings_persistence"] = {
            "profitable_years": profitable_years,
            "total_years": total_years,
            "profit_rate_pct": round(profitable_years / total_years * 100, 0) if total_years else 0,
            "ni_coefficient_of_variation": ni_volatility,
        }

        if total_years > 0 and profitable_years < total_years * 0.7:
            findings.append(self.red_flag(
                f"Inconsistent profitability: profitable in only {profitable_years}/{total_years} years",
                "Company has had loss-making years, indicating earnings are not persistent.",
                f"Net income by year: {dict(zip(sorted(net_incomes.keys()), ni_values))}",
                risk_level=RiskClassification.HIGH, confidence=0.90,
            ))
        elif profitable_years == total_years and total_years >= 4:
            findings.append(self.green_flag(
                f"Consistently profitable for {total_years} consecutive years",
                "Unbroken profitability indicates resilient earnings power.",
                f"Net incomes: {net_incomes}",
                confidence=0.90,
            ))
        details["quality_scores"]["earnings_persistence"] = int(profitable_years / max(total_years, 1) * 5)

        # ── 4. Operating Leverage ─────────────────────────────────
        ebit_values = [ebit_series[yr] for yr in sorted(ebit_series.keys())]
        if len(rev_growth_rates) >= 2 and len(ebit_values) >= 3:
            ebit_growths = []
            ebit_vals_sorted = [ebit_series[yr] for yr in sorted(ebit_series.keys())]
            for i in range(1, len(ebit_vals_sorted)):
                if ebit_vals_sorted[i - 1] and ebit_vals_sorted[i - 1] != 0:
                    ebit_growths.append((ebit_vals_sorted[i] - ebit_vals_sorted[i - 1]) / abs(ebit_vals_sorted[i - 1]) * 100)

            if rev_growth_rates and ebit_growths:
                avg_rev_growth = sum(rev_growth_rates) / len(rev_growth_rates)
                avg_ebit_growth = sum(ebit_growths) / len(ebit_growths)
                op_leverage = avg_ebit_growth / avg_rev_growth if avg_rev_growth != 0 else None
                details["operating_leverage"] = {
                    "avg_revenue_growth_pct": round(avg_rev_growth, 1),
                    "avg_ebit_growth_pct": round(avg_ebit_growth, 1),
                    "degree_of_operating_leverage": round(op_leverage, 2) if op_leverage else None,
                }
                if op_leverage and op_leverage > 2.0:
                    findings.append(self.green_flag(
                        f"Strong operating leverage: {op_leverage:.1f}x (EBIT grows {op_leverage:.1f}x faster than revenue)",
                        "High operating leverage indicates scalable business model with growing profit margins.",
                        f"Avg revenue growth: {avg_rev_growth:.1f}% | Avg EBIT growth: {avg_ebit_growth:.1f}%",
                        confidence=0.80,
                    ))
                elif op_leverage and op_leverage < 0:
                    findings.append(self.make_finding(
                        FindingType.WARNING,
                        f"Negative operating leverage: {op_leverage:.1f}x",
                        "EBIT is contracting while revenue grows — cost base is outpacing revenue growth.",
                        f"Avg revenue growth: {avg_rev_growth:.1f}% | Avg EBIT growth: {avg_ebit_growth:.1f}%",
                        risk_level=RiskClassification.HIGH, confidence=0.80,
                    ))

        # ── 5. Earnings Surprise (via yfinance) ───────────────────
        surprise_data = self._fetch_earnings_surprises(ticker)
        details["earnings_surprise"] = surprise_data
        if surprise_data.get("consistent_misses"):
            findings.append(self.red_flag(
                "Pattern of earnings estimate misses",
                "Company has a history of missing analyst earnings estimates — may indicate guidance credibility issues.",
                f"Miss rate: {surprise_data.get('miss_rate_pct')}%",
                risk_level=RiskClassification.MEDIUM, confidence=0.70,
            ))

        # ── LLM quality assessment ────────────────────────────────
        llm_prompt = self._build_llm_prompt(ticker, company_name, details, net_incomes, cfo_series)
        llm_assessment = self.llm_analyze(EARNINGS_QUALITY_SYSTEM, llm_prompt, max_tokens=1200)
        details["llm_earnings_quality_assessment"] = llm_assessment

        # Extract grade from LLM output
        grade = self._extract_grade(llm_assessment)
        details["overall_grade"] = grade

        self.storage.save_json(details, "earnings_quality.json", "Agent_Outputs")

        # ── Risk scoring ──────────────────────────────────────────
        risk_score = self._compute_risk_score(details, findings)
        risk_cls = self._risk_class(risk_score)

        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=(
                f"Earnings quality assessment for {ticker}: Grade {grade}. "
                f"Cash conversion: {details['quality_scores'].get('cash_conversion', 'N/A')}/5. "
                f"Profitable years: {profitable_years}/{total_years}. "
                f"Findings: {len(findings)}."
            ),
            findings=findings,
            risk_score=risk_score,
            risk_classification=risk_cls,
            payload=details,
            sources_used=["yfinance", "financial_history"],
        )

    # ── Helpers ───────────────────────────────────────────────────

    def _extract_series(self, history: dict, years: list, statement: str, field: str) -> dict:
        result = {}
        for yr in years:
            yr_data = history.get(yr, {})
            if isinstance(yr_data, dict):
                stmt = yr_data.get(statement, {})
                if isinstance(stmt, dict):
                    val = stmt.get(field)
                    if val is not None:
                        try:
                            result[yr] = float(val)
                        except (TypeError, ValueError):
                            pass
        return result

    def _compute_volatility(self, values: list) -> Optional[float]:
        try:
            import statistics
            vals = [v for v in values if v is not None and v != 0]
            if len(vals) < 2:
                return None
            mean = statistics.mean(vals)
            if mean == 0:
                return None
            return round(statistics.stdev(vals) / abs(mean), 3)
        except Exception:
            return None

    def _score_ccr(self, ccr: float) -> int:
        if ccr >= 1.2: return 5
        if ccr >= 1.0: return 4
        if ccr >= 0.8: return 3
        if ccr >= 0.6: return 2
        return 1

    def _fetch_earnings_surprises(self, ticker: str) -> dict:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            cal = t.earnings_dates
            if cal is None or cal.empty:
                return {"available": False}
            misses = 0
            total = 0
            for _, row in cal.iterrows():
                actual = row.get("Reported EPS")
                estimated = row.get("EPS Estimate")
                if actual is not None and estimated is not None:
                    total += 1
                    if actual < estimated * 0.95:
                        misses += 1
            miss_rate = round(misses / total * 100, 0) if total > 0 else 0
            return {
                "available": True,
                "total_quarters": total,
                "misses": misses,
                "miss_rate_pct": miss_rate,
                "consistent_misses": miss_rate > 50 and total >= 4,
            }
        except Exception as e:
            logger.debug(f"Earnings surprise fetch failed: {e}")
            return {"available": False}

    def _build_llm_prompt(self, ticker: str, company: str, details: dict,
                          net_incomes: dict, cfo_series: dict) -> str:
        cc = details.get("cash_conversion", {})
        tax = details.get("tax_rate_analysis", {})
        ep = details.get("earnings_persistence", {})
        ol = details.get("operating_leverage", {})
        return (
            f"Company: {company} ({ticker})\n\n"
            f"Cash Conversion (OCF/NI by year): {cc}\n"
            f"Effective Tax Rates (%): {tax}\n"
            f"Earnings Persistence: Profitable {ep.get('profitable_years')}/{ep.get('total_years')} years "
            f"| NI Coefficient of Variation: {ep.get('ni_coefficient_of_variation')}\n"
            f"Operating Leverage: Rev CAGR {ol.get('avg_revenue_growth_pct')}% | "
            f"EBIT CAGR {ol.get('avg_ebit_growth_pct')}% | DOL: {ol.get('degree_of_operating_leverage')}\n"
            f"Net Incomes (USD M): {net_incomes}\n"
            f"Operating Cash Flows (USD M): {cfo_series}\n\n"
            "Score each dimension 1-5 and provide an overall earnings quality grade (A/B/C/D/F). "
            "Identify the most significant earnings quality concern."
        )

    def _extract_grade(self, analysis: str) -> str:
        import re
        for pattern in [r'grade[:\s]+([A-F])', r'overall[:\s]+([A-F])\b', r'\b([A-F])\s*grade']:
            match = re.search(pattern, analysis, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        if any(kw in analysis.lower() for kw in ["excellent", "strong", "high quality"]):
            return "A"
        if any(kw in analysis.lower() for kw in ["good", "adequate", "above average"]):
            return "B"
        if any(kw in analysis.lower() for kw in ["average", "moderate", "mixed"]):
            return "C"
        if any(kw in analysis.lower() for kw in ["poor", "weak", "below"]):
            return "D"
        if any(kw in analysis.lower() for kw in ["very poor", "manipulation", "fraud"]):
            return "F"
        return "C"

    def _compute_risk_score(self, details: dict, findings: list) -> float:
        grade = details.get("overall_grade", "C")
        grade_risk = {"A": 15, "B": 25, "C": 40, "D": 60, "F": 80, "N/A": 50}
        base = grade_risk.get(grade, 50)
        high_findings = sum(1 for f in findings
                            if hasattr(f, "risk_level") and
                            f.risk_level in (RiskClassification.HIGH, RiskClassification.CRITICAL))
        return min(base + high_findings * 5, 95.0)

    def _risk_class(self, score: float) -> RiskClassification:
        if score >= 70: return RiskClassification.HIGH
        if score >= 45: return RiskClassification.MEDIUM
        return RiskClassification.LOW
