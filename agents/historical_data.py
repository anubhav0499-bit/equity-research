"""
Agent 13 — Historical Data Agent
Retrieves multi-year historical price series, volume, dividends, splits, and return metrics.
Computes volatility, beta vs benchmark, and price-based risk signals.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from .base_agent import BaseAgent
from ..models.research import AgentOutput, AgentStatus, RiskClassification, FindingType
from ..orchestrator.state import ResearchState


HISTORICAL_SYSTEM = """You are a quantitative equity analyst reviewing historical price and volume data.
Analyze the provided historical market data and:
1. Assess price momentum vs relevant benchmarks (3M, 6M, 1Y, 3Y)
2. Identify unusual volume spikes that may indicate informed trading or news events
3. Flag any large drawdowns (>20% peak-to-trough) and their recovery characteristics
4. Comment on dividend history — consistency, growth, and any cuts
5. Assess whether the stock's beta is consistent with its business model

Be quantitative and specific. Note the most significant price events in the history."""


class HistoricalDataAgent(BaseAgent):
    AGENT_ID = "13_historical_data"
    AGENT_NAME = "Historical Data Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        profile = state.company_profile or {}
        ticker = profile.get("ticker", state.ticker)
        company_name = profile.get("name", state.company_name)

        findings = []
        details: dict = {
            "price_history": {},
            "return_metrics": {},
            "dividend_history": [],
            "split_history": [],
            "drawdown_analysis": {},
            "benchmark_comparison": {},
            "volatility_metrics": {},
        }

        try:
            import yfinance as yf
            import numpy as np

            t = yf.Ticker(ticker)

            # ── 5-year daily price history ────────────────────────
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 5 + 30)
            hist = t.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=True,
            )

            if hist.empty:
                return self._no_data_output(ticker, findings)

            # Store price snapshots (not the full series to keep state manageable)
            prices = hist["Close"]
            volumes = hist["Volume"]

            current_price = float(prices.iloc[-1]) if len(prices) else None
            price_1y_ago = float(prices.iloc[-252]) if len(prices) >= 252 else None
            price_3y_ago = float(prices.iloc[-756]) if len(prices) >= 756 else None
            price_max = float(prices.max())
            price_min = float(prices.min())

            details["price_history"] = {
                "current_price": current_price,
                "52w_high": float(prices.iloc[-252:].max()) if len(prices) >= 252 else price_max,
                "52w_low": float(prices.iloc[-252:].min()) if len(prices) >= 252 else price_min,
                "5y_high": price_max,
                "5y_low": price_min,
                "price_1y_ago": price_1y_ago,
                "price_3y_ago": price_3y_ago,
                "avg_daily_volume_90d": float(volumes.iloc[-90:].mean()) if len(volumes) >= 90 else float(volumes.mean()),
            }

            # ── Return metrics ────────────────────────────────────
            returns = prices.pct_change().dropna()
            ret_1m = float((prices.iloc[-1] / prices.iloc[-21] - 1) * 100) if len(prices) >= 21 else None
            ret_3m = float((prices.iloc[-1] / prices.iloc[-63] - 1) * 100) if len(prices) >= 63 else None
            ret_6m = float((prices.iloc[-1] / prices.iloc[-126] - 1) * 100) if len(prices) >= 126 else None
            ret_1y = float((prices.iloc[-1] / prices.iloc[-252] - 1) * 100) if len(prices) >= 252 else None
            ret_3y = float((prices.iloc[-1] / prices.iloc[-756] - 1) * 100) if len(prices) >= 756 else None

            details["return_metrics"] = {
                "return_1m_pct": ret_1m,
                "return_3m_pct": ret_3m,
                "return_6m_pct": ret_6m,
                "return_1y_pct": ret_1y,
                "return_3y_pct": ret_3y,
            }

            # ── Volatility metrics ────────────────────────────────
            ann_vol = float(returns.std() * (252 ** 0.5) * 100) if len(returns) > 21 else None
            vol_1y = float(returns.iloc[-252:].std() * (252 ** 0.5) * 100) if len(returns) >= 252 else ann_vol

            details["volatility_metrics"] = {
                "annualised_volatility_pct": ann_vol,
                "1y_volatility_pct": vol_1y,
                "max_daily_loss_pct": float(returns.min() * 100) if len(returns) else None,
                "max_daily_gain_pct": float(returns.max() * 100) if len(returns) else None,
            }

            # ── Drawdown analysis ─────────────────────────────────
            drawdown = self._compute_drawdown(prices)
            details["drawdown_analysis"] = drawdown
            if drawdown.get("max_drawdown_pct", 0) < -40:
                findings.append(self.red_flag(
                    f"Severe historical drawdown: {drawdown['max_drawdown_pct']:.1f}%",
                    "The stock suffered a peak-to-trough decline exceeding 40%, indicating high historical volatility.",
                    f"Max drawdown: {drawdown['max_drawdown_pct']:.1f}% | Recovery: {drawdown.get('recovery_months', 'N/A')} months",
                    risk_level=RiskClassification.HIGH, confidence=0.90,
                ))
            elif drawdown.get("max_drawdown_pct", 0) < -25:
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    f"Significant historical drawdown: {drawdown['max_drawdown_pct']:.1f}%",
                    "Drawdown exceeds -25%, indicating meaningful historical price risk.",
                    f"Max drawdown: {drawdown['max_drawdown_pct']:.1f}%",
                    risk_level=RiskClassification.MEDIUM, confidence=0.85,
                ))

            # ── Benchmark comparison ───────────────────────────────
            benchmark = self._get_benchmark_ticker(profile)
            bench_return = self._get_benchmark_return(benchmark)
            if bench_return is not None and ret_1y is not None:
                alpha = ret_1y - bench_return
                details["benchmark_comparison"] = {
                    "benchmark": benchmark,
                    "benchmark_1y_return_pct": bench_return,
                    "stock_1y_return_pct": ret_1y,
                    "alpha_1y_pct": alpha,
                }
                if alpha < -15:
                    findings.append(self.red_flag(
                        f"Significant underperformance vs benchmark: {alpha:+.1f}pp",
                        f"Stock underperformed {benchmark} by {abs(alpha):.1f}pp over 1 year.",
                        f"Stock: {ret_1y:.1f}% | Benchmark: {bench_return:.1f}%",
                        risk_level=RiskClassification.MEDIUM, confidence=0.85,
                    ))
                elif alpha > 15:
                    findings.append(self.green_flag(
                        f"Strong outperformance vs benchmark: {alpha:+.1f}pp",
                        f"Stock outperformed {benchmark} by {alpha:.1f}pp over 1 year.",
                        f"Stock: {ret_1y:.1f}% | Benchmark: {bench_return:.1f}%",
                        confidence=0.85,
                    ))

            # ── Dividend history ──────────────────────────────────
            try:
                divs = t.dividends
                if not divs.empty:
                    div_list = []
                    for date, amount in divs.items():
                        div_list.append({"date": str(date.date()), "amount": float(amount)})
                    div_list = sorted(div_list, key=lambda x: x["date"], reverse=True)[:10]
                    details["dividend_history"] = div_list

                    # Check for dividend cut (decline in recent 2 dividends)
                    if len(div_list) >= 4:
                        recent = [d["amount"] for d in div_list[:2]]
                        older = [d["amount"] for d in div_list[2:4]]
                        if sum(recent) < sum(older) * 0.85:
                            findings.append(self.red_flag(
                                "Dividend reduction detected",
                                "Recent dividends are significantly lower than prior periods, suggesting financial stress or policy shift.",
                                f"Recent: {recent} | Prior: {older}",
                                risk_level=RiskClassification.HIGH, confidence=0.90,
                            ))
                        elif len(div_list) >= 8:
                            years_paying = len(set(d["date"][:4] for d in div_list))
                            if years_paying >= 5:
                                findings.append(self.green_flag(
                                    f"Consistent dividend history: {years_paying}+ years",
                                    "Company has maintained dividends for multiple years, indicating financial stability.",
                                    f"Dividend history: {div_list[:3]}",
                                    confidence=0.80,
                                ))
            except Exception as e:
                logger.debug(f"Dividend history fetch failed: {e}")

            # ── Splits ────────────────────────────────────────────
            try:
                splits = t.splits
                if not splits.empty:
                    details["split_history"] = [
                        {"date": str(d.date()), "ratio": float(r)}
                        for d, r in splits.items()
                    ]
            except Exception:
                pass

            # ── LLM analysis ──────────────────────────────────────
            llm_text = self._build_llm_prompt(ticker, company_name, details)
            llm_analysis = self.llm_analyze(HISTORICAL_SYSTEM, llm_text, max_tokens=1200)
            details["llm_historical_analysis"] = llm_analysis

            # ── Volatility risk finding ────────────────────────────
            if ann_vol and ann_vol > 50:
                findings.append(self.red_flag(
                    f"Very high stock volatility: {ann_vol:.1f}% annualised",
                    "Annualised price volatility exceeds 50%, indicating speculative or distressed stock characteristics.",
                    f"Annualised volatility: {ann_vol:.1f}%",
                    risk_level=RiskClassification.HIGH, confidence=0.90,
                ))
            elif ann_vol and ann_vol > 30:
                findings.append(self.make_finding(
                    FindingType.WARNING,
                    f"Elevated stock volatility: {ann_vol:.1f}% annualised",
                    "Volatility exceeds 30% — above typical large-cap range.",
                    f"Annualised volatility: {ann_vol:.1f}%",
                    risk_level=RiskClassification.MEDIUM, confidence=0.85,
                ))

            self.storage.save_json(details, "historical_data.json", "Agent_Outputs")

            risk_score = self._compute_risk(details, findings)
            risk_cls = self._classify_risk_from_score(risk_score)
            return AgentOutput(
                agent_id=self.AGENT_ID,
                agent_name=self.AGENT_NAME,
                status=AgentStatus.COMPLETED,
                summary=(
                    f"Historical data retrieved for {ticker}: "
                    f"1Y return: {ret_1y:.1f}% | Volatility: {ann_vol:.1f}% | "
                    f"Max drawdown: {drawdown.get('max_drawdown_pct', 0):.1f}%"
                    if (ret_1y is not None and ann_vol is not None)
                    else f"Historical data retrieved for {ticker}."
                ),
                findings=findings,
                risk_score=risk_score,
                risk_classification=risk_cls,
                payload=details,
                sources_used=["yfinance_history"],
            )

        except Exception as e:
            logger.error(f"HistoricalDataAgent failed: {e}")
            return self._failure_output(str(e))

    # ── Helpers ───────────────────────────────────────────────────

    def _no_data_output(self, ticker: str, findings: list) -> AgentOutput:
        findings.append(self.make_finding(
            FindingType.WARNING,
            "No historical price data available",
            "Price history could not be retrieved for this ticker.",
            f"Ticker: {ticker}",
            risk_level=RiskClassification.LOW, confidence=0.95,
        ))
        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=f"No historical price data found for {ticker}.",
            findings=findings,
            risk_score=40.0,
            risk_classification=RiskClassification.MEDIUM,
            payload={},
            sources_used=["yfinance_history"],
        )

    def _compute_drawdown(self, prices) -> dict:
        try:
            import numpy as np
            peak = prices.cummax()
            drawdown = (prices - peak) / peak * 100
            max_dd = float(drawdown.min())
            max_dd_date = str(drawdown.idxmin().date()) if hasattr(drawdown.idxmin(), 'date') else str(drawdown.idxmin())

            # Recovery: find first date where price returns to prior peak level after max DD
            max_dd_idx = drawdown.idxmin()
            post_dd = prices[prices.index > max_dd_idx]
            peak_at_dd = float(peak[max_dd_idx])
            recovered = post_dd[post_dd >= peak_at_dd]
            if not recovered.empty:
                dd_date = max_dd_idx
                rec_date = recovered.index[0]
                months = (rec_date - dd_date).days / 30
                recovery_months = round(months, 1)
            else:
                recovery_months = None

            return {
                "max_drawdown_pct": max_dd,
                "max_drawdown_date": max_dd_date,
                "recovery_months": recovery_months,
                "current_drawdown_from_peak_pct": float(drawdown.iloc[-1]),
            }
        except Exception as e:
            logger.debug(f"Drawdown computation failed: {e}")
            return {}

    def _get_benchmark_ticker(self, profile: dict) -> str:
        exchange = profile.get("exchange", "")
        country = profile.get("country", "")
        if exchange in ("NSE", "BSE") or country == "India":
            return "^NSEI"  # Nifty 50
        if country in ("UK", "United Kingdom"):
            return "^FTSE"
        if country in ("Germany",):
            return "^GDAXI"
        if country in ("Japan",):
            return "^N225"
        if country in ("Hong Kong",):
            return "^HSI"
        return "^GSPC"  # S&P 500 default

    def _get_benchmark_return(self, benchmark: str) -> Optional[float]:
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            b = yf.Ticker(benchmark)
            hist = b.history(period="1y", auto_adjust=True)
            if hist.empty or len(hist["Close"]) < 2:
                return None
            prices = hist["Close"]
            return float((prices.iloc[-1] / prices.iloc[0] - 1) * 100)
        except Exception as e:
            logger.debug(f"Benchmark return fetch failed: {e}")
            return None

    def _build_llm_prompt(self, ticker: str, company: str, details: dict) -> str:
        pm = details.get("price_history", {})
        rm = details.get("return_metrics", {})
        vm = details.get("volatility_metrics", {})
        dd = details.get("drawdown_analysis", {})
        bc = details.get("benchmark_comparison", {})
        divs = details.get("dividend_history", [])

        return (
            f"Company: {company} ({ticker})\n\n"
            f"Price History:\n"
            f"  Current: {pm.get('current_price')}\n"
            f"  52W High: {pm.get('52w_high')} | 52W Low: {pm.get('52w_low')}\n"
            f"  Avg Daily Volume (90D): {pm.get('avg_daily_volume_90d', 'N/A')}\n\n"
            f"Returns:\n"
            f"  1M: {rm.get('return_1m_pct')}% | 3M: {rm.get('return_3m_pct')}% | "
            f"6M: {rm.get('return_6m_pct')}% | 1Y: {rm.get('return_1y_pct')}% | "
            f"3Y: {rm.get('return_3y_pct')}%\n\n"
            f"Volatility:\n"
            f"  Annualised: {vm.get('annualised_volatility_pct')}% | "
            f"1Y: {vm.get('1y_volatility_pct')}%\n\n"
            f"Drawdown:\n"
            f"  Max Drawdown: {dd.get('max_drawdown_pct')}% on {dd.get('max_drawdown_date')}\n"
            f"  Recovery: {dd.get('recovery_months')} months\n"
            f"  Current from Peak: {dd.get('current_drawdown_from_peak_pct')}%\n\n"
            f"Benchmark ({bc.get('benchmark', 'N/A')}):\n"
            f"  1Y Return: {bc.get('benchmark_1y_return_pct')}% | Alpha: {bc.get('alpha_1y_pct')}pp\n\n"
            f"Dividend History (last 3): {divs[:3] if divs else 'None'}\n\n"
            "Provide quantitative analysis of price performance, risk characteristics, and any anomalies."
        )

    def _compute_risk(self, details: dict, findings: list) -> float:
        score = 25.0
        vm = details.get("volatility_metrics", {})
        dd = details.get("drawdown_analysis", {})
        vol = vm.get("annualised_volatility_pct", 0) or 0
        max_dd = dd.get("max_drawdown_pct", 0) or 0
        if vol > 50: score += 20
        elif vol > 30: score += 10
        if max_dd < -40: score += 15
        elif max_dd < -25: score += 8
        score += len([f for f in findings if hasattr(f, 'risk_level') and
                      f.risk_level in (RiskClassification.HIGH, RiskClassification.CRITICAL)]) * 5
        return min(score, 90.0)

    def _classify_risk_from_score(self, score: float) -> RiskClassification:
        if score >= 70: return RiskClassification.HIGH
        if score >= 45: return RiskClassification.MEDIUM
        return RiskClassification.LOW
