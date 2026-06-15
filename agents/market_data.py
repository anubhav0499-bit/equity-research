"""
Agent 04 — Market Data Agent
Retrieves current price, historical price series, market cap, and peer market data.
"""

from __future__ import annotations
from typing import Optional
from loguru import logger

from .base_agent import BaseAgent
from ..models.research import AgentOutput, AgentStatus, RiskClassification
from ..orchestrator.state import ResearchState


class MarketDataAgent(BaseAgent):
    AGENT_ID = "04_market_data"
    AGENT_NAME = "Market Data Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        profile = state.company_profile or {}
        ticker = profile.get("ticker", state.ticker)
        exchange = profile.get("exchange", "")
        currency = profile.get("currency", "USD")

        market_data: dict = {}
        findings = []

        # Fetch from yfinance
        yf_data = self._fetch_yfinance(ticker, exchange)
        if yf_data:
            market_data.update(yf_data)

        current_price = market_data.get("current_price")
        if not current_price:
            findings.append(self.red_flag(
                title="Current market price unavailable",
                detail="Could not retrieve current trading price. Valuation upside/downside cannot be computed.",
                evidence=f"Ticker: {ticker}, Exchange: {exchange}",
                risk_level=RiskClassification.MEDIUM,
                confidence=0.9,
            ))

        # Fetch peer data
        sector = profile.get("sector", "")
        peer_data = self._fetch_peer_data(sector, exchange)
        market_data["peer_market_data"] = peer_data

        # 52-week assessment
        if market_data.get("week_52_high") and current_price:
            pct_from_high = (current_price / market_data["week_52_high"] - 1) * 100
            market_data["pct_from_52w_high"] = round(pct_from_high, 1)

        self.storage.save_json(market_data, "market_data.json", "Financial_Statements")

        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=(
                f"Market data for {ticker}: Price={current_price} {currency}, "
                f"Market Cap={market_data.get('market_cap_usd', 'N/A')}, "
                f"52W High={market_data.get('week_52_high', 'N/A')}, "
                f"Beta={market_data.get('beta', 'N/A')}."
            ),
            findings=findings,
            risk_score=10.0,
            risk_classification=RiskClassification.LOW,
            payload={"market_data": market_data},
            sources_used=["yfinance"],
        )

    def _fetch_yfinance(self, ticker: str, exchange: str) -> dict:
        try:
            import yfinance as yf
            import pandas as pd
            candidates = [ticker]
            if exchange == "NSE": candidates += [ticker + ".NS", ticker + ".BO"]
            elif exchange == "BSE": candidates += [ticker + ".BO", ticker + ".NS"]

            for t in candidates:
                try:
                    stock = yf.Ticker(t)
                    info = stock.info
                    if not info.get("currentPrice") and not info.get("regularMarketPrice"):
                        continue
                    price = info.get("currentPrice") or info.get("regularMarketPrice")
                    hist = stock.history(period="1y")
                    price_series: dict = {}
                    if hist is not None and not hist.empty:
                        price_series = {
                            str(d.date()): round(float(v), 4)
                            for d, v in hist["Close"].items()
                        }
                    return {
                        "ticker_used": t,
                        "current_price": price,
                        "currency": info.get("currency", "USD"),
                        "market_cap_usd": info.get("marketCap"),
                        "enterprise_value": info.get("enterpriseValue"),
                        "shares_outstanding": info.get("sharesOutstanding"),
                        "float_shares": info.get("floatShares"),
                        "week_52_high": info.get("fiftyTwoWeekHigh"),
                        "week_52_low":  info.get("fiftyTwoWeekLow"),
                        "beta": info.get("beta"),
                        "pe_ttm": info.get("trailingPE"),
                        "pe_forward": info.get("forwardPE"),
                        "pb": info.get("priceToBook"),
                        "ps_ttm": info.get("priceToSalesTrailing12Months"),
                        "ev_ebitda": info.get("enterpriseToEbitda"),
                        "ev_revenue": info.get("enterpriseToRevenue"),
                        "dividend_yield": info.get("dividendYield"),
                        "payout_ratio": info.get("payoutRatio"),
                        "roe": info.get("returnOnEquity"),
                        "roa": info.get("returnOnAssets"),
                        "debt_to_equity_mkt": info.get("debtToEquity"),
                        "revenue_ttm": info.get("totalRevenue"),
                        "ebitda_ttm": info.get("ebitda"),
                        "net_income_ttm": info.get("netIncomeToCommon"),
                        "fcf_ttm": info.get("freeCashflow"),
                        "analyst_target_price": info.get("targetMeanPrice"),
                        "analyst_recommendation": info.get("recommendationMean"),
                        "analyst_count": info.get("numberOfAnalystOpinions"),
                        "short_ratio": info.get("shortRatio"),
                        "short_pct_float": info.get("shortPercentOfFloat"),
                        "price_series_1y": price_series,
                        "avg_volume": info.get("averageVolume"),
                        "institutional_pct": info.get("institutionPercentHeld"),
                        "insider_pct": info.get("insiderPercentHeld"),
                    }
                except Exception as e:
                    logger.debug(f"yfinance {t}: {e}")
                    continue
        except ImportError:
            pass
        return {}

    def _fetch_peer_data(self, sector: str, exchange: str) -> list[dict]:
        peer_map = {
            "Information Technology": ["MSFT", "AAPL", "GOOGL", "META", "NVDA"],
            "Financials": ["JPM", "BAC", "WFC", "GS", "MS"],
            "Health Care": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
            "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
            "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST"],
            "Energy": ["XOM", "CVX", "SLB", "COP", "EOG"],
            "Industrials": ["HON", "UPS", "CAT", "LMT", "GE"],
            "Materials": ["LIN", "APD", "ECL", "SHW", "NEM"],
        }
        if exchange in ("NSE", "BSE"):
            peer_map.update({
                "Information Technology": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS"],
                "Financials": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS"],
            })
        peers = peer_map.get(sector, [])[:5]
        peer_data = []
        try:
            import yfinance as yf
            for p in peers:
                try:
                    info = yf.Ticker(p).info
                    peer_data.append({
                        "ticker": p,
                        "name": info.get("shortName", p),
                        "pe": info.get("trailingPE"),
                        "pb": info.get("priceToBook"),
                        "ps": info.get("priceToSalesTrailing12Months"),
                        "ev_ebitda": info.get("enterpriseToEbitda"),
                        "market_cap": info.get("marketCap"),
                        "revenue_growth": info.get("revenueGrowth"),
                        "profit_margins": info.get("profitMargins"),
                        "roe": info.get("returnOnEquity"),
                    })
                except Exception:
                    pass
        except ImportError:
            pass
        return peer_data
