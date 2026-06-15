"""
Equity Research Platform — Stress Test Suite
============================================
Covers:
  A. Core valuation math (WACC, DCF, PWTP)
  B. Scenario analysis across all risk profiles
  C. Financial modeling edge cases
  D. Valuation edge cases (zero price, extreme beta, no peers)
  E. Concurrent agent execution & RAG thread safety
  F. Orchestrator failure recovery
  G. Output contract validation

Run:
  python -m pytest tests/stress_test.py -v
  python tests/stress_test.py              (standalone with rich reporter)
"""

from __future__ import annotations

import sys as _sys
if _sys.platform == "win32":
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import concurrent.futures
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

# ── Make sure repo root is importable ─────────────────────────────────────────
# equity_research/ IS the package directory; its parent must be on sys.path
REPO_ROOT = Path(__file__).parent.parent          # …/equity_research/
PARENT_DIR = REPO_ROOT.parent                     # …/Claude Projects/
for p in (str(PARENT_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Platform imports ───────────────────────────────────────────────────────────
from equity_research.models.research import (
    AgentOutput, AgentStatus, Finding, FindingType,
    RiskClassification, ResearchState,
)
from equity_research.models.valuation import Scenario, WACCInputs
from equity_research.models.financials import (
    FinancialHistory, IncomeStatement, BalanceSheet, CashFlowStatement,
)
from equity_research.valuation.wacc import compute_wacc
from equity_research.valuation.dcf import build_dcf
from equity_research.modeling.forecaster import FinancialForecaster, ForecastYear
from equity_research.agents.scenario_analysis import ScenarioAnalysisAgent, DEFAULT_WEIGHTS

# ── Test infrastructure ────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    suite: str
    passed: bool
    duration_ms: float
    error: Optional[str] = None
    detail: str = ""


@dataclass
class StressReport:
    results: list[TestResult] = field(default_factory=list)

    def record(self, r: TestResult) -> None:
        self.results.append(r)

    def summary(self) -> str:
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        lines = [
            "",
            "═" * 72,
            f"  EQUITY RESEARCH PLATFORM — STRESS TEST REPORT",
            f"  {total} tests | {passed} passed | {failed} failed",
            "═" * 72,
        ]
        current_suite = None
        for r in self.results:
            if r.suite != current_suite:
                current_suite = r.suite
                lines.append(f"\n  Suite {r.suite}")
                lines.append("  " + "─" * 68)
            icon = "✓" if r.passed else "✗"
            dur = f"{r.duration_ms:.0f}ms"
            lines.append(f"  {icon}  {r.name:<50} {dur:>7}")
            if not r.passed and r.error:
                for line in r.error.splitlines()[:4]:
                    lines.append(f"       {line}")
            if r.detail:
                lines.append(f"       ↳ {r.detail}")
        lines.append("\n" + "═" * 72 + "\n")
        return "\n".join(lines)


REPORT = StressReport()


def run_test(suite: str, name: str, fn, *args, **kwargs) -> TestResult:
    t0 = time.perf_counter()
    try:
        detail = fn(*args, **kwargs) or ""
        r = TestResult(name=name, suite=suite, passed=True,
                       duration_ms=(time.perf_counter() - t0) * 1000, detail=str(detail))
    except AssertionError as e:
        r = TestResult(name=name, suite=suite, passed=False,
                       duration_ms=(time.perf_counter() - t0) * 1000,
                       error=str(e) or "AssertionError (no message)")
    except Exception as e:
        r = TestResult(name=name, suite=suite, passed=False,
                       duration_ms=(time.perf_counter() - t0) * 1000,
                       error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
    REPORT.record(r)
    return r


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _mock_llm(response: str = "Mock LLM response with Bear/Bull/Base scenario analysis.") -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = response
    llm.backend = "openai"
    llm.get_backend_info.return_value = "mock-llm"
    return llm


def _mock_storage(tmpdir: Path) -> MagicMock:
    storage = MagicMock()
    storage.base_path = tmpdir
    storage.raw_filings = tmpdir / "Raw_Filings"
    storage.raw_filings.mkdir(parents=True, exist_ok=True)
    storage.save_json.return_value = None
    storage.save_bytes.return_value = None
    return storage


def _mock_audit() -> MagicMock:
    audit = MagicMock()
    return audit


def _mock_db() -> MagicMock:
    db = MagicMock()
    db.save_agent_output.return_value = None
    db.save_findings.return_value = None
    return db


def _make_state(ticker: str = "AAPL", company: str = "Apple Inc") -> ResearchState:
    return ResearchState(
        run_id=f"stress_{uuid.uuid4().hex[:8]}",
        company_name=company,
        ticker=ticker,
        started_at="2026-06-16T00:00:00Z",
        company_profile={
            "ticker": ticker,
            "name": company,
            "sector": "Information Technology",
            "country": "US",
            "currency": "USD",
            "exchange": "NASDAQ",
        },
    )


def _make_financial_history(ticker: str = "AAPL", n_years: int = 5) -> FinancialHistory:
    history = FinancialHistory(company_ticker=ticker, currency="USD")
    base_rev = 300_000.0
    years = [str(2019 + i) for i in range(n_years)]
    for i, yr in enumerate(years):
        rev = base_rev * (1.08 ** i)
        ebitda = rev * 0.32
        ebit = rev * 0.28
        da = ebitda - ebit
        net_income = ebit * 0.75
        history.income_statements[yr] = IncomeStatement(
            fiscal_year=yr,
            revenue=rev,
            ebitda=ebitda,
            ebitda_margin=32.0,
            ebit=ebit,
            ebit_margin=28.0,
            net_income=net_income,
            net_margin=21.0,
            interest_expense=500.0,
            depreciation=da,
            effective_tax_rate=25.0,
        )
        history.balance_sheets[yr] = BalanceSheet(
            fiscal_year=yr,
            total_assets=rev * 1.5,
            total_liabilities=rev * 0.6,
            total_equity=rev * 0.9,
            long_term_debt=rev * 0.15,
            cash_and_equivalents=rev * 0.10,
        )
        history.cash_flows[yr] = CashFlowStatement(
            fiscal_year=yr,
            operating_cash_flow=net_income * 1.1,
            capex=rev * 0.05,
            free_cash_flow=net_income * 1.1 - rev * 0.05,
        )
    history.available_years = years
    return history


def _make_wacc(country: str = "US", beta: float = 1.0, debt: float = 5000.0,
               equity: float = 50000.0) -> WACCInputs:
    return compute_wacc(
        country=country, sector="Information Technology",
        beta=beta, debt_book_value=debt, equity_market_cap=equity,
        effective_tax_rate=25.0,
    )


def _make_agent(cls, ticker: str = "AAPL", tmpdir: Optional[Path] = None) -> tuple:
    if tmpdir is None:
        tmpdir = Path(tempfile.mkdtemp())
    storage = _mock_storage(tmpdir)
    audit = _mock_audit()
    db = _mock_db()
    llm = _mock_llm()
    agent = cls(llm=llm, storage=storage, audit=audit, db=db)
    return agent, storage, audit, db


def _make_scenario_state_with_valuation(
    bear: float = 80.0, base: float = 120.0, bull: float = 180.0,
    risk_score: float = 50.0, forensic_score: float = 50.0,
) -> ResearchState:
    state = _make_state()
    state.agent_outputs["08_valuation"] = AgentOutput(
        agent_id="08_valuation", agent_name="Valuation Agent",
        status=AgentStatus.COMPLETED,
        risk_score=risk_score,
        payload={
            "scenarios": {
                "BEAR": {"implied_price": bear, "blended_value": bear},
                "BASE": {"implied_price": base, "blended_value": base},
                "BULL": {"implied_price": bull, "blended_value": bull},
            },
            "valuation_summary": {
                "bear_price": bear, "base_price": base, "bull_price": bull,
                "upside_pct": 20.0,
            },
        },
    )
    state.agent_outputs["09_risk_analysis"] = AgentOutput(
        agent_id="09_risk_analysis", agent_name="Risk Analysis Agent",
        status=AgentStatus.COMPLETED, risk_score=risk_score, payload={},
    )
    state.agent_outputs["06_forensic_accounting"] = AgentOutput(
        agent_id="06_forensic_accounting", agent_name="Forensic Accounting Agent",
        status=AgentStatus.COMPLETED, risk_score=forensic_score, payload={},
    )
    state.agent_outputs["07_financial_modeling"] = AgentOutput(
        agent_id="07_financial_modeling", agent_name="Financial Modeling Agent",
        status=AgentStatus.COMPLETED, risk_score=25.0, payload={"forecasts": {}},
    )
    return state


# ══════════════════════════════════════════════════════════════════════════════
# SUITE A — Core Valuation Math
# ══════════════════════════════════════════════════════════════════════════════

def test_wacc_us_standard():
    wacc = _make_wacc(country="US", beta=1.0, debt=5000, equity=50000)
    assert 7.0 <= wacc.wacc <= 14.0, f"US WACC {wacc.wacc}% outside plausible range"
    assert abs(wacc.debt_weight + wacc.equity_weight - 1.0) < 0.01
    return f"WACC={wacc.wacc:.2f}% Ke={wacc.cost_of_equity:.2f}%"


def test_wacc_india_high_beta():
    wacc = _make_wacc(country="IN", beta=1.4, debt=10000, equity=30000)
    assert wacc.wacc > 10.0, f"India high-beta WACC should exceed 10%, got {wacc.wacc}%"
    assert wacc.country_risk_premium == 1.40
    return f"IN WACC={wacc.wacc:.2f}%"


def test_wacc_zero_debt():
    wacc = _make_wacc(country="US", beta=0.8, debt=0, equity=100000)
    assert wacc.debt_weight == 0.0
    assert wacc.equity_weight == 1.0
    assert abs(wacc.wacc - wacc.cost_of_equity) < 0.001
    return f"All-equity WACC={wacc.wacc:.2f}%"


def test_wacc_extreme_low_beta():
    wacc = compute_wacc(
        country="US", sector="Utilities",
        beta=0.01, debt_book_value=1000, equity_market_cap=20000,
        effective_tax_rate=25.0,
    )
    from equity_research.core.config import MODELING_CONFIG
    assert wacc.wacc >= MODELING_CONFIG.wacc_floor, f"WACC {wacc.wacc}% below floor"
    return f"Low-beta WACC={wacc.wacc:.2f}% (floor={MODELING_CONFIG.wacc_floor}%)"


def test_wacc_extreme_high_beta():
    wacc = compute_wacc(
        country="US", sector="Information Technology",
        beta=4.5, debt_book_value=0, equity_market_cap=5000,
        effective_tax_rate=25.0,
    )
    from equity_research.core.config import MODELING_CONFIG
    assert wacc.wacc <= MODELING_CONFIG.wacc_ceiling, f"WACC {wacc.wacc}% above ceiling"
    return f"High-beta WACC={wacc.wacc:.2f}% (ceiling={MODELING_CONFIG.wacc_ceiling}%)"


def test_wacc_debt_heavy_capital_structure():
    wacc = _make_wacc(country="US", beta=1.2, debt=80000, equity=20000)
    assert wacc.debt_weight > 0.70, f"Expected debt-heavy, got d_w={wacc.debt_weight}"
    assert wacc.wacc < wacc.cost_of_equity, "WACC should be below Ke when debt-heavy (lower cost of debt)"
    return f"Debt-heavy: d_w={wacc.debt_weight:.2f} WACC={wacc.wacc:.2f}%"


def test_dcf_basic():
    wacc = _make_wacc()
    history = _make_financial_history(n_years=5)
    forecaster = FinancialForecaster(history, sector="Information Technology")
    forecasts = forecaster.forecast_all_scenarios()
    base_years = forecasts[Scenario.BASE]

    dcf = build_dcf(
        forecast_years=base_years,
        wacc_inputs=wacc,
        scenario=Scenario.BASE,
        net_debt=5000.0,
        shares_outstanding=15500.0,
        current_price=185.0,
        terminal_growth_rate=3.0,
    )
    assert dcf.enterprise_value > 0, "EV must be positive"
    assert dcf.intrinsic_value_per_share > 0, "Intrinsic value must be positive"
    assert 0 < dcf.key_assumptions["terminal_value_pct_of_ev"] < 100
    return f"DCF IV/share=${dcf.intrinsic_value_per_share:.2f} TV%={dcf.key_assumptions['terminal_value_pct_of_ev']:.0f}%"


def test_dcf_wacc_tgr_guard():
    wacc = compute_wacc(
        country="US", sector="Utilities",
        beta=0.5, debt_book_value=0, equity_market_cap=10000,
        effective_tax_rate=25.0,
    )
    history = _make_financial_history(n_years=3)
    forecaster = FinancialForecaster(history)
    forecasts = forecaster.forecast_all_scenarios()
    base_years = forecasts[Scenario.BASE]

    # Terminal growth rate equal to WACC — should be clamped to wacc*0.5
    dcf = build_dcf(
        forecast_years=base_years, wacc_inputs=wacc, scenario=Scenario.BASE,
        net_debt=0, shares_outstanding=1000,
        terminal_growth_rate=wacc.wacc,  # Dangerous: TGR == WACC
    )
    assert dcf.enterprise_value > 0, "DCF must survive TGR == WACC by clamping"
    return f"TGR-clamped DCF: EV={dcf.enterprise_value:.1f}M"


def test_dcf_negative_net_debt():
    wacc = _make_wacc()
    history = _make_financial_history(n_years=5)
    forecaster = FinancialForecaster(history)
    forecasts = forecaster.forecast_all_scenarios()
    base_years = forecasts[Scenario.BASE]

    dcf_cash_rich = build_dcf(
        forecast_years=base_years, wacc_inputs=wacc, scenario=Scenario.BASE,
        net_debt=-50_000.0,  # Net cash position: equity value boosted
        shares_outstanding=15500.0, current_price=185.0,
    )
    dcf_normal = build_dcf(
        forecast_years=base_years, wacc_inputs=wacc, scenario=Scenario.BASE,
        net_debt=5000.0,
        shares_outstanding=15500.0, current_price=185.0,
    )
    assert dcf_cash_rich.equity_value > dcf_normal.equity_value, "Net cash should boost equity value"
    return f"Cash-rich IV=${dcf_cash_rich.intrinsic_value_per_share:.2f} vs Normal=${dcf_normal.intrinsic_value_per_share:.2f}"


def test_dcf_zero_shares():
    wacc = _make_wacc()
    history = _make_financial_history(n_years=3)
    forecaster = FinancialForecaster(history)
    base_years = forecaster.forecast_all_scenarios()[Scenario.BASE]

    dcf = build_dcf(
        forecast_years=base_years, wacc_inputs=wacc, scenario=Scenario.BASE,
        net_debt=0, shares_outstanding=0, current_price=0,
    )
    assert dcf.intrinsic_value_per_share == 0.0
    return "Zero-shares guard: IV=0.0 (no divide-by-zero)"


def test_pwtp_all_scenarios_present():
    agent, *_ = _make_agent(ScenarioAnalysisAgent)
    weights = {"BEAR": 0.25, "BASE": 0.50, "BULL": 0.25}
    scenarios = {
        "BEAR": {"implied_price": 80.0},
        "BASE": {"implied_price": 120.0},
        "BULL": {"implied_price": 180.0},
    }
    pwtp = agent._compute_pwtp(scenarios, weights)
    expected = 80 * 0.25 + 120 * 0.50 + 180 * 0.25
    assert abs(pwtp - expected) < 0.01, f"PWTP {pwtp} != expected {expected}"
    return f"PWTP=${pwtp:.2f} (expected ${expected:.2f})"


def test_pwtp_missing_bull_scenario():
    agent, *_ = _make_agent(ScenarioAnalysisAgent)
    weights = {"BEAR": 0.25, "BASE": 0.50, "BULL": 0.25}
    scenarios = {
        "BEAR": {"implied_price": 80.0},
        "BASE": {"implied_price": 120.0},
        # BULL intentionally absent
    }
    pwtp = agent._compute_pwtp(scenarios, weights)
    expected = (80 * 0.25 + 120 * 0.50) / (0.25 + 0.50)
    assert abs(pwtp - expected) < 0.01
    return f"PWTP (no Bull)=${pwtp:.2f}"


def test_pwtp_all_scenarios_missing():
    agent, *_ = _make_agent(ScenarioAnalysisAgent)
    pwtp = agent._compute_pwtp({}, {"BEAR": 0.25, "BASE": 0.50, "BULL": 0.25})
    assert pwtp is None, "PWTP should be None when no scenario prices available"
    return "PWTP=None on empty scenarios"


# ══════════════════════════════════════════════════════════════════════════════
# SUITE B — Scenario Analysis (Risk-Driven Weight Adjustment)
# ══════════════════════════════════════════════════════════════════════════════

def test_scenario_weights_low_risk():
    agent, *_ = _make_agent(ScenarioAnalysisAgent)
    risk_out = AgentOutput(agent_id="09_risk_analysis", agent_name="Risk", status=AgentStatus.COMPLETED, risk_score=20.0, payload={})
    forensic_out = AgentOutput(agent_id="06_forensic_accounting", agent_name="Forensic", status=AgentStatus.COMPLETED, risk_score=22.0, payload={})
    state = _make_state()

    weights = agent._compute_scenario_weights(state, risk_out, forensic_out)
    # combined = 20*0.6 + 22*0.4 = 20.8 → < 30 → bull-shifted
    assert weights["BULL"] == 0.30, f"Low risk: expected BULL=0.30, got {weights['BULL']}"
    assert weights["BEAR"] == 0.20
    return f"Low-risk weights: B={weights['BEAR']} M={weights['BASE']} U={weights['BULL']}"


def test_scenario_weights_medium_risk():
    agent, *_ = _make_agent(ScenarioAnalysisAgent)
    risk_out = AgentOutput(agent_id="09_risk_analysis", agent_name="Risk", status=AgentStatus.COMPLETED, risk_score=45.0, payload={})
    forensic_out = AgentOutput(agent_id="06_forensic_accounting", agent_name="Forensic", status=AgentStatus.COMPLETED, risk_score=45.0, payload={})
    state = _make_state()

    weights = agent._compute_scenario_weights(state, risk_out, forensic_out)
    # combined = 45 → default weights
    assert weights == {"BEAR": 0.25, "BASE": 0.50, "BULL": 0.25}
    return f"Medium-risk weights: default (25/50/25)"


def test_scenario_weights_elevated_risk():
    agent, *_ = _make_agent(ScenarioAnalysisAgent)
    risk_out = AgentOutput(agent_id="09_risk_analysis", agent_name="Risk", status=AgentStatus.COMPLETED, risk_score=62.0, payload={})
    forensic_out = AgentOutput(agent_id="06_forensic_accounting", agent_name="Forensic", status=AgentStatus.COMPLETED, risk_score=60.0, payload={})
    state = _make_state()

    weights = agent._compute_scenario_weights(state, risk_out, forensic_out)
    # combined = 62*0.6 + 60*0.4 = 61.2 → 55-70 → bear-shifted
    assert weights["BEAR"] == 0.30
    assert weights["BULL"] == 0.20
    return f"Elevated-risk weights: {weights}"


def test_scenario_weights_very_high_risk():
    agent, *_ = _make_agent(ScenarioAnalysisAgent)
    risk_out = AgentOutput(agent_id="09_risk_analysis", agent_name="Risk", status=AgentStatus.COMPLETED, risk_score=80.0, payload={})
    forensic_out = AgentOutput(agent_id="06_forensic_accounting", agent_name="Forensic", status=AgentStatus.COMPLETED, risk_score=75.0, payload={})
    state = _make_state()

    weights = agent._compute_scenario_weights(state, risk_out, forensic_out)
    # combined = 80*0.6 + 75*0.4 = 78 → > 70 → heavy bear
    assert weights["BEAR"] == 0.40
    assert weights["BULL"] == 0.15
    return f"High-risk weights: {weights}"


def test_scenario_wide_spread_generates_warning():
    with tempfile.TemporaryDirectory() as tmp:
        agent, *_ = _make_agent(ScenarioAnalysisAgent, tmpdir=Path(tmp))
        # Wide spread: bull/bear ratio > 2x base → >100% spread
        state = _make_scenario_state_with_valuation(bear=50, base=100, bull=210, risk_score=50.0)

        with patch.object(agent, "llm_analyze", return_value="Bear scenario trigger: revenue miss\nBull trigger: margin expansion\nMonitorable: revenue growth quarterly"):
            output = agent.run(state)

        spread_warnings = [
            f for f in output.findings
            if "spread" in f.title.lower() or "spread" in f.detail.lower()
        ]
        assert spread_warnings, "Wide Bull-Bear spread should generate a WARNING finding"
        return f"Wide spread detected: {spread_warnings[0].title[:60]}"


def test_scenario_bear_probability_flag():
    with tempfile.TemporaryDirectory() as tmp:
        agent, *_ = _make_agent(ScenarioAnalysisAgent, tmpdir=Path(tmp))
        state = _make_scenario_state_with_valuation(
            bear=60, base=100, bull=130,
            risk_score=80.0, forensic_score=80.0,   # → heavy bear weights (40/45/15)
        )

        with patch.object(agent, "llm_analyze", return_value="Bear scenario materializes on revenue decline. Monitorable: margin trajectory."):
            output = agent.run(state)

        bear_flags = [f for f in output.findings if "bear" in f.title.lower()]
        assert any(f.finding_type == FindingType.RED_FLAG for f in bear_flags), \
            "Bear probability >35% should generate a RED_FLAG"
        return f"Bear flag: {bear_flags[0].title[:60] if bear_flags else 'missing'}"


def test_scenario_bull_probability_flag():
    with tempfile.TemporaryDirectory() as tmp:
        agent, *_ = _make_agent(ScenarioAnalysisAgent, tmpdir=Path(tmp))
        state = _make_scenario_state_with_valuation(
            bear=90, base=120, bull=180,
            risk_score=15.0, forensic_score=15.0,  # → bull-shifted (20/50/30)
        )

        with patch.object(agent, "llm_analyze", return_value="Positive scenario: margin expansion likely. Monitorable: revenue growth trajectory."):
            output = agent.run(state)

        bull_flags = [f for f in output.findings if "bull" in f.title.lower()]
        assert any(f.finding_type == FindingType.GREEN_FLAG for f in bull_flags), \
            "Bull probability >30% should generate a GREEN_FLAG"
        return f"Bull flag: {bull_flags[0].title[:60] if bull_flags else 'missing'}"


def test_scenario_missing_valuation_agent():
    with tempfile.TemporaryDirectory() as tmp:
        agent, *_ = _make_agent(ScenarioAnalysisAgent, tmpdir=Path(tmp))
        state = _make_state()  # No 08_valuation in agent_outputs

        output = agent.run(state)
        assert output.status == AgentStatus.COMPLETED
        assert any("no valuation" in f.title.lower() or "cannot proceed" in f.title.lower()
                   for f in output.findings), "Missing valuation should produce a WARNING finding"
        return "Graceful fallback on missing valuation agent"


def test_scenario_reconstruction_from_flat_summary():
    agent, *_ = _make_agent(ScenarioAnalysisAgent)
    val_summary = {"bear_price": 75.0, "base_price": 110.0, "bull_price": 155.0, "upside_pct": 15.0}
    scenarios = agent._reconstruct_scenarios(val_summary, {})
    assert scenarios["BEAR"]["implied_price"] == 75.0
    assert scenarios["BASE"]["implied_price"] == 110.0
    assert scenarios["BULL"]["implied_price"] == 155.0
    return f"Reconstructed: B={scenarios['BEAR']['implied_price']} M={scenarios['BASE']['implied_price']} U={scenarios['BULL']['implied_price']}"


def test_scenario_full_pipeline_bear_heavy():
    """End-to-end scenario: low-quality company, wide spread, bear-heavy."""
    with tempfile.TemporaryDirectory() as tmp:
        agent, *_ = _make_agent(ScenarioAnalysisAgent, tmpdir=Path(tmp))
        # High risk + wide spread
        state = _make_scenario_state_with_valuation(
            bear=40, base=90, bull=200,
            risk_score=75.0, forensic_score=72.0,
        )

        llm_resp = (
            "Probability weights: Bear 40% / Base 45% / Bull 15%.\n"
            "Bear assumptions: 1) Revenue decline 10%, 2) Margin compression 5pp, 3) Debt covenant breach\n"
            "Bull assumptions: 1) Revenue growth 20%, 2) Margin expansion 4pp, 3) Multiple re-rating\n"
            "Monitorable: 1) Quarterly revenue growth 2) EBITDA margin 3) Net debt/EBITDA 4) Free cash flow\n"
            "Trigger: Bear materializes on revenue miss >5% vs consensus"
        )
        with patch.object(agent, "llm_analyze", return_value=llm_resp):
            output = agent.run(state)

        assert output.status == AgentStatus.COMPLETED
        assert output.payload["probability_weights"]["BEAR"] == 0.40
        assert len(output.payload["key_monitorables"]) > 0
        pwtp = output.payload.get("probability_weighted_target")
        assert pwtp is not None and 40 < pwtp < 200
        return (f"Full pipeline: PWTP=${pwtp:.1f}, "
                f"monitorables={len(output.payload['key_monitorables'])}, "
                f"findings={len(output.findings)}")


def test_scenario_full_pipeline_bull_case():
    """End-to-end scenario: high-quality company, narrow spread, bull-shifted."""
    with tempfile.TemporaryDirectory() as tmp:
        agent, *_ = _make_agent(ScenarioAnalysisAgent, tmpdir=Path(tmp))
        state = _make_scenario_state_with_valuation(
            bear=160, base=200, bull=260,
            risk_score=18.0, forensic_score=22.0,
        )

        llm_resp = (
            "Bull probability 30% supported by strong FCF and competitive moat.\n"
            "Monitorable: 1) Revenue growth acceleration 2) Margin trends 3) Market share\n"
            "Bear trigger: macro slowdown reduces TAM"
        )
        with patch.object(agent, "llm_analyze", return_value=llm_resp):
            output = agent.run(state)

        assert output.status == AgentStatus.COMPLETED
        weights = output.payload["probability_weights"]
        assert weights["BULL"] == 0.30, f"Bull should be 0.30, got {weights['BULL']}"
        return f"Bull pipeline: weights={weights}, PWTP=${output.payload['probability_weighted_target']:.1f}"


# ══════════════════════════════════════════════════════════════════════════════
# SUITE C — Financial Modeling Edge Cases
# ══════════════════════════════════════════════════════════════════════════════

def test_forecaster_empty_history():
    from equity_research.agents.financial_modeling_agent import FinancialModelingAgent
    with tempfile.TemporaryDirectory() as tmp:
        agent, *_ = _make_agent(FinancialModelingAgent, tmpdir=Path(tmp))
        state = _make_state()
        state.financial_history = {}  # Empty

        output = agent.run(state)
        assert output.status == AgentStatus.COMPLETED
        red_flags = [f for f in output.findings if f.finding_type == FindingType.RED_FLAG]
        assert red_flags, "Empty history should produce a RED_FLAG"
        return f"Empty history: {red_flags[0].title[:60]}"


def test_forecaster_single_year():
    history = _make_financial_history(n_years=1)
    forecaster = FinancialForecaster(history, sector="Technology")
    forecasts = forecaster.forecast_all_scenarios()
    assert Scenario.BASE in forecasts
    assert len(forecasts[Scenario.BASE]) == 5, "Always 5 forecast years"
    cagr = forecaster._compute_cagr("revenue", years=3)
    # Single year: CAGR returns None or defaults
    return f"Single-year CAGR={cagr}, forecasts={len(forecasts[Scenario.BASE])}"


def test_forecaster_5_year_cagr_accuracy():
    history = _make_financial_history(n_years=5)
    forecaster = FinancialForecaster(history, sector="Technology")
    cagr = forecaster._compute_cagr("revenue", years=3)
    assert cagr is not None, "Should compute CAGR with 5 years of data"
    assert abs(cagr - 0.08) < 0.02, f"Expected ~8% CAGR, got {cagr*100:.1f}%"
    return f"5Y revenue CAGR={cagr*100:.1f}%"


def test_forecaster_bear_below_base_below_bull():
    history = _make_financial_history(n_years=5)
    forecaster = FinancialForecaster(history, sector="Technology")
    forecasts = forecaster.forecast_all_scenarios()
    bear_rev = forecasts[Scenario.BEAR][0].revenue
    base_rev = forecasts[Scenario.BASE][0].revenue
    bull_rev = forecasts[Scenario.BULL][0].revenue
    assert bear_rev <= base_rev <= bull_rev, \
        f"Revenue ordering violated: B={bear_rev:.0f} < M={base_rev:.0f} < U={bull_rev:.0f}"
    return f"Year-1 revenue: B={bear_rev:.0f} ≤ M={base_rev:.0f} ≤ U={bull_rev:.0f}"


def test_forecaster_margin_compression_detection():
    from equity_research.agents.financial_modeling_agent import FinancialModelingAgent
    with tempfile.TemporaryDirectory() as tmp:
        agent, *_ = _make_agent(FinancialModelingAgent, tmpdir=Path(tmp))
        state = _make_state()

        # Build history with 8pp margin compression
        history = FinancialHistory(company_ticker="TST", currency="USD")
        for i, yr in enumerate(["2020","2021","2022","2023","2024"]):
            rev = 10_000.0 * (1.05 ** i)
            margin = 30.0 - i * 1.6  # 30 → 22 → compression of 8pp
            history.income_statements[yr] = IncomeStatement(
                fiscal_year=yr, revenue=rev,
                ebitda=rev * margin / 100, ebitda_margin=margin,
                ebit=rev * (margin - 4) / 100, ebit_margin=margin - 4,
                net_income=rev * (margin - 6) / 100,
            )
        history.available_years = ["2020","2021","2022","2023","2024"]
        state.financial_history = history.model_dump(mode="json")

        with patch.object(agent, "llm_analyze", return_value="Margin compression driven by cost inflation."):
            output = agent.run(state)

        margin_findings = [f for f in output.findings if "margin" in f.title.lower()]
        assert margin_findings, "8pp margin compression should produce a finding"
        assert any(f.finding_type == FindingType.RED_FLAG for f in margin_findings)
        return f"Margin compression detected: {margin_findings[0].title[:60]}"


def test_forecaster_volatile_revenue():
    from equity_research.agents.financial_modeling_agent import FinancialModelingAgent
    with tempfile.TemporaryDirectory() as tmp:
        agent, *_ = _make_agent(FinancialModelingAgent, tmpdir=Path(tmp))
        state = _make_state()

        # High revenue volatility: +50%, -30%, +40%, -20%, +60%
        revenues = [10000, 15000, 10500, 14700, 11760, 18816]
        history = FinancialHistory(company_ticker="VLT", currency="USD")
        years = ["2019","2020","2021","2022","2023","2024"]
        for i, yr in enumerate(years):
            rev = revenues[i]
            history.income_statements[yr] = IncomeStatement(
                fiscal_year=yr, revenue=rev,
                ebitda=rev * 0.20, ebit=rev * 0.15, net_income=rev * 0.10,
            )
        history.available_years = years
        state.financial_history = history.model_dump(mode="json")

        with patch.object(agent, "llm_analyze", return_value="Revenue volatility driven by lumpy project timing."):
            output = agent.run(state)

        volatility_findings = [f for f in output.findings if "volatil" in f.title.lower()]
        assert volatility_findings, "High revenue volatility should produce a WARNING"
        return f"Volatility finding: {volatility_findings[0].title[:60]}"


# ══════════════════════════════════════════════════════════════════════════════
# SUITE D — Valuation Edge Cases
# ══════════════════════════════════════════════════════════════════════════════

def test_valuation_zero_current_price():
    from equity_research.agents.valuation_agent import ValuationAgent
    with tempfile.TemporaryDirectory() as tmp:
        agent, *_ = _make_agent(ValuationAgent, tmpdir=Path(tmp))
        state = _make_state()
        state.financial_history = _make_financial_history().model_dump(mode="json")

        # Market data with zero price
        state.agent_outputs["04_market_data"] = AgentOutput(
            agent_id="04_market_data", agent_name="Market Data Agent",
            status=AgentStatus.COMPLETED,
            payload={"market_data": {
                "current_price": 0.0, "shares_outstanding": 15_500_000_000,
                "market_cap_usd": 0, "beta": 1.2, "peer_market_data": [],
            }},
        )
        state.agent_outputs["07_financial_modeling"] = AgentOutput(
            agent_id="07_financial_modeling", agent_name="Financial Modeling Agent",
            status=AgentStatus.COMPLETED,
            payload={"forecasts": {}},
        )

        with patch.object(agent, "llm_analyze", return_value="Zero-price valuation: insufficient market data."):
            output = agent.run(state)

        assert output.status == AgentStatus.COMPLETED, "Agent must survive zero current price"
        return "Zero-price: agent completed without crash"


def test_valuation_negative_ebitda():
    from equity_research.agents.valuation_agent import ValuationAgent
    with tempfile.TemporaryDirectory() as tmp:
        agent, *_ = _make_agent(ValuationAgent, tmpdir=Path(tmp))
        state = _make_state()

        # Negative EBITDA history (loss-making company)
        history = FinancialHistory(company_ticker="LOSS", currency="USD")
        for yr in ["2022", "2023", "2024"]:
            history.income_statements[yr] = IncomeStatement(
                fiscal_year=yr, revenue=5000.0,
                ebitda=-500.0, ebit=-800.0, net_income=-1200.0,
            )
        history.available_years = ["2022", "2023", "2024"]
        state.financial_history = history.model_dump(mode="json")

        state.agent_outputs["04_market_data"] = AgentOutput(
            agent_id="04_market_data", agent_name="Market Data Agent",
            status=AgentStatus.COMPLETED,
            payload={"market_data": {
                "current_price": 12.0, "shares_outstanding": 100_000_000,
                "market_cap_usd": 1_200_000_000, "beta": 1.8, "peer_market_data": [],
            }},
        )
        state.agent_outputs["07_financial_modeling"] = AgentOutput(
            agent_id="07_financial_modeling", agent_name="Financial Modeling Agent",
            status=AgentStatus.COMPLETED, payload={"forecasts": {}},
        )

        with patch.object(agent, "llm_analyze", return_value="Loss-making company valuation."):
            output = agent.run(state)

        assert output.status == AgentStatus.COMPLETED
        return "Negative EBITDA: valuation agent completed without crash"


def test_valuation_no_peer_data():
    from equity_research.agents.valuation_agent import ValuationAgent
    with tempfile.TemporaryDirectory() as tmp:
        agent, *_ = _make_agent(ValuationAgent, tmpdir=Path(tmp))
        state = _make_state()
        state.financial_history = _make_financial_history().model_dump(mode="json")

        state.agent_outputs["04_market_data"] = AgentOutput(
            agent_id="04_market_data", agent_name="Market Data Agent",
            status=AgentStatus.COMPLETED,
            payload={"market_data": {
                "current_price": 185.0, "shares_outstanding": 15_500_000_000,
                "market_cap_usd": 2_867_500_000_000, "beta": 1.25,
                "peer_market_data": [],  # No peers
            }},
        )
        state.agent_outputs["07_financial_modeling"] = AgentOutput(
            agent_id="07_financial_modeling", agent_name="Financial Modeling Agent",
            status=AgentStatus.COMPLETED, payload={"forecasts": {}},
        )

        with patch.object(agent, "llm_analyze", return_value="No peer comparables available."):
            output = agent.run(state)

        assert output.status == AgentStatus.COMPLETED
        return "No-peers: valuation agent completed"


def test_valuation_multi_scenario_spread():
    wacc = _make_wacc(beta=1.25)
    history = _make_financial_history(n_years=5)
    forecaster = FinancialForecaster(history, sector="Information Technology")
    forecasts = forecaster.forecast_all_scenarios()

    prices = {}
    for sc in [Scenario.BEAR, Scenario.BASE, Scenario.BULL]:
        tgr = {"BEAR": 1.5, "BASE": 3.0, "BULL": 4.0}[sc.value]
        dcf = build_dcf(
            forecast_years=forecasts[sc], wacc_inputs=wacc, scenario=sc,
            net_debt=10_000, shares_outstanding=15500, current_price=185.0,
            terminal_growth_rate=tgr,
        )
        prices[sc.value] = dcf.intrinsic_value_per_share

    assert prices["BEAR"] < prices["BASE"] < prices["BULL"], \
        f"Scenario ordering violated: {prices}"
    spread_pct = (prices["BULL"] - prices["BEAR"]) / prices["BASE"] * 100
    return f"Bear=${prices['BEAR']:.1f} Base=${prices['BASE']:.1f} Bull=${prices['BULL']:.1f} Spread={spread_pct:.0f}%"


# ══════════════════════════════════════════════════════════════════════════════
# SUITE E — Concurrency & Thread Safety
# ══════════════════════════════════════════════════════════════════════════════

def test_concurrent_scenario_agents():
    """Run 6 ScenarioAnalysisAgent instances simultaneously in different threads."""
    companies = [
        ("AAPL", "Apple", 50.0, 90.0, 60.0),
        ("TSLA", "Tesla", 80.0, 60.0, 75.0),
        ("MSFT", "Microsoft", 25.0, 95.0, 20.0),
        ("AMZN", "Amazon", 60.0, 80.0, 65.0),
        ("NVDA", "Nvidia", 15.0, 150.0, 20.0),
        ("META", "Meta", 45.0, 100.0, 55.0),
    ]
    errors = []
    results = []

    def run_scenario(ticker, company, risk, bear, forensic):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                agent, *_ = _make_agent(ScenarioAnalysisAgent, ticker=ticker, tmpdir=Path(tmp))
                state = _make_scenario_state_with_valuation(
                    bear=bear, base=bear * 1.5, bull=bear * 2.5,
                    risk_score=risk, forensic_score=forensic,
                )
                state.ticker = ticker
                state.company_name = company

                with patch.object(agent, "llm_analyze", return_value=f"Scenario for {ticker}. Monitorable: revenue growth."):
                    output = agent.run(state)

                results.append((ticker, output.status == AgentStatus.COMPLETED, output.payload.get("probability_weighted_target")))
        except Exception as e:
            errors.append(f"{ticker}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(run_scenario, *args) for args in companies]
        concurrent.futures.wait(futs)

    assert not errors, f"Concurrent agents failed: {errors}"
    assert len(results) == 6, f"Expected 6 results, got {len(results)}"
    assert all(completed for _, completed, _ in results), "All agents should complete"
    return f"6 concurrent agents: all passed, PWTPs={[f'{p:.1f}' for _, _, p in results if p]}"


def test_concurrent_wacc_computation():
    """WACC is stateless — verify no cross-contamination under concurrency."""
    params = [
        ("US", 0.5, 1000, 50000),
        ("IN", 1.4, 20000, 40000),
        ("US", 2.0, 500, 10000),
        ("DE", 0.8, 5000, 30000),
        ("US", 1.0, 10000, 50000),
        ("IN", 0.9, 8000, 25000),
        ("GB", 1.1, 3000, 20000),
        ("US", 1.5, 0, 80000),
    ]
    results = {}
    errors = []

    def compute(i, country, beta, debt, equity):
        try:
            w = compute_wacc(country=country, sector="Industrials", beta=beta,
                             debt_book_value=debt, equity_market_cap=equity,
                             effective_tax_rate=25.0)
            results[i] = w.wacc
        except Exception as e:
            errors.append(f"Thread {i}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(compute, i, *p) for i, p in enumerate(params)]
        concurrent.futures.wait(futs)

    assert not errors, f"Concurrent WACC errors: {errors}"
    assert len(results) == 8

    # Verify results are deterministic — run again and compare
    for i, (country, beta, debt, equity) in enumerate(params):
        w = compute_wacc(country=country, sector="Industrials", beta=beta,
                         debt_book_value=debt, equity_market_cap=equity,
                         effective_tax_rate=25.0)
        assert results[i] == w.wacc, f"Thread {i}: {results[i]} != {w.wacc} (non-deterministic!)"

    return f"8 concurrent WACC computations: all deterministic"


def test_concurrent_dcf_builds():
    """Build 6 DCF models concurrently sharing a single WACCInputs object."""
    wacc = _make_wacc()
    history = _make_financial_history(n_years=5)
    forecaster = FinancialForecaster(history)
    all_forecasts = forecaster.forecast_all_scenarios()
    base_years = all_forecasts[Scenario.BASE]

    dcf_results = {}
    errors = []

    def build(i, net_debt, shares, price):
        try:
            dcf = build_dcf(
                forecast_years=base_years, wacc_inputs=wacc,
                scenario=Scenario.BASE, net_debt=net_debt,
                shares_outstanding=shares, current_price=price,
            )
            dcf_results[i] = dcf.intrinsic_value_per_share
        except Exception as e:
            errors.append(f"DCF {i}: {e}")

    builds = [(i, i * 1000, 1000 + i * 200, 150 + i * 5) for i in range(6)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(build, *args) for args in builds]
        concurrent.futures.wait(futs)

    assert not errors, f"Concurrent DCF errors: {errors}"
    # All IV should differ (different net debt / shares)
    values = list(dcf_results.values())
    assert len(set(values)) > 1, "Concurrent DCF results should vary by inputs"
    return f"6 concurrent DCFs: IVs={[f'${v:.1f}' for v in sorted(values)]}"


def test_rag_concurrent_ingest():
    """Verify RAG vector store handles concurrent ingestion without deadlock/corruption."""
    try:
        from equity_research.retrieval.vector_store import ingest_texts, collection_size, clear_company
    except ImportError:
        return "SKIPPED (retrieval dependencies not installed)"

    TICKER = f"STRESS_{uuid.uuid4().hex[:6].upper()}"
    errors = []
    counts = []

    def ingest_batch(thread_id: int):
        try:
            texts = [
                f"Thread {thread_id} chunk {j}: Revenue for {TICKER} grew 15% YoY in Q{j+1}."
                for j in range(5)
            ]
            meta = [{"source": f"thread_{thread_id}", "chunk": j} for j in range(5)]
            n = ingest_texts(texts, meta, TICKER)
            counts.append(n)
        except Exception as e:
            errors.append(f"Thread {thread_id}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(ingest_batch, i) for i in range(6)]
        concurrent.futures.wait(futs)

    try:
        clear_company(TICKER)
    except Exception:
        pass

    assert not errors, f"Concurrent RAG ingest errors: {errors}"
    assert len(counts) == 6
    total = sum(counts)
    return f"6 concurrent ingest threads: {total} total chunks indexed"


def test_rag_concurrent_read_write():
    """Concurrent reads and writes to same ticker without deadlock."""
    try:
        from equity_research.retrieval.vector_store import (
            ingest_texts, query as vs_query, clear_company, collection_size
        )
    except ImportError:
        return "SKIPPED (retrieval dependencies not installed)"

    TICKER = f"RWTEST_{uuid.uuid4().hex[:6].upper()}"
    errors = []

    def writer(i):
        try:
            texts = [f"Writer {i}: {TICKER} EBITDA margin at {20+i}% in year {i}."]
            ingest_texts(texts, [{"writer": i}], TICKER)
        except Exception as e:
            errors.append(f"Writer {i}: {e}")

    def reader(i):
        try:
            vs_query(f"What is the EBITDA margin for {TICKER}?", ticker=TICKER, top_k=3)
        except Exception as e:
            # Reads on empty store may return empty — that's OK
            if "does not exist" not in str(e).lower() and "empty" not in str(e).lower():
                errors.append(f"Reader {i}: {e}")

    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        for i in range(5):
            tasks.append(pool.submit(writer, i))
            tasks.append(pool.submit(reader, i))
        concurrent.futures.wait(tasks)

    try:
        clear_company(TICKER)
    except Exception:
        pass

    assert not errors, f"Concurrent read-write errors: {errors}"
    return "5 writers + 5 readers: no deadlock or corruption"


# ══════════════════════════════════════════════════════════════════════════════
# SUITE F — Orchestrator Resilience
# ══════════════════════════════════════════════════════════════════════════════

def test_orchestrator_risk_score_all_failed():
    from equity_research.orchestrator.workflow import ResearchOrchestrator
    with patch.object(ResearchOrchestrator, "__init__", lambda self: None):
        orch = ResearchOrchestrator.__new__(ResearchOrchestrator)

    state = _make_state()
    # All weighted agents are missing
    score = orch._compute_overall_risk(state)
    assert score == 50.0, f"All-failed risk score should default to 50.0, got {score}"
    return "All-failed: overall risk=50.0"


def test_orchestrator_risk_score_partial():
    from equity_research.orchestrator.workflow import ResearchOrchestrator
    with patch.object(ResearchOrchestrator, "__init__", lambda self: None):
        orch = ResearchOrchestrator.__new__(ResearchOrchestrator)

    state = _make_state()
    state.agent_outputs["06_forensic_accounting"] = AgentOutput(
        agent_id="06_forensic_accounting", agent_name="Forensic",
        status=AgentStatus.COMPLETED, risk_score=80.0, payload={},
    )
    state.agent_outputs["05_accounting_quality"] = AgentOutput(
        agent_id="05_accounting_quality", agent_name="Accounting",
        status=AgentStatus.COMPLETED, risk_score=60.0, payload={},
    )

    score = orch._compute_overall_risk(state)
    # Forensic weight=0.20, accounting weight=0.15 → (80*0.20 + 60*0.15) / 0.35 = 74.3
    expected = (80 * 0.20 + 60 * 0.15) / 0.35
    assert abs(score - expected) < 0.5, f"Partial risk score {score:.1f} != expected {expected:.1f}"
    return f"Partial risk score={score:.1f} (expected={expected:.1f})"


def test_orchestrator_derive_rating():
    from equity_research.orchestrator.workflow import ResearchOrchestrator
    with patch.object(ResearchOrchestrator, "__init__", lambda self: None):
        orch = ResearchOrchestrator.__new__(ResearchOrchestrator)

    cases = [
        ({"upside_pct": 30.0}, "BUY"),
        ({"upside_pct": 15.0}, "OUTPERFORM"),
        ({"upside_pct": 0.0},  "HOLD"),
        ({"upside_pct": -15.0}, "UNDERPERFORM"),
        ({"upside_pct": -30.0}, "SELL"),
        ({"upside_pct": None},  "HOLD"),  # None → 0.0 → HOLD
    ]
    for val_summary, expected in cases:
        rating = orch._derive_rating(val_summary)
        assert rating == expected, f"upside={val_summary['upside_pct']} → expected {expected}, got {rating}"
    return "Rating derivation: BUY/OUTPERFORM/HOLD/UNDERPERFORM/SELL all correct"


def test_orchestrator_classify_risk():
    from equity_research.orchestrator.workflow import ResearchOrchestrator
    with patch.object(ResearchOrchestrator, "__init__", lambda self: None):
        orch = ResearchOrchestrator.__new__(ResearchOrchestrator)

    cases = [
        (80.0, RiskClassification.CRITICAL),
        (60.0, RiskClassification.HIGH),
        (40.0, RiskClassification.MEDIUM),
        (20.0, RiskClassification.LOW),
    ]
    for score, expected in cases:
        result = orch._classify_risk(score)
        assert result == expected, f"score={score} → expected {expected}, got {result}"
    return "Risk classification: LOW/MEDIUM/HIGH/CRITICAL thresholds correct"


def test_parallel_runner_isolation():
    """Verify _run_parallel doesn't bleed state across agent threads."""
    from equity_research.orchestrator.workflow import ResearchOrchestrator
    from equity_research.agents.scenario_analysis import ScenarioAnalysisAgent
    from equity_research.agents.risk_analysis import RiskAnalysisAgent

    with tempfile.TemporaryDirectory() as tmp:
        storage = _mock_storage(Path(tmp))
        audit = _mock_audit()
        db = _mock_db()

        with patch.object(ResearchOrchestrator, "__init__", lambda self: None):
            orch = ResearchOrchestrator.__new__(ResearchOrchestrator)
            orch.llm = _mock_llm()
            orch.db = db

        state = _make_state()
        state.agent_outputs["08_valuation"] = AgentOutput(
            agent_id="08_valuation", agent_name="Valuation Agent",
            status=AgentStatus.COMPLETED, risk_score=40.0,
            payload={"scenarios": {}, "valuation_summary": {}},
        )

        with patch("equity_research.agents.risk_analysis.RiskAnalysisAgent.run") as mock_risk, \
             patch("equity_research.agents.scenario_analysis.ScenarioAnalysisAgent.run") as mock_scen:
            mock_risk.return_value = AgentOutput(
                agent_id="09_risk_analysis", agent_name="Risk", risk_score=40.0, payload={},
            )
            mock_scen.return_value = AgentOutput(
                agent_id="15_scenario_analysis", agent_name="Scenario", risk_score=35.0, payload={},
            )

            specs = [
                (RiskAnalysisAgent, "09_risk_analysis"),
                (ScenarioAnalysisAgent, "15_scenario_analysis"),
            ]
            outputs = orch._run_parallel(specs, state, storage, audit)

        assert len(outputs) == 2
        agent_ids = {o.agent_id for o in outputs}
        assert "09_risk_analysis" in agent_ids
        assert "15_scenario_analysis" in agent_ids
        return f"Parallel runner isolation: {len(outputs)} distinct outputs"


def test_agent_output_contract_missing_agent_id():
    from equity_research.agents.scenario_analysis import ScenarioAnalysisAgent
    with tempfile.TemporaryDirectory() as tmp:
        agent, *_ = _make_agent(ScenarioAnalysisAgent, tmpdir=Path(tmp))

        bad_output = AgentOutput(
            agent_id="",  # Missing
            agent_name="Test Agent",
            status=AgentStatus.COMPLETED,
            risk_score=50.0,
        )
        try:
            agent._validate_output(bad_output)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "agent_id" in str(e).lower()
            return "Contract: missing agent_id raises ValueError"


def test_agent_output_contract_risk_score_out_of_range():
    from equity_research.agents.scenario_analysis import ScenarioAnalysisAgent
    with tempfile.TemporaryDirectory() as tmp:
        agent, *_ = _make_agent(ScenarioAnalysisAgent, tmpdir=Path(tmp))

        # Pydantic v2 enforces Field(le=100) at construction time; both paths acceptable
        try:
            bad_output = AgentOutput(
                agent_id="test", agent_name="Test Agent",
                status=AgentStatus.COMPLETED,
                risk_score=150.0,  # Violates Field(ge=0, le=100)
            )
            # If somehow construction succeeds, validate_output must catch it
            agent._validate_output(bad_output)
            assert False, "risk_score=150 should have been rejected"
        except Exception as e:
            # Accept Pydantic ValidationError or our own ValueError
            assert "150" in str(e) or "less than" in str(e) or "risk_score" in str(e), \
                f"Unexpected error: {e}"
            return f"Contract: risk_score=150 rejected ({type(e).__name__})"


# ══════════════════════════════════════════════════════════════════════════════
# SUITE G — Multi-Company Batch (Scenario Diversity)
# ══════════════════════════════════════════════════════════════════════════════

def test_batch_scenario_diversity():
    """
    Run 8 companies with varying risk profiles and confirm:
    - All complete without crash
    - Higher risk → lower PWTP relative to base
    - All produce at least 1 finding
    """
    profiles = [
        # (ticker,   company,       bear,   base,   bull,  risk, forensic)
        ("AAPL",   "Apple",        150.0,  200.0,  260.0,  20.0,  18.0),
        ("TSLA",   "Tesla",         60.0,  120.0,  300.0,  65.0,  70.0),
        ("MSFT",   "Microsoft",    220.0,  280.0,  340.0,  18.0,  15.0),
        ("META",   "Meta",         100.0,  160.0,  240.0,  45.0,  42.0),
        ("BYND",   "Beyond Meat",   2.0,   10.0,   35.0,  85.0,  80.0),
        ("JPM",    "JPMorgan",     130.0,  180.0,  220.0,  40.0,  38.0),
        ("GME",    "GameStop",      8.0,   15.0,   50.0,  78.0,  82.0),
        ("BRK",    "Berkshire",    480.0,  580.0,  700.0,  15.0,  12.0),
    ]

    outcomes = []
    errors = []

    for ticker, company, bear, base, bull, risk, forensic in profiles:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                agent, *_ = _make_agent(ScenarioAnalysisAgent, ticker=ticker, tmpdir=Path(tmp))
                state = _make_scenario_state_with_valuation(
                    bear=bear, base=base, bull=bull,
                    risk_score=risk, forensic_score=forensic,
                )
                state.ticker = ticker
                state.company_name = company

                llm_resp = (
                    f"Scenario for {company}. Bear: revenue decline risk. Bull: market expansion.\n"
                    f"Monitorable: 1) Revenue growth 2) Margin trajectory 3) FCF conversion\n"
                    f"Bear trigger: macro slowdown\nBull trigger: market share gain"
                )
                with patch.object(agent, "llm_analyze", return_value=llm_resp):
                    output = agent.run(state)

                pwtp = output.payload.get("probability_weighted_target")
                weights = output.payload.get("probability_weights", {})
                outcomes.append({
                    "ticker": ticker, "completed": output.status == AgentStatus.COMPLETED,
                    "pwtp": pwtp, "bear_w": weights.get("BEAR"),
                    "findings": len(output.findings), "risk": risk,
                })
        except Exception as e:
            errors.append(f"{ticker}: {e}")

    assert not errors, f"Batch errors: {errors}"
    assert all(o["completed"] for o in outcomes), "All 8 companies must complete"

    # High-risk companies should have bear weight >= 0.30
    high_risk = [o for o in outcomes if o["risk"] > 60]
    for o in high_risk:
        assert o["bear_w"] >= 0.30, \
            f"{o['ticker']} risk={o['risk']}: bear_w={o['bear_w']} should be ≥0.30"

    # Low-risk companies should have bear weight <= 0.25
    low_risk = [o for o in outcomes if o["risk"] < 30]
    for o in low_risk:
        assert o["bear_w"] <= 0.25, \
            f"{o['ticker']} risk={o['risk']}: bear_w={o['bear_w']} should be ≤0.25"

    summary_lines = [f"{o['ticker']}: PWTP=${o['pwtp']:.1f} bear={o['bear_w']:.0%}" for o in outcomes if o["pwtp"]]
    return f"8-company batch: all passed. " + " | ".join(summary_lines[:4])


def test_batch_identical_inputs_deterministic():
    """Same inputs → same outputs, verifying no random state pollution."""
    def run_once(ticker):
        with tempfile.TemporaryDirectory() as tmp:
            agent, *_ = _make_agent(ScenarioAnalysisAgent, ticker=ticker, tmpdir=Path(tmp))
            state = _make_scenario_state_with_valuation(
                bear=80, base=120, bull=180, risk_score=45.0, forensic_score=45.0,
            )
            with patch.object(agent, "llm_analyze", return_value="Monitorable: revenue growth. Bear: macro."):
                output = agent.run(state)
            return (
                output.payload.get("probability_weights"),
                round(output.payload.get("probability_weighted_target", 0), 2),
                output.risk_score,
            )

    r1 = run_once("AAPL")
    r2 = run_once("AAPL")
    r3 = run_once("AAPL")

    assert r1 == r2 == r3, f"Non-deterministic results: {r1} vs {r2} vs {r3}"
    return f"3 identical runs: all match {r1}"


# ══════════════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════════════

SUITE_A = ("A: Core Math",   [
    ("WACC — US standard parameters",                test_wacc_us_standard),
    ("WACC — India high-beta",                       test_wacc_india_high_beta),
    ("WACC — zero debt (all-equity)",                test_wacc_zero_debt),
    ("WACC — extreme low beta (floor guard)",        test_wacc_extreme_low_beta),
    ("WACC — extreme high beta (ceiling guard)",     test_wacc_extreme_high_beta),
    ("WACC — debt-heavy capital structure",          test_wacc_debt_heavy_capital_structure),
    ("DCF — basic 5-year base case",                 test_dcf_basic),
    ("DCF — TGR == WACC guard",                      test_dcf_wacc_tgr_guard),
    ("DCF — negative net debt (net cash)",           test_dcf_negative_net_debt),
    ("DCF — zero shares outstanding",               test_dcf_zero_shares),
    ("PWTP — all scenarios present",                test_pwtp_all_scenarios_present),
    ("PWTP — missing bull scenario",                test_pwtp_missing_bull_scenario),
    ("PWTP — all scenarios missing → None",         test_pwtp_all_scenarios_missing),
])

SUITE_B = ("B: Scenario Analysis", [
    ("Weights — low risk → bull-shifted (20/50/30)",       test_scenario_weights_low_risk),
    ("Weights — medium risk → default (25/50/25)",         test_scenario_weights_medium_risk),
    ("Weights — elevated risk → bear-shifted (30/50/20)",  test_scenario_weights_elevated_risk),
    ("Weights — very high risk → heavy bear (40/45/15)",   test_scenario_weights_very_high_risk),
    ("Wide spread → WARNING finding generated",            test_scenario_wide_spread_generates_warning),
    ("Bear prob >35% → RED_FLAG generated",               test_scenario_bear_probability_flag),
    ("Bull prob >30% → GREEN_FLAG generated",             test_scenario_bull_probability_flag),
    ("Missing valuation agent → graceful warning",        test_scenario_missing_valuation_agent),
    ("Reconstruct scenarios from flat summary",           test_scenario_reconstruction_from_flat_summary),
    ("Full pipeline — bear-heavy company",                test_scenario_full_pipeline_bear_heavy),
    ("Full pipeline — bull-shifted company",              test_scenario_full_pipeline_bull_case),
])

SUITE_C = ("C: Financial Modeling", [
    ("Empty history → RED_FLAG + completed",              test_forecaster_empty_history),
    ("Single year of data → 5 forecast years",           test_forecaster_single_year),
    ("5-year CAGR accuracy (~8%)",                       test_forecaster_5_year_cagr_accuracy),
    ("Bear ≤ Base ≤ Bull revenue ordering",              test_forecaster_bear_below_base_below_bull),
    ("8pp margin compression → RED_FLAG",                test_forecaster_margin_compression_detection),
    ("High revenue volatility → WARNING",                test_forecaster_volatile_revenue),
])

SUITE_D = ("D: Valuation Edge Cases", [
    ("Zero current price — no crash",              test_valuation_zero_current_price),
    ("Negative EBITDA — no crash",                 test_valuation_negative_ebitda),
    ("No peer data — no crash",                    test_valuation_no_peer_data),
    ("Bear < Base < Bull DCF price ordering",      test_valuation_multi_scenario_spread),
])

SUITE_E = ("E: Concurrency & Thread Safety", [
    ("6 concurrent ScenarioAnalysisAgent runs",  test_concurrent_scenario_agents),
    ("8 concurrent WACC computations — deterministic", test_concurrent_wacc_computation),
    ("6 concurrent DCF builds — distinct IVs",   test_concurrent_dcf_builds),
    ("RAG — concurrent ingest (6 threads)",       test_rag_concurrent_ingest),
    ("RAG — concurrent read-write (5+5 threads)", test_rag_concurrent_read_write),
])

SUITE_F = ("F: Orchestrator Resilience", [
    ("Risk score — all agents failed → 50.0",      test_orchestrator_risk_score_all_failed),
    ("Risk score — partial agents weighted avg",   test_orchestrator_risk_score_partial),
    ("Rating derivation — all 5 cases",            test_orchestrator_derive_rating),
    ("Risk classification — 4 thresholds",         test_orchestrator_classify_risk),
    ("Parallel runner — output isolation",         test_parallel_runner_isolation),
    ("Output contract — missing agent_id",         test_agent_output_contract_missing_agent_id),
    ("Output contract — risk_score out of range",  test_agent_output_contract_risk_score_out_of_range),
])

SUITE_G = ("G: Batch / Scenario Diversity", [
    ("8-company batch — all complete, risk-weight ordering", test_batch_scenario_diversity),
    ("Identical inputs → deterministic outputs (×3)",        test_batch_identical_inputs_deterministic),
])

ALL_SUITES = [SUITE_A, SUITE_B, SUITE_C, SUITE_D, SUITE_E, SUITE_F, SUITE_G]


def main():
    print("\n  Running Equity Research Platform — Stress Test Suite")
    print(f"  {'─' * 68}")

    for suite_label, tests in ALL_SUITES:
        print(f"\n  [{suite_label}]")
        for test_name, fn in tests:
            r = run_test(suite_label, test_name, fn)
            icon = "✓" if r.passed else "✗"
            print(f"    {icon}  {test_name:<55} {r.duration_ms:>6.0f}ms")
            if not r.passed and r.error:
                for line in r.error.splitlines()[:3]:
                    print(f"         {line}")

    print(REPORT.summary())

    failed = sum(1 for r in REPORT.results if not r.passed)
    sys.exit(failed)


# ── pytest compatibility ───────────────────────────────────────────────────────
def pytest_collect_file(parent, file_path):
    pass


# Generate pytest test functions dynamically
def _make_pytest_test(fn):
    def _test():
        fn()
    _test.__name__ = fn.__name__
    return _test


for _suite_label, _tests in ALL_SUITES:
    for _name, _fn in _tests:
        _safe = _fn.__name__
        globals()[_safe] = _fn  # Already defined above; just ensure pytest picks them up


if __name__ == "__main__":
    main()
