"""
CIO-Orchestrated Equity Research Workflow — 20-agent platform.

The orchestrator acts as Chief Investment Officer (CIO): it coordinates
specialist agents, validates evidence quality, resolves contradictions,
and generates the final investment recommendation.

Research sequence follows the mandatory 11-step philosophy:
  1 Macro → 2 Industry → 3 Business → 4 Management → 5 Financial →
  6 Risks → 7 Accounting → 8 Governance → 9 Forecast → 10 Valuation → 11 Thesis

Phase A: Company Profiling + Filing Retrieval (sequential)
Phase B: Financial Extraction + Market Data + Transcripts + Historical (parallel)
Phase C: Analysis agents — Accounting, Forensic, Risk, Earnings, Industry Intelligence,
         Management Governance, ESG Sustainability (parallel)
Phase D: Financial Modeling → Valuation → Scenario Analysis (sequential)
Phase E: Narrative Generation → Compliance & Standards Validation (sequential)
Phase F: Report Generation (20-section CIO report)
"""

from __future__ import annotations
import concurrent.futures
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Type
from loguru import logger

from ..models.research import ResearchState, AgentOutput, AgentStatus, RiskClassification
from ..core.config import LLM_CONFIG, OUTPUT_DIR
from ..core.llm_manager import LLMManager
from ..storage.storage_manager import StorageManager
from ..storage.audit_trail import AuditTrail
from ..storage.database import ResearchDatabase

from ..agents.base_agent import BaseAgent
from ..agents.company_profiling import CompanyProfilingAgent
from ..agents.filing_retrieval import FilingRetrievalAgent
from ..agents.financial_extraction import FinancialExtractionAgent
from ..agents.market_data import MarketDataAgent
from ..agents.accounting_quality import AccountingQualityAgent
from ..agents.forensic_accounting import ForensicAccountingAgent
from ..agents.financial_modeling_agent import FinancialModelingAgent
from ..agents.valuation_agent import ValuationAgent
from ..agents.risk_analysis import RiskAnalysisAgent
from ..agents.narrative_agent import NarrativeGenerationAgent
from ..agents.compliance_agent import ComplianceValidationAgent
from ..agents.transcript_retrieval import TranscriptRetrievalAgent
from ..agents.historical_data import HistoricalDataAgent
from ..agents.earnings_quality import EarningsQualityAgent
from ..agents.scenario_analysis import ScenarioAnalysisAgent
from ..agents.report_generation import ReportGenerationAgent
from ..agents.industry_intelligence import IndustryIntelligenceAgent
from ..agents.management_governance import ManagementGovernanceAgent
from ..agents.esg_sustainability import ESGSustainabilityAgent


def _supports_parallel(llm: LLMManager) -> bool:
    return llm.backend in ("groq", "openai", "anthropic", "gemini", "together", "openrouter")


class ResearchOrchestrator:
    """
    Production orchestrator for the equity research platform.
    Manages: company profiling → data retrieval → processing → analysis → modeling →
             valuation → narrative → compliance → report generation.
    """

    def __init__(self):
        logger.info("Initialising Equity Research Platform...")
        self.llm = LLMManager()
        self.db = ResearchDatabase()
        logger.info(f"LLM: {self.llm.get_backend_info()}")

    def research(
        self,
        company_name: str,
        ticker: str = "",
        output_dir: Optional[Path] = None,
    ) -> dict:
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        start = time.perf_counter()
        logger.info(f"\n{'='*60}\nResearch run started: {company_name} | run_id={run_id}\n{'='*60}")

        # ── Setup infrastructure ──────────────────────────────────
        storage = StorageManager(company_name, run_id, base_dir=output_dir or OUTPUT_DIR)
        audit = AuditTrail(storage.base_path, company_name, run_id)
        self.db.start_run(run_id, 0, str(storage.base_path))

        state = ResearchState(
            run_id=run_id,
            company_name=company_name,
            ticker=ticker.upper() if ticker else company_name[:6].upper(),
            started_at=datetime.now(timezone.utc).isoformat(),
            storage_path=str(storage.base_path),
        )

        def make(cls: Type[BaseAgent]) -> BaseAgent:
            return cls(llm=self.llm, storage=storage, audit=audit, db=self.db)

        try:
            # ── Phase A: Company Profiling ────────────────────────
            logger.info("Phase A: Company Profiling")
            profiling_agent = make(CompanyProfilingAgent)
            profiling_out = profiling_agent.execute(state)
            state.add_agent_output(profiling_out)
            if profiling_out.status == AgentStatus.COMPLETED:
                state.company_profile = profiling_out.payload.get("company_profile", {})
                state.ticker = state.company_profile.get("ticker", state.ticker)
                logger.info(f"  Profile: {state.company_profile.get('name')} ({state.ticker})")

            # ── Phase A2: Filing Retrieval ────────────────────────
            logger.info("Phase A2: Filing Retrieval")
            filing_out = make(FilingRetrievalAgent).execute(state)
            state.add_agent_output(filing_out)

            # ── Phase B: All data gathering (parallel) ────────────
            logger.info("Phase B: Financial Extraction + Market Data + Transcripts + Historical")
            phase_b_agents = [
                (FinancialExtractionAgent, "03_financial_extraction"),
                (MarketDataAgent, "04_market_data"),
                (TranscriptRetrievalAgent, "12_transcript_retrieval"),
                (HistoricalDataAgent, "13_historical_data"),
            ]
            phase_b_results = self._run_parallel(phase_b_agents, state, storage, audit)
            for output in phase_b_results:
                state.add_agent_output(output)
                if output.agent_id == "03_financial_extraction":
                    state.financial_history = output.payload.get("financial_history")

            # ── Phase C: Analysis agents (parallel) ───────────────
            # Sequence steps 2, 4, 6, 7, 8, 9 run in parallel after data is gathered.
            # Industry Intelligence (step 2), Management Governance (step 4), and
            # ESG Sustainability (step 8) are the new CIO framework agents.
            logger.info(
                "Phase C: Accounting Quality + Forensic + Risk + Earnings Quality + "
                "Industry Intelligence + Management Governance + ESG Sustainability"
            )
            phase_c_agents = [
                (AccountingQualityAgent,       "05_accounting_quality"),
                (ForensicAccountingAgent,      "06_forensic_accounting"),
                (RiskAnalysisAgent,            "09_risk_analysis"),
                (EarningsQualityAgent,         "14_earnings_quality"),
                (IndustryIntelligenceAgent,    "17_industry_intelligence"),
                (ManagementGovernanceAgent,    "18_management_governance"),
                (ESGSustainabilityAgent,       "19_esg_sustainability"),
            ]
            phase_c_results = self._run_parallel(phase_c_agents, state, storage, audit)
            for output in phase_c_results:
                state.add_agent_output(output)

            # ── Phase D: Modeling → Valuation → Scenarios ────────
            logger.info("Phase D: Financial Modeling")
            fm_out = make(FinancialModelingAgent).execute(state)
            state.add_agent_output(fm_out)

            logger.info("Phase D2: Valuation")
            val_out = make(ValuationAgent).execute(state)
            state.add_agent_output(val_out)
            if val_out.payload.get("valuation_summary"):
                state.valuation_summary = val_out.payload["valuation_summary"]
                vs = val_out.payload["valuation_summary"]
                state.target_price = vs.get("base_price")
                state.investment_rating = self._derive_rating(vs)

            logger.info("Phase D3: Scenario Analysis")
            scen_out = make(ScenarioAnalysisAgent).execute(state)
            state.add_agent_output(scen_out)

            # ── Phase E: Narrative + Compliance ───────────────────
            logger.info("Phase E: Narrative Generation")
            narr_out = make(NarrativeGenerationAgent).execute(state)
            state.add_agent_output(narr_out)
            if narr_out.payload.get("sections"):
                state.report_sections.update(narr_out.payload["sections"])

            logger.info("Phase E2: Compliance Validation")
            comp_out = make(ComplianceValidationAgent).execute(state)
            state.add_agent_output(comp_out)
            state.validation_results.extend(comp_out.payload.get("validation_results_obj", []))
            state.validation_failed = not comp_out.payload.get("validation_passed", True)

            # ── Phase F: Report Generation (dedicated agent) ──────
            logger.info("Phase F: Report Generation")
            report_agent_out = make(ReportGenerationAgent).execute(state)
            state.add_agent_output(report_agent_out)
            state.report_path = report_agent_out.payload.get("report_path")

            # ── Overall risk score ────────────────────────────────
            state.overall_risk_score = self._compute_overall_risk(state)
            state.overall_risk_classification = self._classify_risk(state.overall_risk_score)

            # ── Persist final state ───────────────────────────────
            elapsed = round(time.perf_counter() - start, 1)
            self.db.complete_run(
                run_id=run_id,
                risk_score=state.overall_risk_score,
                rating=state.investment_rating,
                target_price=state.target_price,
                report_path=state.report_path or "",
            )
            audit_summary = audit.export_summary()
            storage.save_json(audit_summary, "audit_summary.json", "Audit_Trail")
            storage.save_json(self._build_result(state, elapsed), "research_result.json")

            logger.info(
                f"\n{'='*60}\nResearch Complete: {company_name}\n"
                f"Duration: {elapsed:.0f}s | Risk: {state.overall_risk_score:.0f}/100 | "
                f"Rating: {state.investment_rating}\n{'='*60}\n"
            )
            return self._build_result(state, elapsed)

        except Exception as e:
            logger.error(f"Research run failed: {e}", exc_info=True)
            audit.log("orchestrator", "ResearchOrchestrator", "FATAL_ERROR", str(e), severity="ERROR")
            return {"company_name": company_name, "run_id": run_id, "error": str(e), "status": "FAILED"}

    # ── Parallel runner ───────────────────────────────────────────

    def _run_parallel(
        self,
        agent_specs: list[tuple[Type[BaseAgent], str]],
        state: ResearchState,
        storage: StorageManager,
        audit: AuditTrail,
    ) -> list[AgentOutput]:
        results: list[AgentOutput] = []
        if _supports_parallel(self.llm) and len(agent_specs) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(agent_specs), 6)) as pool:
                futures = {
                    pool.submit(
                        self._run_single, cls, state, storage, audit
                    ): name for cls, name in agent_specs
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        name = futures[future]
                        logger.error(f"Agent {name} thread failed: {e}")
                        results.append(AgentOutput(
                            agent_id=name, agent_name=name,
                            status=AgentStatus.FAILED, error=str(e),
                        ))
        else:
            for cls, name in agent_specs:
                results.append(self._run_single(cls, state, storage, audit))
        return results

    def _run_single(
        self, cls: Type[BaseAgent], state: ResearchState,
        storage: StorageManager, audit: AuditTrail
    ) -> AgentOutput:
        agent = cls(llm=self.llm, storage=storage, audit=audit, db=self.db)
        return agent.execute(state)

    # ── Risk scoring ──────────────────────────────────────────────

    def _compute_overall_risk(self, state: ResearchState) -> float:
        # CIO risk weight matrix — aligned with CFA Institute, OECD Corporate Governance
        # Principles, and MSCI Quality Factor research.
        #
        # Key design principles:
        #   1. Governance is the single strongest predictor of long-term risk-adjusted return
        #      (OECD 2023, Fama-French quality factor, Buffett's "first filter").
        #   2. Industry structure (Porter + TAM) determines the durability of cash flows;
        #      systematically underweighted in traditional models.
        #   3. Forensic signals remain high — the best early-warning for catastrophic downside.
        #   4. Financial modeling and data extraction are analyst outputs / process checks,
        #      not independent risk signals; they receive minimal weight.
        #   5. ESG raised to 7%: ISSB S1+S2 mandate makes climate/social risk financially
        #      material and auditable from FY2025 onwards.
        weights = {
            "18_management_governance": 0.20,  # governance quality: board independence,
                                               # capital allocation, RPT exposure, pledging
            "06_forensic_accounting":   0.15,  # Beneish/Altman/Piotroski: highest pure
                                               # downside signal; fraud/manipulation early-warning
            "09_risk_analysis":         0.13,  # macro / credit / operational risk matrix
            "17_industry_intelligence": 0.12,  # Porter Five Forces, TAM, moat durability;
                                               # structural driver of long-run cash flows
            "05_accounting_quality":    0.10,  # accrual ratio, revenue recognition quality;
                                               # MSCI quality factor component
            "14_earnings_quality":      0.08,  # guidance accuracy, beat/miss patterns;
                                               # earnings predictability premium
            "08_valuation":             0.08,  # valuation risk (paying too much);
                                               # secondary to business quality signals
            "19_esg_sustainability":    0.07,  # ISSB S1+S2 / BRSR: material for energy,
                                               # materials, utilities, and large-cap India cos
            "03_financial_extraction":  0.04,  # data quality / completeness signal;
                                               # not a fundamental risk driver
            "11_compliance":            0.02,  # regulatory risk: largely binary pass/fail;
                                               # captured upstream by governance weight
            "07_financial_modeling":    0.01,  # 5-yr model accuracy: analyst output,
                                               # not an independent risk signal
        }
        score = 0.0
        total_weight = 0.0
        for agent_id, weight in weights.items():
            output = state.agent_outputs.get(agent_id)
            if output and output.status == AgentStatus.COMPLETED:
                score += output.risk_score * weight
                total_weight += weight
        return round(score / total_weight, 1) if total_weight > 0 else 50.0

    def _classify_risk(self, score: float) -> RiskClassification:
        if score >= 75: return RiskClassification.CRITICAL
        if score >= 55: return RiskClassification.HIGH
        if score >= 35: return RiskClassification.MEDIUM
        return RiskClassification.LOW

    def _derive_rating(self, valuation_summary: dict) -> str:
        upside = valuation_summary.get("upside_pct", 0.0) or 0.0
        if upside > 25: return "BUY"
        if upside > 10: return "OUTPERFORM"
        if upside > -10: return "HOLD"
        if upside > -25: return "UNDERPERFORM"
        return "SELL"

    def _build_result(self, state: ResearchState, elapsed: float) -> dict:
        val = state.valuation_summary or {}
        return {
            "run_id": state.run_id,
            "company_name": state.company_name,
            "ticker": state.ticker,
            "started_at": state.started_at,
            "duration_seconds": elapsed,
            "company_profile": state.company_profile,
            "overall_risk_score": state.overall_risk_score,
            "overall_risk_classification": state.overall_risk_classification.value,
            "investment_rating": state.investment_rating,
            "target_price": state.target_price,
            "current_price": (state.agent_outputs.get("04_market_data") or AgentOutput(
                agent_id="x", agent_name="x")).payload.get("market_data", {}).get("current_price"),
            "upside_pct": val.get("upside_pct"),
            "agents_completed": len(state.completed_agents),
            "agents_failed": len(state.failed_agents),
            "total_findings": len(state.all_findings),
            "critical_findings": state.critical_findings_count(),
            "validation_passed": not state.validation_failed,
            "report_path": state.report_path,
            "storage_path": str(state.storage_path or ""),
            "agent_risk_scores": {
                aid: o.risk_score for aid, o in state.agent_outputs.items()
            },
        }
