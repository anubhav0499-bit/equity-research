"""
Agent 17 — Report Generation Agent
Dedicated agent that assembles and generates the full institutional DOCX research report.
Separates report generation from the orchestrator and gives it proper agent contract.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
from loguru import logger

from .base_agent import BaseAgent
from ..models.research import AgentOutput, AgentStatus, RiskClassification, FindingType
from ..orchestrator.state import ResearchState


REPORT_QC_SYSTEM = """You are an equity research publication editor performing final quality control.
Review the research report metadata provided and:
1. Confirm all mandatory sections are present and non-empty
2. Flag any inconsistencies between the investment rating and supporting analysis
3. Verify the target price is supported by valuation methodology described
4. Ensure risk factors are consistent with identified red flags
5. Confirm the investment thesis is differentiated and not generic

Respond with: QC_PASS or QC_FAIL, then a bullet list of specific issues (if any)."""


class ReportGenerationAgent(BaseAgent):
    AGENT_ID = "17_report_generation"
    AGENT_NAME = "Report Generation Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        profile = state.company_profile or {}
        ticker = profile.get("ticker", state.ticker)
        company_name = profile.get("name", state.company_name)

        findings = []
        details: dict = {
            "report_path": None,
            "sections_generated": [],
            "sections_missing": [],
            "page_count_estimate": 0,
            "word_count_estimate": 0,
            "qc_status": "PENDING",
            "qc_issues": [],
        }

        # ── Pre-generation validation ──────────────────────────────
        required_agents = [
            "01_company_profiling",
            "03_financial_extraction",
            "06_forensic_accounting",
            "07_financial_modeling",
            "08_valuation",
            "10_narrative",
        ]
        missing_agents = [a for a in required_agents if a not in state.agent_outputs]
        if missing_agents:
            findings.append(self.make_finding(
                FindingType.WARNING,
                f"Report generated with missing upstream agents: {missing_agents}",
                "Some report sections may be incomplete due to missing agent outputs.",
                f"Missing: {missing_agents}",
                risk_level=RiskClassification.MEDIUM, confidence=0.95,
            ))

        # ── Generate DOCX report ───────────────────────────────────
        report_path = self._generate_docx(state)
        details["report_path"] = str(report_path) if report_path else None

        if not report_path:
            return AgentOutput(
                agent_id=self.AGENT_ID,
                agent_name=self.AGENT_NAME,
                status=AgentStatus.FAILED,
                summary=f"DOCX report generation failed for {ticker}.",
                findings=[self.red_flag(
                    "Report generation failed",
                    "DOCX generator raised an error. Check logs for details.",
                    "generate_docx_report() returned None",
                    risk_level=RiskClassification.HIGH, confidence=0.99,
                )],
                risk_score=0.0,
                payload=details,
                sources_used=[],
            )

        # ── Post-generation QC ────────────────────────────────────
        section_check = self._check_sections(state)
        details["sections_generated"] = section_check["present"]
        details["sections_missing"] = section_check["missing"]

        # ── Estimate report size ───────────────────────────────────
        try:
            from docx import Document
            doc = Document(str(report_path))
            word_count = sum(len(para.text.split()) for para in doc.paragraphs)
            details["word_count_estimate"] = word_count
            details["page_count_estimate"] = max(1, word_count // 300)
        except Exception as e:
            logger.debug(f"Word count estimation failed: {e}")

        # ── LLM QC check ──────────────────────────────────────────
        qc_prompt = self._build_qc_prompt(state, details)
        qc_response = self.llm_analyze(REPORT_QC_SYSTEM, qc_prompt, max_tokens=600)
        details["qc_llm_response"] = qc_response

        qc_passed = "QC_PASS" in qc_response.upper()
        details["qc_status"] = "PASS" if qc_passed else "FAIL"
        if not qc_passed:
            details["qc_issues"] = self._parse_qc_issues(qc_response)
            findings.append(self.make_finding(
                FindingType.WARNING,
                "Report QC identified issues",
                "Post-generation quality check flagged potential inconsistencies.",
                f"QC issues: {details['qc_issues'][:3]}",
                risk_level=RiskClassification.LOW, confidence=0.65,
            ))
        else:
            findings.append(self.green_flag(
                "Report passed automated quality control",
                "LLM QC reviewer confirmed all mandatory sections present and consistent.",
                f"Sections: {len(details['sections_generated'])} | Words: {details['word_count_estimate']}",
                confidence=0.70,
            ))

        # ── Missing section findings ───────────────────────────────
        if details["sections_missing"]:
            findings.append(self.make_finding(
                FindingType.WARNING,
                f"Missing report sections: {details['sections_missing']}",
                "Incomplete report sections reduce institutional quality standards.",
                f"Present: {details['sections_generated']} | Missing: {details['sections_missing']}",
                risk_level=RiskClassification.MEDIUM, confidence=0.90,
            ))

        self.storage.save_json(details, "report_metadata.json", "Agent_Outputs")
        logger.info(f"Report generated: {report_path} | Words: {details['word_count_estimate']} | QC: {details['qc_status']}")

        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=(
                f"Institutional research report generated for {company_name} ({ticker}). "
                f"Report path: {report_path.name if report_path else 'N/A'}. "
                f"Estimated {details['word_count_estimate']} words across "
                f"{details['page_count_estimate']} pages. "
                f"QC: {details['qc_status']}. "
                f"Sections present: {len(details['sections_generated'])}."
            ),
            findings=findings,
            risk_score=10.0,
            risk_classification=RiskClassification.LOW,
            payload=details,
            sources_used=["all_prior_agents", "docx_generator"],
        )

    # ── Helpers ───────────────────────────────────────────────────

    def _generate_docx(self, state: ResearchState) -> Optional[Path]:
        try:
            from ..reporting.docx_generator import generate_docx_report
            path = generate_docx_report(state, self.storage)
            return path
        except Exception as e:
            logger.error(f"ReportGenerationAgent DOCX error: {e}", exc_info=True)
            return None

    def _check_sections(self, state: ResearchState) -> dict:
        mandatory_sections = [
            "executive_summary",
            "investment_thesis",
            "accounting_quality",
            "risk_analysis",
            "scenario_analysis",
        ]
        present = []
        missing = []
        report_sections = state.report_sections or {}
        for section in mandatory_sections:
            if section in report_sections and report_sections[section]:
                present.append(section)
            else:
                missing.append(section)
        return {"present": present, "missing": missing}

    def _build_qc_prompt(self, state: ResearchState, details: dict) -> str:
        val_summary = state.valuation_summary or {}
        return (
            f"Company: {state.company_name} ({state.ticker})\n"
            f"Investment Rating: {state.investment_rating}\n"
            f"Target Price: ${state.target_price}\n"
            f"Current Price: {val_summary.get('current_price', 'N/A')}\n"
            f"Upside: {val_summary.get('upside_pct', 'N/A')}%\n"
            f"Overall Risk Score: {state.overall_risk_score}/100\n"
            f"Critical Findings: {state.critical_findings_count()}\n"
            f"Sections Present: {details['sections_generated']}\n"
            f"Sections Missing: {details['sections_missing']}\n"
            f"Word Count: {details['word_count_estimate']}\n"
            f"Agents Completed: {len(state.completed_agents)}\n"
            f"Validation Passed: {not state.validation_failed}\n\n"
            "Perform QC on this research report. Respond with QC_PASS or QC_FAIL followed by issues."
        )

    def _parse_qc_issues(self, response: str) -> list[str]:
        issues = []
        for line in response.split("\n"):
            stripped = line.strip().lstrip("-•*123456789. ").strip()
            if stripped and len(stripped) > 10 and "QC_" not in stripped:
                issues.append(stripped)
        return issues[:5]
