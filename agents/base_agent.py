"""
Base Agent — abstract superclass enforcing the contract every agent must fulfill:
  Objective | Inputs | Outputs | Validation | Error Handling | Logging | Retry Logic
"""

from __future__ import annotations
import time
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Any
from loguru import logger

from ..models.research import (
    AgentOutput, AgentStatus, Finding, FindingType, RiskClassification, ResearchState
)
from ..core.llm_manager import LLMManager
from ..storage.storage_manager import StorageManager
from ..storage.audit_trail import AuditTrail
from ..storage.database import ResearchDatabase


class BaseAgent(ABC):
    """
    Every agent must:
      1. Define AGENT_ID and AGENT_NAME
      2. Implement run(state) -> AgentOutput
      3. Never return unvalidated data
      4. Never output free-form text between agents (only AgentOutput)
    """

    AGENT_ID: str = "base"
    AGENT_NAME: str = "Base Agent"

    def __init__(
        self,
        llm: LLMManager,
        storage: StorageManager,
        audit: AuditTrail,
        db: ResearchDatabase,
    ):
        self.llm = llm
        self.storage = storage
        self.audit = audit
        self.db = db
        self._logger = logger.bind(agent=self.AGENT_NAME)

    def execute(self, state: ResearchState) -> AgentOutput:
        """
        Public entry point — wraps run() with timing, logging, error handling, and retry.
        Never call run() directly from outside.
        """
        self._logger.info(f"Starting {self.AGENT_NAME}")
        self.audit.log_agent_start(self.AGENT_ID, self.AGENT_NAME)
        start_time = time.perf_counter()

        for attempt in range(self._max_retries()):
            try:
                output = self.run(state)
                output = self._validate_output(output)
                elapsed = round(time.perf_counter() - start_time, 2)
                try:
                    output.execution_time_seconds = elapsed
                except Exception:
                    pass

                self._persist(output, state)
                self.audit.log_agent_complete(
                    self.AGENT_ID, self.AGENT_NAME, output.risk_score, len(output.findings)
                )
                self._logger.info(
                    f"Completed {self.AGENT_NAME} in {elapsed:.1f}s | "
                    f"risk={output.risk_score:.0f} | findings={len(output.findings)}"
                )
                return output

            except Exception as e:
                tb = traceback.format_exc()
                self._logger.error(f"Attempt {attempt+1}/{self._max_retries()} failed: {e}")
                if attempt < self._max_retries() - 1:
                    time.sleep(2 ** attempt)
                    continue
                self.audit.log_agent_error(self.AGENT_ID, self.AGENT_NAME, str(e))
                return self._failure_output(str(e), tb)

    @abstractmethod
    def run(self, state: ResearchState) -> AgentOutput:
        """Core agent logic. Implement in every subclass."""
        ...

    def _validate_output(self, output: AgentOutput) -> AgentOutput:
        """Validate agent output is structurally complete. Raise ValueError if not."""
        if not output.agent_id:
            raise ValueError(f"{self.AGENT_NAME}: AgentOutput missing agent_id")
        if not output.agent_name:
            raise ValueError(f"{self.AGENT_NAME}: AgentOutput missing agent_name")
        if not 0 <= output.risk_score <= 100:
            raise ValueError(f"{self.AGENT_NAME}: risk_score {output.risk_score} out of range")
        for finding in output.findings:
            if not finding.title or not finding.agent_id:
                raise ValueError(f"{self.AGENT_NAME}: Finding missing required fields")
        return output

    def _persist(self, output: AgentOutput, state: ResearchState) -> None:
        try:
            self.storage.save_json(
                output.model_dump(mode="json"),
                f"agent_{self.AGENT_ID}_{self.AGENT_NAME.replace(' ', '_')}.json",
                "Agent_Outputs",
            )
            self.db.save_agent_output(state.run_id, output.model_dump(mode="json"))
            if output.findings:
                self.db.save_findings(state.run_id, [f.model_dump(mode="json") for f in output.findings])
        except Exception as e:
            self._logger.warning(f"Persist failed for {self.AGENT_NAME}: {e}")

    def _failure_output(self, error: str, traceback_str: str = "") -> AgentOutput:
        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.FAILED,
            error=error,
            summary=f"{self.AGENT_NAME} failed: {error[:200]}",
            risk_score=0.0,
        )

    def _max_retries(self) -> int:
        return 3

    # ── Helpers available to all agents ──────────────────────────

    def make_finding(
        self,
        finding_type: FindingType,
        title: str,
        detail: str,
        evidence: str,
        risk_level: RiskClassification = RiskClassification.MEDIUM,
        confidence: float = 0.75,
        fiscal_year: Optional[str] = None,
        source_url: Optional[str] = None,
        quantified_impact: Optional[float] = None,
    ) -> Finding:
        return Finding(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            finding_type=finding_type,
            title=title[:200],
            detail=detail,
            evidence=evidence,
            risk_level=risk_level,
            confidence=confidence,
            fiscal_year=fiscal_year,
            source_url=source_url,
            quantified_impact=quantified_impact,
        )

    def rag_query(self, question: str, state: ResearchState, top_k: int = 5) -> str:
        """
        Query the equity RAG pipeline for contextual information about the company.
        Uses the per-ticker vector store (filed documents + indexed transcripts) as
        the primary source, falling back to internet and tools as needed.

        Returns the answer string; logs a warning on failure.
        """
        ticker = (state.company_profile or {}).get("ticker", state.ticker) or "UNKNOWN"
        company = (state.company_profile or {}).get("name", state.company_name)
        try:
            from ..retrieval.rag_pipeline import query as _rag_q
            answer = _rag_q(question, company_name=company, ticker=ticker)
            self._logger.debug(f"rag_query answered ({len(answer)} chars): {question[:60]}")
            return answer
        except Exception as e:
            self._logger.warning(f"rag_query failed: {e}")
            return ""

    def red_flag(self, title: str, detail: str, evidence: str,
                 risk_level: RiskClassification = RiskClassification.HIGH,
                 **kwargs) -> Finding:
        return self.make_finding(FindingType.RED_FLAG, title, detail, evidence, risk_level, **kwargs)

    def green_flag(self, title: str, detail: str, evidence: str,
                   confidence: float = 0.8, **kwargs) -> Finding:
        return self.make_finding(FindingType.GREEN_FLAG, title, detail, evidence,
                                 RiskClassification.LOW, confidence, **kwargs)

    def risk_from_scores(self, scores: list[float]) -> float:
        if not scores:
            return 50.0
        return round(sum(scores) / len(scores), 1)

    def get_financial_series(self, state: ResearchState, field: str) -> dict[str, float]:
        history = state.financial_history or {}
        is_dict = isinstance(history, dict)
        result = {}
        for year, data in history.items():
            if is_dict:
                if isinstance(data, dict):
                    v = data.get(field)
                    if v is not None:
                        result[year] = float(v)
        return result

    def llm_analyze(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        return self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
