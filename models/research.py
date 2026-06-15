"""
Research workflow state and agent output schemas.
Every agent returns an AgentOutput; the orchestrator accumulates them into ResearchState.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field


class RiskClassification(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AgentStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class FindingType(str, Enum):
    RED_FLAG = "RED_FLAG"
    GREEN_FLAG = "GREEN_FLAG"
    NEUTRAL = "NEUTRAL"
    WARNING = "WARNING"


class Finding(BaseModel):
    agent_id: str
    agent_name: str
    finding_type: FindingType
    title: str = Field(..., max_length=200)
    detail: str
    evidence: str
    source_url: Optional[str] = None
    fiscal_year: Optional[str] = None
    risk_level: RiskClassification = RiskClassification.MEDIUM
    confidence: float = Field(..., ge=0.0, le=1.0)
    quantified_impact: Optional[float] = None
    model_config = {"frozen": True}


class ValidationResult(BaseModel):
    check_name: str
    passed: bool
    severity: RiskClassification
    description: str
    corrective_action: Optional[str] = None
    auto_corrected: bool = False
    model_config = {"frozen": True}


class ComplianceCheck(BaseModel):
    regulation: str
    jurisdiction: str
    check_description: str
    status: str  # PASS / FAIL / WARN / NA
    evidence: str
    source: Optional[str] = None
    model_config = {"frozen": True}


class AgentOutput(BaseModel):
    """
    Structured output from any single agent.
    No free-form text is exchanged between agents — only this schema.
    """
    agent_id: str
    agent_name: str
    status: AgentStatus = AgentStatus.COMPLETED
    error: Optional[str] = None

    # Core outputs
    summary: str = ""
    findings: list[Finding] = Field(default_factory=list)
    risk_score: float = Field(0.0, ge=0.0, le=100.0)
    risk_classification: RiskClassification = RiskClassification.LOW
    confidence: float = Field(1.0, ge=0.0, le=1.0)

    # Typed payloads — each agent populates the relevant one
    payload: dict[str, Any] = Field(default_factory=dict)

    # Audit
    sources_used: list[str] = Field(default_factory=list)
    execution_time_seconds: float = 0.0
    tokens_used: int = 0
    retry_count: int = 0

    @property
    def red_flags(self) -> list[Finding]:
        return [f for f in self.findings if f.finding_type == FindingType.RED_FLAG]

    @property
    def green_flags(self) -> list[Finding]:
        return [f for f in self.findings if f.finding_type == FindingType.GREEN_FLAG]

    def to_context_string(self, max_chars: int = 800) -> str:
        lines = [
            f"[{self.agent_name}] Risk={self.risk_score:.0f}/100 ({self.risk_classification.value})",
            f"Summary: {self.summary[:400]}",
        ]
        for f in self.red_flags[:3]:
            lines.append(f"  RED FLAG [{f.risk_level.value}]: {f.title}")
        return "\n".join(lines)[:max_chars]


class ForensicScores(BaseModel):
    beneish_m_score: Optional[float] = None
    beneish_classification: Optional[str] = None
    piotroski_f_score: Optional[int] = None
    piotroski_classification: Optional[str] = None
    altman_z_score: Optional[float] = None
    altman_classification: Optional[str] = None
    accrual_ratio: Optional[float] = None
    sloan_accrual: Optional[float] = None
    overall_forensic_risk: RiskClassification = RiskClassification.MEDIUM
    component_details: dict[str, Any] = Field(default_factory=dict)


class ResearchState(BaseModel):
    """
    Shared LangGraph state — accumulated across all 17 agents.
    Immutable fields (company_profile) set at start; mutable fields updated per agent.
    """
    # Identity (set once)
    run_id: str
    company_name: str
    ticker: str
    started_at: str

    # Agent outputs (populated as each agent completes)
    agent_outputs: dict[str, AgentOutput] = Field(default_factory=dict)

    # Typed research artifacts
    company_profile: Optional[dict] = None          # CompanyProfile serialised
    financial_history: Optional[dict] = None         # FinancialHistory serialised
    financial_forecast: Optional[dict] = None
    valuation_summary: Optional[dict] = None
    forensic_scores: Optional[ForensicScores] = None
    sector_kpis: Optional[dict] = None

    # Consolidated findings across all agents
    all_findings: list[Finding] = Field(default_factory=list)
    validation_results: list[ValidationResult] = Field(default_factory=list)
    compliance_checks: list[ComplianceCheck] = Field(default_factory=list)

    # Report artifacts
    report_sections: dict[str, str] = Field(default_factory=dict)
    report_path: Optional[str] = None

    # Workflow control
    current_agent: Optional[str] = None
    completed_agents: list[str] = Field(default_factory=list)
    failed_agents: list[str] = Field(default_factory=list)
    validation_failed: bool = False
    requires_correction: bool = False
    corrections_applied: int = 0

    # Overall risk
    overall_risk_score: float = 0.0
    overall_risk_classification: RiskClassification = RiskClassification.LOW
    investment_rating: str = ""
    target_price: Optional[float] = None

    # Audit
    audit_log: list[dict] = Field(default_factory=list)
    storage_path: Optional[str] = None

    def add_agent_output(self, output: AgentOutput) -> None:
        self.agent_outputs[output.agent_id] = output
        if output.status == AgentStatus.FAILED:
            if output.agent_id not in self.failed_agents:
                self.failed_agents.append(output.agent_id)
        else:
            if output.agent_id not in self.completed_agents:
                self.completed_agents.append(output.agent_id)
        for finding in output.findings:
            self.all_findings.append(finding)

    def get_inter_agent_context(self, exclude_agent: str = "", max_chars: int = 3000) -> str:
        parts = []
        for aid, output in self.agent_outputs.items():
            if aid == exclude_agent:
                continue
            parts.append(output.to_context_string())
        return "\n\n".join(parts)[:max_chars]

    def critical_findings_count(self) -> int:
        return sum(1 for f in self.all_findings if f.risk_level == RiskClassification.CRITICAL)
