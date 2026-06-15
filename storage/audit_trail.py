"""
Audit Trail — immutable, append-only log of every decision and data point.
Every agent writes here so the full research chain is reproducible.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from loguru import logger


class AuditTrail:
    def __init__(self, storage_path: Path, company_name: str, run_id: str):
        self.company_name = company_name
        self.run_id = run_id
        self.log_path = storage_path / "Audit_Trail" / "audit_log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []
        self._write_event("AUDIT_INIT", "AuditTrail", "run_started",
                          f"Investigation started for {company_name}", run_id=run_id)

    def log(
        self,
        agent_id: str,
        agent_name: str,
        event_type: str,
        finding: str,
        source_url: Optional[str] = None,
        data: Optional[dict] = None,
        severity: str = "INFO",
    ) -> None:
        self._write_event(event_type, agent_name, finding, finding,
                          agent_id=agent_id, source_url=source_url,
                          data=data, severity=severity)

    def log_data_point(
        self,
        agent_id: str,
        field_name: str,
        value: Any,
        source_url: str,
        source_document: str,
        fiscal_year: Optional[str] = None,
        confidence: float = 1.0,
    ) -> None:
        entry = {
            "timestamp": self._now(),
            "run_id": self.run_id,
            "event_type": "DATA_POINT",
            "agent_id": agent_id,
            "field_name": field_name,
            "value": value,
            "source_url": source_url,
            "source_document": source_document,
            "fiscal_year": fiscal_year,
            "confidence": confidence,
        }
        self._append(entry)

    def log_validation_failure(
        self, check_name: str, description: str, severity: str, corrective_action: str = ""
    ) -> None:
        self._write_event(
            "VALIDATION_FAILURE", "ValidationEngine", check_name,
            description, severity=severity, data={"corrective_action": corrective_action}
        )

    def log_agent_start(self, agent_id: str, agent_name: str) -> None:
        self._write_event("AGENT_START", agent_name, "started", f"{agent_name} started", agent_id=agent_id)

    def log_agent_complete(self, agent_id: str, agent_name: str, risk_score: float, finding_count: int) -> None:
        self._write_event(
            "AGENT_COMPLETE", agent_name, "completed",
            f"{agent_name} completed: risk={risk_score:.1f}, findings={finding_count}",
            agent_id=agent_id,
            data={"risk_score": risk_score, "finding_count": finding_count},
        )

    def log_agent_error(self, agent_id: str, agent_name: str, error: str) -> None:
        self._write_event("AGENT_ERROR", agent_name, "error", error, agent_id=agent_id, severity="ERROR")

    def export_summary(self) -> dict:
        entries = self._read_all()
        return {
            "run_id": self.run_id,
            "company": self.company_name,
            "total_events": len(entries),
            "agents": list({e.get("agent_id", "") for e in entries if e.get("agent_id")}),
            "data_points": sum(1 for e in entries if e.get("event_type") == "DATA_POINT"),
            "validation_failures": sum(1 for e in entries if e.get("event_type") == "VALIDATION_FAILURE"),
            "errors": sum(1 for e in entries if e.get("severity") == "ERROR"),
            "sources": list({e.get("source_url", "") for e in entries if e.get("source_url")}),
            "log_path": str(self.log_path),
        }

    # ── Internal ──────────────────────────────────────────────────

    def _write_event(
        self, event_type: str, agent_name: str, finding_type: str, detail: str,
        agent_id: str = "", source_url: str = "", data: dict = None, severity: str = "INFO", run_id: str = None,
    ) -> None:
        entry = {
            "timestamp": self._now(),
            "run_id": run_id or self.run_id,
            "event_type": event_type,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "finding_type": finding_type,
            "detail": detail[:1000],
            "source_url": source_url or "",
            "severity": severity,
            "data": data or {},
        }
        self._append(entry)
        if severity == "ERROR":
            logger.error(f"[Audit] {agent_name}: {detail[:200]}")

    def _append(self, entry: dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")

    def _read_all(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
