"""
SQLite + DuckDB storage for structured research data.
SQLite for transactional record-keeping; DuckDB for analytical queries.
"""

from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from loguru import logger

from ..core.config import DB_CONFIG


class ResearchDatabase:
    def __init__(self, db_path: Path = DB_CONFIG.sqlite_path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS companies (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT NOT NULL,
                    ticker      TEXT,
                    isin        TEXT,
                    exchange    TEXT,
                    sector      TEXT,
                    industry    TEXT,
                    country     TEXT,
                    currency    TEXT,
                    market_cap  REAL,
                    created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, exchange)
                );

                CREATE TABLE IF NOT EXISTS research_runs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id          TEXT UNIQUE NOT NULL,
                    company_id      INTEGER REFERENCES companies(id),
                    started_at      TEXT NOT NULL,
                    completed_at    TEXT,
                    status          TEXT DEFAULT 'running',
                    risk_score      REAL,
                    rating          TEXT,
                    target_price    REAL,
                    report_path     TEXT,
                    storage_path    TEXT,
                    error           TEXT
                );

                CREATE TABLE IF NOT EXISTS financial_data (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_id      INTEGER REFERENCES companies(id),
                    run_id          TEXT,
                    fiscal_year     TEXT NOT NULL,
                    statement_type  TEXT NOT NULL,
                    data_json       TEXT NOT NULL,
                    quality         TEXT DEFAULT 'UNKNOWN',
                    source_url      TEXT,
                    extracted_at    TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(company_id, fiscal_year, statement_type)
                );

                CREATE TABLE IF NOT EXISTS agent_outputs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id          TEXT NOT NULL,
                    agent_id        TEXT NOT NULL,
                    agent_name      TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    risk_score      REAL,
                    risk_class      TEXT,
                    summary         TEXT,
                    payload_json    TEXT,
                    findings_count  INTEGER DEFAULT 0,
                    red_flags_count INTEGER DEFAULT 0,
                    exec_time_sec   REAL,
                    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(run_id, agent_id)
                );

                CREATE TABLE IF NOT EXISTS findings (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id          TEXT NOT NULL,
                    agent_id        TEXT NOT NULL,
                    finding_type    TEXT NOT NULL,
                    title           TEXT NOT NULL,
                    detail          TEXT,
                    evidence        TEXT,
                    risk_level      TEXT,
                    confidence      REAL,
                    fiscal_year     TEXT,
                    source_url      TEXT,
                    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS valuation_outputs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id          TEXT NOT NULL,
                    scenario        TEXT NOT NULL,
                    dcf_value       REAL,
                    relative_value  REAL,
                    sotp_value      REAL,
                    blended_value   REAL,
                    wacc            REAL,
                    terminal_growth REAL,
                    current_price   REAL,
                    upside_pct      REAL,
                    payload_json    TEXT,
                    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(run_id, scenario)
                );

                CREATE INDEX IF NOT EXISTS idx_runs_company ON research_runs(company_id);
                CREATE INDEX IF NOT EXISTS idx_findings_run ON findings(run_id);
                CREATE INDEX IF NOT EXISTS idx_financial_company ON financial_data(company_id, fiscal_year);
            """)

    def upsert_company(self, data: dict) -> int:
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO companies (name, ticker, isin, exchange, sector, industry, country, currency, market_cap, updated_at)
                VALUES (:name, :ticker, :isin, :exchange, :sector, :industry, :country, :currency, :market_cap, :now)
                ON CONFLICT(ticker, exchange) DO UPDATE SET
                    name=excluded.name, sector=excluded.sector, updated_at=excluded.updated_at
            """, {**data, "now": self._now()})
            row = conn.execute("SELECT id FROM companies WHERE ticker=? AND exchange=?",
                               (data.get("ticker", ""), data.get("exchange", ""))).fetchone()
            return row["id"] if row else 0

    def start_run(self, run_id: str, company_id: int, storage_path: str) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO research_runs (run_id, company_id, started_at, storage_path)
                VALUES (?, ?, ?, ?)
            """, (run_id, company_id, self._now(), storage_path))

    def complete_run(
        self, run_id: str, risk_score: float, rating: str,
        target_price: Optional[float], report_path: str
    ) -> None:
        with self._conn() as conn:
            conn.execute("""
                UPDATE research_runs SET
                    completed_at=?, status='completed', risk_score=?, rating=?,
                    target_price=?, report_path=?
                WHERE run_id=?
            """, (self._now(), risk_score, rating, target_price, report_path, run_id))

    def save_financial_data(
        self, company_id: int, run_id: str, fiscal_year: str,
        statement_type: str, data: dict, quality: str = "UNKNOWN", source_url: str = ""
    ) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO financial_data (company_id, run_id, fiscal_year, statement_type, data_json, quality, source_url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(company_id, fiscal_year, statement_type) DO UPDATE SET
                    data_json=excluded.data_json, quality=excluded.quality
            """, (company_id, run_id, fiscal_year, statement_type,
                  json.dumps(data, default=str), quality, source_url))

    def load_financial_history(self, company_id: int, years: int = 7) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT fiscal_year, statement_type, data_json, quality, source_url
                FROM financial_data WHERE company_id=?
                ORDER BY fiscal_year DESC LIMIT ?
            """, (company_id, years * 3)).fetchall()
            return [dict(r) for r in rows]

    def save_agent_output(self, run_id: str, output_data: dict) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO agent_outputs
                    (run_id, agent_id, agent_name, status, risk_score, risk_class,
                     summary, payload_json, findings_count, red_flags_count, exec_time_sec)
                VALUES (:run_id, :agent_id, :agent_name, :status, :risk_score, :risk_class,
                        :summary, :payload_json, :findings_count, :red_flags_count, :exec_time_sec)
                ON CONFLICT(run_id, agent_id) DO UPDATE SET
                    status=excluded.status, risk_score=excluded.risk_score,
                    summary=excluded.summary, payload_json=excluded.payload_json
            """, {
                "run_id": run_id,
                "agent_id": output_data.get("agent_id", ""),
                "agent_name": output_data.get("agent_name", ""),
                "status": output_data.get("status", ""),
                "risk_score": output_data.get("risk_score", 0),
                "risk_class": output_data.get("risk_classification", ""),
                "summary": output_data.get("summary", "")[:2000],
                "payload_json": json.dumps(output_data.get("payload", {}), default=str)[:50000],
                "findings_count": len(output_data.get("findings", [])),
                "red_flags_count": sum(1 for f in output_data.get("findings", [])
                                       if f.get("finding_type") == "RED_FLAG"),
                "exec_time_sec": output_data.get("execution_time_seconds", 0),
            })

    def save_findings(self, run_id: str, findings: list[dict]) -> None:
        with self._conn() as conn:
            conn.executemany("""
                INSERT INTO findings
                    (run_id, agent_id, finding_type, title, detail, evidence,
                     risk_level, confidence, fiscal_year, source_url)
                VALUES (:run_id, :agent_id, :finding_type, :title, :detail, :evidence,
                        :risk_level, :confidence, :fiscal_year, :source_url)
            """, [{
                "run_id": run_id,
                "agent_id": f.get("agent_id", ""),
                "finding_type": f.get("finding_type", "NEUTRAL"),
                "title": f.get("title", "")[:200],
                "detail": f.get("detail", "")[:2000],
                "evidence": f.get("evidence", "")[:2000],
                "risk_level": f.get("risk_level", "MEDIUM"),
                "confidence": f.get("confidence", 0.5),
                "fiscal_year": f.get("fiscal_year", ""),
                "source_url": f.get("source_url", ""),
            } for f in findings])

    def save_valuation(self, run_id: str, scenario: str, data: dict) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO valuation_outputs
                    (run_id, scenario, dcf_value, relative_value, sotp_value,
                     blended_value, wacc, terminal_growth, current_price, upside_pct, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, scenario) DO UPDATE SET
                    blended_value=excluded.blended_value, payload_json=excluded.payload_json
            """, (
                run_id, scenario,
                data.get("dcf_value"), data.get("relative_value"),
                data.get("sotp_value"), data.get("blended_value"),
                data.get("wacc"), data.get("terminal_growth"),
                data.get("current_price"), data.get("upside_pct"),
                json.dumps(data, default=str)[:50000],
            ))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
