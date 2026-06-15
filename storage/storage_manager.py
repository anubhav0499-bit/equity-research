"""
Storage Manager — manages per-run file system layout and intermediate output persistence.
Every agent writes its outputs here for full auditability and reproducibility.
"""

from __future__ import annotations
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from loguru import logger

from ..core.config import OUTPUT_DIR


class StorageManager:
    """
    Per-run storage with consistent directory structure.

    Layout:
    <output_dir>/<company_slug>/<run_id>/
        Company_Profile/
        Raw_Filings/
        Parsed_Data/
            Text/
            Tables/
        Financial_Statements/
        Agent_Outputs/
        Forecasts/
        Valuation/
        Reports/
        Audit_Trail/
    """

    SUBDIRS = [
        "Company_Profile", "Raw_Filings", "Parsed_Data/Text",
        "Parsed_Data/Tables", "Financial_Statements", "Agent_Outputs",
        "Forecasts", "Valuation", "Reports", "Audit_Trail",
    ]

    def __init__(self, company_name: str, run_id: str, base_dir: Path = OUTPUT_DIR):
        self.company_name = company_name
        self.run_id = run_id
        self.company_slug = self._slugify(company_name)
        self.base_path = base_dir / self.company_slug / run_id
        self._create_dirs()

    @property
    def company_profile(self) -> Path:
        return self.base_path / "Company_Profile"

    @property
    def raw_filings(self) -> Path:
        return self.base_path / "Raw_Filings"

    @property
    def parsed_text(self) -> Path:
        return self.base_path / "Parsed_Data" / "Text"

    @property
    def parsed_tables(self) -> Path:
        return self.base_path / "Parsed_Data" / "Tables"

    @property
    def financial_statements(self) -> Path:
        return self.base_path / "Financial_Statements"

    @property
    def agent_outputs(self) -> Path:
        return self.base_path / "Agent_Outputs"

    @property
    def forecasts(self) -> Path:
        return self.base_path / "Forecasts"

    @property
    def valuation(self) -> Path:
        return self.base_path / "Valuation"

    @property
    def reports(self) -> Path:
        return self.base_path / "Reports"

    @property
    def audit_trail(self) -> Path:
        return self.base_path / "Audit_Trail"

    def _create_dirs(self) -> None:
        for sub in self.SUBDIRS:
            (self.base_path / sub).mkdir(parents=True, exist_ok=True)

    def save_json(self, data: Any, filename: str, subdir: str = "Agent_Outputs") -> Path:
        path = self.base_path / subdir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        return path

    def load_json(self, filename: str, subdir: str = "Agent_Outputs") -> Any:
        path = self.base_path / subdir / filename
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def save_text(self, text: str, filename: str, subdir: str = "Parsed_Data/Text") -> Path:
        path = self.base_path / subdir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def save_bytes(self, data: bytes, filename: str, subdir: str = "Raw_Filings") -> Path:
        path = self.base_path / subdir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def list_files(self, subdir: str, extension: str = "") -> list[Path]:
        d = self.base_path / subdir
        if not d.exists():
            return []
        if extension:
            return list(d.rglob(f"*.{extension.lstrip('.')}"))
        return [f for f in d.rglob("*") if f.is_file()]

    def save_parquet(self, df, filename: str, subdir: str = "Financial_Statements") -> Path:
        try:
            import pandas as pd
            path = self.base_path / subdir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=True)
            return path
        except Exception as e:
            logger.warning(f"Could not save parquet {filename}: {e}")
            return self.save_json(df.to_dict(), filename.replace(".parquet", ".json"), subdir)

    def get_run_manifest(self) -> dict:
        manifest: dict = {"run_id": self.run_id, "company": self.company_name,
                          "base_path": str(self.base_path), "files": {}}
        for sub in self.SUBDIRS:
            files = self.list_files(sub)
            manifest["files"][sub] = [f.name for f in files]
        return manifest

    @staticmethod
    def _slugify(name: str) -> str:
        import re
        name = name.upper().strip()
        name = re.sub(r"[^\w\s-]", "", name)
        name = re.sub(r"[\s_-]+", "_", name)
        return name[:50]
