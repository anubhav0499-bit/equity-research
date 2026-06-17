"""
Central configuration for the Equity Research Platform.
All settings are loaded from environment variables or config.yaml with typed defaults.
"""

from __future__ import annotations
import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Output directory resolution (env var > platform detection > default home dir)
_env_reports_dir = os.getenv("ER_REPORTS_DIR", "")
IS_COLAB = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ
IS_KAGGLE = "KAGGLE_URL_BASE" in os.environ

if _env_reports_dir:
    OUTPUT_DIR = Path(_env_reports_dir)
elif IS_COLAB:
    OUTPUT_DIR = Path("/content/equity_research_reports")
elif IS_KAGGLE:
    OUTPUT_DIR = Path("/kaggle/working/equity_research_reports")
else:
    OUTPUT_DIR = Path.home() / "equity_research_reports"

REPORTS_DIR = OUTPUT_DIR  # alias kept for backward compat


def _load_yaml() -> dict:
    cfg_path = BASE_DIR / "config.yaml"
    if cfg_path.exists():
        try:
            with open(cfg_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}

_YAML = _load_yaml()


def _get_nested(keys: list[str], default=None):
    """Read a nested YAML value by key path, e.g. ['llm', 'temperature']."""
    node = _YAML
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k)
        if node is None:
            return default
    return node


# ── API Keys ──────────────────────────────────────────────────────
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
TOGETHER_API_KEY   = os.getenv("TOGETHER_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
FMP_API_KEY        = os.getenv("FMP_API_KEY", "")          # Financial Modeling Prep
ALPHA_VANTAGE_KEY  = os.getenv("ALPHA_VANTAGE_KEY", "")
POLYGON_API_KEY    = os.getenv("POLYGON_API_KEY", "")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")

PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "openai":      {"primary": "gpt-4o",              "fast": "gpt-4o-mini"},
    "anthropic":   {"primary": "claude-opus-4-8",     "fast": "claude-haiku-4-5-20251001"},
    "groq":        {"primary": "llama-3.3-70b-versatile", "fast": "llama-3.1-8b-instant"},
    "gemini":      {"primary": "gemini-1.5-pro",      "fast": "gemini-2.0-flash"},
    "together":    {"primary": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    "fast":    "meta-llama/Llama-3.1-8B-Instruct-Turbo"},
    "openrouter":  {"primary": "anthropic/claude-opus-4-8",
                    "fast":    "google/gemini-flash-1.5"},
    "ollama":      {"primary": "qwen2.5:7b",          "fast": "phi3.5:3.8b"},
}


@dataclass
class LLMConfig:
    provider: str = LLM_PROVIDER
    primary_model: str = PROVIDER_MODELS.get(LLM_PROVIDER, {}).get("primary", "gpt-4o")
    fast_model: str = PROVIDER_MODELS.get(LLM_PROVIDER, {}).get("fast", "gpt-4o-mini")
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 300
    max_retries: int = 3
    retry_delay: float = 2.0
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    lmstudio_base_url: str = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234")

LLM_CONFIG = LLMConfig()


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-large-en-v1.5"
    fallback_model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    chunk_size: int = 512
    chunk_overlap: int = 100

EMBEDDING_CONFIG = EmbeddingConfig()


@dataclass
class DatabaseConfig:
    sqlite_path: Path = DATA_DIR / "equity_research.db"
    duckdb_path: Path = DATA_DIR / "analytics.duckdb"
    faiss_dir: Path = DATA_DIR / "faiss_index"
    parquet_dir: Path = DATA_DIR / "parquet"

DB_CONFIG = DatabaseConfig()


@dataclass
class AcquisitionConfig:
    sec_base_url: str = "https://efts.sec.gov/LATEST/search-index"
    sec_submissions_url: str = "https://data.sec.gov/submissions"
    sec_company_facts_url: str = "https://data.sec.gov/api/xbrl/companyfacts"
    sec_user_agent: str = "EquityResearchPlatform equity.research@platform.com"
    nse_base_url: str = "https://www.nseindia.com"
    bse_base_url: str = "https://www.bseindia.com"
    download_timeout: int = 60
    max_file_size_mb: int = 200
    years_history: int = 7
    rate_limit_rps: float = 0.5
    playwright_headless: bool = True

ACQUISITION_CONFIG = AcquisitionConfig()


@dataclass
class ModelingConfig:
    min_history_years: int = 5
    forecast_years: int = 5
    terminal_growth_rate_default: float = 3.0
    wacc_floor: float = 6.0
    wacc_ceiling: float = 20.0
    peer_group_min_size: int = 3
    peer_group_max_size: int = 10

MODELING_CONFIG = ModelingConfig()


@dataclass
class ReportConfig:
    min_word_count: int = 15000
    target_word_count: int = 25000
    firm_name: str = "Equity Intelligence Research"
    firm_tagline: str = "Institutional-Grade Independent Research"
    analyst_name: str = "AI Research Platform"
    heading_font: str = "Calibri"
    body_font: str = "Calibri"
    primary_color: str = "#1a237e"
    accent_color: str = "#b71c1c"

REPORT_CONFIG = ReportConfig()


# ── Forensic Thresholds ───────────────────────────────────────────
@dataclass
class ForensicThresholds:
    beneish_manipulation: float = -1.78
    beneish_high_risk: float = -1.0
    altman_safe: float = 2.60
    altman_distress: float = 1.10
    piotroski_strong: int = 7
    piotroski_weak: int = 2
    accrual_ratio_high: float = 0.10
    accrual_ratio_moderate: float = 0.05

FORENSIC_THRESHOLDS = ForensicThresholds()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = DATA_DIR / "logs"


@dataclass
class RAGConfig:
    # ── Embedding ─────────────────────────────────────────────────────
    model_name: str   = "BAAI/bge-small-en-v1.5"   # ~130MB, free, local
    fallback_model: str = "all-MiniLM-L6-v2"
    # ── Chunking ──────────────────────────────────────────────────────
    chunking_mode: str  = "auto"       # auto | semantic | recursive | contextual
    child_chunk_size: int   = 256      # small chunks for high-precision retrieval
    child_chunk_overlap: int = 32
    parent_chunk_size: int  = 1024     # large chunks returned as context (multi-vector)
    parent_chunk_overlap: int = 128
    semantic_threshold: float = 0.75   # cosine threshold for semantic split boundaries
    # ── Retrieval ─────────────────────────────────────────────────────
    top_k: int = 5
    candidate_multiplier: int = 4      # retrieve top_k × multiplier before re-ranking
    min_candidates: int = 20
    hyde_enabled: bool = True          # Hypothetical Document Embeddings
    # ── Compression ───────────────────────────────────────────────────
    compression_enabled: bool = True
    compression_max_chars: int = 8000
    # ── Memory ────────────────────────────────────────────────────────
    memory_max_turns: int = 10
    memory_max_chars: int = 4000
    # ── Guardrails ────────────────────────────────────────────────────
    groundedness_threshold: float = 0.70
    confidence_threshold: float  = 0.60
    # ── Evaluation ────────────────────────────────────────────────────
    ragas_enabled: bool = False        # expensive — opt-in per run
    # ── Backend ───────────────────────────────────────────────────────
    vector_backend: str = "faiss"      # faiss | chroma

RAG_CONFIG = RAGConfig()


def validate_llm_config() -> str:
    """
    Return the active provider name, or raise RuntimeError with setup instructions
    if no LLM provider is reachable.
    """
    _keys = {
        "groq":        GROQ_API_KEY,
        "openai":      OPENAI_API_KEY,
        "anthropic":   ANTHROPIC_API_KEY,
        "together":    TOGETHER_API_KEY,
        "openrouter":  OPENROUTER_API_KEY,
        "gemini":      GOOGLE_API_KEY,
    }
    if LLM_PROVIDER != "auto":
        if LLM_PROVIDER == "ollama":
            return "ollama"
        if _keys.get(LLM_PROVIDER, ""):
            return LLM_PROVIDER
        raise RuntimeError(
            f"LLM_PROVIDER is set to '{LLM_PROVIDER}' but {LLM_PROVIDER.upper()}_API_KEY is empty.\n"
            f"Set the key in .env or switch LLM_PROVIDER=auto to use any available provider."
        )
    # auto — cascade through providers
    for provider, key in _keys.items():
        if key:
            return provider
    # Ollama is always last resort (no key needed)
    return "ollama"
