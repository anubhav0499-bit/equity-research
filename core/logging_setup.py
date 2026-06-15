"""Structured logging setup using loguru with JSON output for production."""

from __future__ import annotations
import sys
from pathlib import Path
from loguru import logger
from .config import LOG_LEVEL, LOG_DIR


def setup_logging(run_id: str = "", level: str = LOG_LEVEL) -> None:
    logger.remove()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "{message}"
    )
    logger.add(sys.stderr, format=fmt, level=level, colorize=True)

    log_file = LOG_DIR / f"equity_research{'_' + run_id if run_id else ''}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
        level="DEBUG",
        rotation="50 MB",
        retention="30 days",
        serialize=False,
    )

    json_file = LOG_DIR / f"equity_research{'_' + run_id if run_id else ''}.jsonl"
    logger.add(
        json_file,
        serialize=True,
        level="INFO",
        rotation="100 MB",
        retention="90 days",
    )
    logger.info(f"Logging initialised | run_id={run_id or 'n/a'} | level={level}")


def get_logger(name: str):
    return logger.bind(module=name)
