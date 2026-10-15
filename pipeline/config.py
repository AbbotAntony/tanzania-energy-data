"""Project-wide paths and settings."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
CACHE_DIR: Path = RAW_DIR / "_cache"
PROCESSED_DIR: Path = DATA_DIR / "processed"
LOG_DIR: Path = PROJECT_ROOT / "logs"

_CONTACT: str = os.environ.get("TZ_SCRAPER_CONTACT", "").strip()
USER_AGENT: str = (
    "TZEnergyResearch/0.1 (academic research on Tanzanian natural gas "
    "and petroleum statistics" + (f"; contact: {_CONTACT}" if _CONTACT else "") + ")"
)
MIN_DELAY_S: float = 3.0
MAX_DELAY_S: float = 5.0
CONNECT_TIMEOUT_S: float = 15.0
READ_TIMEOUT_S: float = 120.0
MAX_RETRIES: int = 3
BACKOFF_FACTOR: float = 2.0

STUDY_YEARS: range = range(2015, 2026)
