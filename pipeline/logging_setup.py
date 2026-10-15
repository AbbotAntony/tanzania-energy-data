"""Configure console + file logging into logs/."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from pipeline import config


def setup_logging(name: str, log_dir: Path = config.LOG_DIR, verbose: bool = False) -> Path:
    """Log INFO to console and DEBUG to ``logs/<name>_<YYYY-MM-DD>.log``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}_{date.today().isoformat()}.log"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    root.addHandler(fh)
    root.addHandler(ch)
    for noisy in ("pdfminer", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file
