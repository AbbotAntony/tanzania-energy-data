"""NLP analysis of Tanzanian energy news coverage (Daily News, 2021-2026)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repo root
NLP_DIR = ROOT / "nlp"
DATA_DIR = ROOT / "data" / "nlp"                     # gitignored: anything holding article text
TABLES_DIR = NLP_DIR / "tables"
FIGURES_DIR = NLP_DIR / "figures"
SEED = 42
