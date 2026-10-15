"""Write tidy CSV and Stata .dta files from provenance records.

Values keep the unit as published. The unit is put into the column name:
each distinct published unit gets its own ``value_<unit>`` column (for
example ``value_mmscf`` or ``value_tzs_per_litre``), so a column never mixes
units and nothing is converted.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from pipeline import config
from pipeline.editions import mark_latest_edition
from pipeline.provenance import RECORD_FIELDS, Record

log = logging.getLogger(__name__)

_LABELS: dict[str, str] = {
    "source": "Publishing organisation",
    "url": "URL of source document",
    "doc_title": "Document title",
    "pub_date": "Publication date (as published)",
    "retrieved_date": "Date document was retrieved",
    "pdf_page": "PDF page number (1-based)",
    "table_no": "Table number on PDF page / in report",
    "variable": "Variable measured",
    "unit": "Unit as published",
    "year": "Year (for fiscal-year data: year the FY ends on 30 June)",
    "month": "Month (1-12)",
    "fy": "Fiscal year (July-June)",
    "category": "Category (customer type, product, source...)",
    "value_as_printed": "Value exactly as printed in source",
    "conversion_note": "Description of any unit conversion applied",
    "qa_flag": "Quality-assurance flag",
    "latest_edition": "1 = value from most recent report edition (default series)",
}


def unit_slug(unit: str) -> str:
    """Turn a published unit into a column-name-safe slug ("TZS/litre" -> "tzs_per_litre")."""
    s = unit.strip().lower().replace("/", " per ").replace("%", "pct")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unitless"


def records_to_frame(records: Iterable[Record]) -> pd.DataFrame:
    """Build a tidy frame with one ``value_<unit>`` column per published unit."""
    rows = [r.to_row() for r in records]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=list(RECORD_FIELDS))
    for unit in sorted(df["unit"].dropna().unique()):
        col = f"value_{unit_slug(unit)}"
        df[col] = df["value"].where(df["unit"] == unit)
    df = df.drop(columns=["value"])
    value_cols = sorted(c for c in df.columns if c.startswith("value_") and c != "value_as_printed")
    front = ["year", "month", "fy", "variable", "category", *value_cols, "unit"]
    rest = [c for c in df.columns if c not in front]
    return df[[c for c in front if c in df.columns] + rest]


def _stata_ready(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    out = df.copy()
    renames = {c: re.sub(r"[^A-Za-z0-9_]", "_", c)[:32] for c in out.columns}
    out = out.rename(columns=renames)
    labels: dict[str, str] = {}
    for orig, new in renames.items():
        col = out[new]
        if col.dtype == object or pd.api.types.is_string_dtype(col):
            out[new] = col.fillna("").astype(str).str.slice(0, 2000)
        elif pd.api.types.is_integer_dtype(col) or col.isna().all():
            out[new] = col.astype("float64")
        if orig.startswith("value_") and orig != "value_as_printed":
            labels[new] = f"Value in {orig[6:]} (as published)"
        elif orig in _LABELS:
            labels[new] = _LABELS[orig]
    return out, labels


def export(records: list[Record], name: str, out_dir: Path = config.PROCESSED_DIR) -> pd.DataFrame:
    """Write ``<name>.csv`` and ``<name>.dta``; return the tidy frame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = mark_latest_edition(records_to_frame(records))
    df = df.sort_values([c for c in ("year", "month", "variable", "category") if c in df.columns],
                        na_position="last", kind="stable")
    df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
    sdf, labels = _stata_ready(df)
    sdf.to_stata(out_dir / f"{name}.dta", write_index=False, version=118,
                 variable_labels=labels, data_label=f"{name} (tz energy scraper)"[:80])
    log.info("wrote %s rows to %s.{csv,dta}", len(df), out_dir / name)
    return df
