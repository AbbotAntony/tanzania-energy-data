"""Quality-assurance reports: missing years and disagreeing figures.

Nothing here edits data. Problems are written to CSV files in
data/processed/ for the researcher to resolve:

* ``qa_missing_years.csv``: for each dataset/variable/category, the study
  years (2015-2025) with no observation.
* ``qa_discrepancies.csv``: the same variable, category and period reported
  with different values by different documents (report editions, or
  different sources). Differences under ``REL_TOLERANCE`` are ignored.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pipeline import config

log = logging.getLogger(__name__)

REL_TOLERANCE = 0.001  # 0.1 %; smaller gaps are treated as rounding
KEY_COLS = ["variable", "category", "fy", "year", "month", "x_seller", "x_market", "x_port",
            "x_price_basis", "x_measure_as_labelled"]


def _value(df: pd.DataFrame) -> pd.Series:
    """Collapse the value_<unit> columns back to one numeric series."""
    cols = [c for c in df.columns if c.startswith("value_") and c != "value_as_printed"]
    return df[cols].bfill(axis=1).iloc[:, 0] if cols else pd.Series(dtype=float)


def missing_years(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if df.empty or "year" not in df.columns:
        return pd.DataFrame()
    num = df[_value(df).notna()]
    rows = []
    for (variable, category), g in num.groupby(["variable", "category"], dropna=False):
        have = set(g["year"].dropna().astype(int))
        missing = [y for y in config.STUDY_YEARS if y not in have]
        if missing:
            rows.append({"dataset": dataset, "variable": variable, "category": category,
                         "years_present": ",".join(map(str, sorted(have))),
                         "years_missing": ",".join(map(str, missing))})
    return pd.DataFrame(rows)


def discrepancies(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.assign(_v=_value(df))
    work = work[work["_v"].notna()]
    keys = [c for c in KEY_COLS if c in work.columns]
    work[keys] = work[keys].fillna("")
    rows = []
    for key, g in work.groupby(keys):
        docs = g.drop_duplicates(["url", "_v"])
        if docs["_v"].nunique() < 2:
            continue
        lo, hi = docs["_v"].min(), docs["_v"].max()
        if hi == 0 or (hi - lo) / abs(hi) <= REL_TOLERANCE:
            continue
        rows.append({
            "dataset": dataset, **dict(zip(keys, key)),
            "n_documents": docs["url"].nunique(), "min": lo, "max": hi,
            "rel_diff_pct": round(100 * (hi - lo) / abs(hi), 2),
            "values_by_document": " | ".join(
                f"{r.value_as_printed} [{r.doc_title[:70]} p{r.pdf_page}]"
                for r in docs.itertuples()),
        })
    return pd.DataFrame(rows)


def run_all(processed: Path = config.PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan every dataset CSV in ``processed`` and write the two QA reports."""
    miss, disc = [], []
    for csv in sorted(processed.glob("*.csv")):
        if csv.name.startswith("qa_") or "narrative" in csv.name:
            continue
        df = pd.read_csv(csv, low_memory=False)
        miss.append(missing_years(df, csv.stem))
        disc.append(discrepancies(df, csv.stem))
    m = pd.concat(miss, ignore_index=True) if miss else pd.DataFrame()
    d = pd.concat(disc, ignore_index=True) if disc else pd.DataFrame()
    m.to_csv(processed / "qa_missing_years.csv", index=False)
    d.to_csv(processed / "qa_discrepancies.csv", index=False)
    log.info("QA: %d variable/category series with missing years; %d discrepancies",
             len(m), len(d))
    from pipeline import crosscheck

    crosscheck.run(processed)
    return m, d
