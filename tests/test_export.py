"""Export: unit-named value columns, provenance, Stata round trip."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.export import export, unit_slug
from pipeline.provenance import PROVENANCE_FIELDS, Record


def _rec(value: float, unit: str, variable: str = "gas_consumption") -> Record:
    return Record(source="EWURA", url="https://x/y.pdf", doc_title="Report", retrieved_date="2026-10-15",
                  variable=variable, value=value, unit=unit, pub_date="2025-04-09", pdf_page=34,
                  table_no=31, year=2025, fy="2024/25", category="Power",
                  value_as_printed=f"{value:,}", extra={"seller": "TPDC"})


def test_unit_slug() -> None:
    assert unit_slug("MMscf") == "mmscf"
    assert unit_slug("TZS/litre") == "tzs_per_litre"
    assert unit_slug("%") == "pct"


def test_each_unit_gets_its_own_value_column(tmp_path: Path) -> None:
    df = export([_rec(47113.48, "MMscf"), _rec(74.43, "%", "gas_consumption_share")], "t", tmp_path)
    assert {"value_mmscf", "value_pct"} <= set(df.columns)
    row = df[df["variable"] == "gas_consumption"].iloc[0]
    assert row["value_mmscf"] == 47113.48 and pd.isna(row["value_pct"])


def test_provenance_columns_present(tmp_path: Path) -> None:
    df = export([_rec(1.0, "MMscf")], "t", tmp_path)
    assert set(PROVENANCE_FIELDS) <= set(df.columns)
    assert "x_seller" in df.columns


def test_stata_round_trip(tmp_path: Path) -> None:
    export([_rec(47113.48, "MMscf")], "t", tmp_path)
    back = pd.read_stata(tmp_path / "t.dta")
    assert back.loc[0, "value_mmscf"] == 47113.48
    reader = pd.io.stata.StataReader(tmp_path / "t.dta")
    with reader:
        labels = reader.variable_labels()
    assert "as published" in labels["value_mmscf"]
