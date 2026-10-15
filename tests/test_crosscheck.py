"""Cross-source comparison and plant-name canonicalisation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from parsers.bot_zonal import canon_plant
from pipeline import crosscheck


def _write(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_cross_source_uses_latest_edition_and_flags_differences(tmp_path: Path) -> None:
    base = {"variable": "gas_consumption", "doc_title": "d", "pdf_page": 1}
    _write(tmp_path / "ewura_gas.csv", [
        {**base, "category": "Power", "fy": "2022/23", "x_seller": "Total",
         "value_mmscf": 51609.78, "latest_edition": 1},
        {**base, "category": "Power", "fy": "2022/23", "x_seller": "Total",
         "value_mmscf": 99999.0, "latest_edition": 0},          # superseded edition: ignored
        {**base, "category": "Power", "fy": "2022/23", "x_seller": "TPDC",
         "value_mmscf": 1.0, "latest_edition": 1},              # seller split: ignored
    ])
    _write(tmp_path / "bot_gas.csv", [
        {**base, "category": "Power generating plants", "fy": "2022/23",
         "value_mmscf": 68787.6, "latest_edition": 1},
    ])
    out = crosscheck.run(tmp_path)
    row = out[out.concept.str.startswith("Gas consumed by power")].iloc[0]
    assert (row.value_a, row.value_b, row.flag) == (51609.78, 68787.6, "differs")
    assert (tmp_path / "qa_cross_source.csv").exists()


def test_canon_plant() -> None:
    assert canon_plant("Tanesco") == canon_plant("TANESCO") == "TANESCO"
    assert canon_plant("New Pangani Power Plant") == "New Pangani Falls (hydro)"
    assert canon_plant("Unknown Plant") == "Unknown Plant"
