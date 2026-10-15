"""Bank of Tanzania listing, zonal-report and MER parsers (offline fixtures)."""

from __future__ import annotations

from pathlib import Path

import pytest

from parsers import bot_mer, bot_zonal
from parsers.listing import parse_bot_publications
from scrapers.bot import title_period
from tests.conftest import load_pages, make_doc


def _one(records, **want):
    hits = [r for r in records
            if all((r.extra.get(k[2:]) if k.startswith("x_") else getattr(r, k)) == v
                   for k, v in want.items())]
    assert len(hits) == 1, f"{want}: {len(hits)} matches"
    return hits[0]


def _zonal(name: str):
    return bot_zonal.parse_pages(load_pages(f"bot/{name}"), make_doc("Zonal", "zonal"))


# ---------------------------------------------------------------- listing
def test_listing_and_title_periods(fixtures_dir: Path) -> None:
    html = (fixtures_dir / "bot" / "zonal_listing.html").read_text(encoding="utf-8")
    refs = parse_bot_publications(html, "https://www.bot.go.tz/Publications/Filter/36", "bot", "zonal")
    assert len(refs) == 42 and all(r.url.endswith(".pdf") for r in refs)
    assert refs[0].title == "Consolidated Zonal Economic Report - Jun 26"


@pytest.mark.parametrize("title, expected", [
    ("Monthly Economic Review - Sep 26", (2026, 9)),
    ("Consolidated Zonal Economic Report - Sep-18", (2018, 9)),
    ("Consolidated Zonal Economic Report - Jun - 2024", (2024, 6)),
    ("Monthly Economic Review - Apr 2003", (2003, 4)),
])
def test_title_period(title: str, expected: tuple[int, int]) -> None:
    assert title_period(title) == expected


# ---------------------------------------------------------------- zonal
def test_gas_production_and_consumption_fiscal_years() -> None:
    recs = _zonal("zonal_2026_06_pages.json")
    power = _one(recs, variable="gas_consumption", category="Power generating plants", fy="2024/25")
    assert power.value == 44970.9 and power.unit == "MMscf" and power.year == 2025
    assert _one(recs, variable="gas_production", category="Mnazi Bay", fy="2025/26").value == 36896.3
    assert "Tanzania Petroleum Development Corporation" in power.extra["original_source"]


def test_quarterly_columns_ignore_percentage_change_labels() -> None:
    recs = _zonal("zonal_2026_03_pages.json")
    total = [r for r in recs if r.variable == "gas_production" and r.category == "TOTAL"]
    assert [(r.year, r.month, r.value) for r in total] == [
        (2025, 3, 15493.5), (2025, 12, 15877.0), (2026, 3, 15930.4)]


def test_generation_by_source_and_total_definition() -> None:
    recs = _zonal("zonal_2026_06_pages.json")
    hydro = _one(recs, variable="electricity_generation_by_source",
                 category="TANESCO plants: hydro", fy="2024/25")
    assert hydro.value == 8041406.6 and hydro.unit == "MWh" and "revised" in hydro.qa_flag
    total = _one(recs, variable="electricity_generation_by_source",
                 category="Total (as printed)", fy="2024/25")
    assert total.extra["total_definition"] == "domestic generation (TANESCO + private)"


def test_2021_total_includes_imports_and_glued_numbers_are_split() -> None:
    recs = _zonal("zonal_2021_12_pages.json")
    total = _one(recs, variable="electricity_generation_by_source",
                 category="Total (as printed)", year=2020, month=12)
    assert total.value == 2093664.2
    assert total.extra["total_definition"] == "domestic generation plus imports"
    assert "numbers_repaired_from_extraction" in total.qa_flag
    # "423,596.11,313,257.1" in the PDF text: Central and Dar es Salaam columns
    central = _one(recs, variable="electricity_generation_by_zone", category="Central",
                   year=2020, month=12)
    assert central.value == 423596.1
    assert _one(recs, variable="gas_production", year=2021, month=12).value == 18841.1


def test_units_as_printed_and_flagged() -> None:
    recs2018 = _zonal("zonal_2018_09_pages.json")
    elec = _one(recs2018, variable="electricity_generation_by_zone",
                category="Total (as printed)", year=2018, month=9)
    assert elec.unit == "Megawatts (as printed)" and "unit_label_suspect" in elec.qa_flag
    recs2016 = _zonal("zonal_2016_06_pages.json")
    dsm = _one(recs2016, variable="electricity_generation_by_zone", category="Dar es Salaam",
               fy="2015/16")
    assert dsm.unit == "000 kWh" and dsm.value == 4137098.3  # "000' KWh" is not read as a value


def test_split_digits_are_rejoined_and_flagged() -> None:
    """2016 text reads "3 4,770.7" for 34,770.7."""
    recs = _zonal("zonal_2016_06_pages.json")
    gas = _one(recs, variable="gas_production", fy="2013/14")
    assert gas.value == 34770.7 and "numbers_repaired_from_extraction" in gas.qa_flag


def test_rows_with_missing_cells_are_skipped_not_misaligned() -> None:
    recs = _zonal("zonal_2026_06_pages.json")
    assert not [r for r in recs if "Kenya" in r.category]


# ---------------------------------------------------------------- MER
def test_mer_petroleum_imports_and_exchange_rate() -> None:
    recs = bot_mer.parse_pages(load_pages("bot/mer_2026_09_pages.json"), make_doc("MER Sep 26", "mer"))
    pet = _one(recs, variable="imports_value_12m_cumulative", category="Petroleum products", year=2025)
    assert pet.value == 2257.4 and pet.unit == "USD million" and pet.month == 8
    assert "unit_label_typo_UDS" in pet.qa_flag
    fx = _one(recs, variable="exchange_rate", category="Annual average", year=2024)
    assert fx.value == 2597.4 and fx.unit == "TZS/USD" and "revised" in fx.qa_flag
