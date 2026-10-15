"""EWURA natural gas report parser, on page text cut from the real PDFs."""

from __future__ import annotations

import pytest

from parsers import ewura_gas
from parsers.pdf_text import decode_cids
from pipeline.provenance import PROVENANCE_FIELDS
from tests.conftest import load_pages, make_doc

FY2425 = "Natural Gas Sub-Sector Regulatory Performance Report for the Year Ended 30th June 2025"
FY2324 = "Natural Gas Sub-Sector Regulatory Performance Report for the Year Ended 30th June 2024"
FY2223 = "Natural Gas Sub-Sector Regulatory Performance Report for the Year Ended 30th June 2023"
FY1920 = "Natural Gas Sub-Sector Regulatory Performance Report For the Year Ended 30th June 2020"


def _get(records, variable, category, fy, seller=None):
    hits = [r for r in records if r.variable == variable and r.category == category and r.fy == fy
            and (seller is None or r.extra.get("seller") == seller)]
    assert len(hits) == 1, f"{variable}/{category}/{fy}/{seller}: {len(hits)} matches"
    return hits[0]


@pytest.fixture(scope="module")
def fy2425():
    return ewura_gas.parse_pages(load_pages("ewura/gas_fy2425_pages.json"),
                                 make_doc(FY2425, "gas_report"))


def test_report_fy() -> None:
    assert ewura_gas.report_fy(FY2425) == "2024/25"
    assert ewura_gas.report_fy(FY1920) == "2019/20"


def test_consumption_by_category_and_seller(fy2425) -> None:
    power = _get(fy2425, "gas_consumption", "Power", "2024/25", "Total")
    assert power.value == 47113.48 and power.unit == "MMscf"
    assert power.pdf_page == 34 and power.table_no == 31
    assert _get(fy2425, "gas_consumption", "Power", "2024/25", "TPDC").value == 33807.51
    assert _get(fy2425, "gas_consumption", "Industries", "2024/25", "M&P").value is None  # "-"
    assert _get(fy2425, "gas_consumption_share", "Power", "2024/25").value == 74.43


def test_annex_multi_year_consumption_handles_blank_cells(fy2425) -> None:
    assert _get(fy2425, "gas_consumption", "Power", "2023/24", "Total").value == 70061.45
    assert _get(fy2425, "gas_consumption", "Power", "2020/21", "Total").value == 45664.79
    # CNG FY2020/21 volume is blank (only "0.00%" printed); later years must not shift left
    assert _get(fy2425, "gas_consumption", "CNG Vehicles", "2020/21", "Total").value is None
    assert _get(fy2425, "gas_consumption", "CNG Vehicles", "2021/22", "Total").value == 94.6
    hh = _get(fy2425, "gas_consumption", "Households, Commercial and Institutions", "2022/23", "Total")
    assert hh.value == 18.02  # label wraps over three lines in the PDF


def test_supply_by_plant_five_years(fy2425) -> None:
    rows = {r.fy: r.value for r in fy2425
            if r.variable == "gas_supply_by_plant" and r.category == "Songas"}
    assert rows == {"2020/21": 23978.40, "2021/22": 28440.94, "2022/23": 34220.40,
                    "2023/24": 32335.60, "2024/25": 20130.70}


def test_production_by_field_and_pipeline_utilisation(fy2425) -> None:
    assert _get(fy2425, "gas_produced_by_field", "Mnazi Bay Field", "2024/25").value == 34216.55
    assert _get(fy2425, "pipeline_capacity_utilisation", "Songas", "2024/25").value == 50
    assert _get(fy2425, "pipeline_capacity_utilisation", "TPDC", "2020/21").value == 13
    assert _get(fy2425, "pipeline_installed_capacity", "TPDC", "2024/25").value == 784


def test_provenance_on_every_record(fy2425) -> None:
    for r in fy2425:
        for f in PROVENANCE_FIELDS:
            assert getattr(r, f) is not None, f
        assert r.year == 2025 or r.fy != "2024/25"


def test_seller_sum_mismatch_is_flagged_not_fixed() -> None:
    recs = ewura_gas.parse_pages(load_pages("ewura/gas_fy2324_pages.json"),
                                 make_doc(FY2324, "gas_report"))
    cng = _get(recs, "gas_consumption", "CNG Vehicles", "2023/24", "Total")
    assert cng.value == 308.72  # as printed, although 271.88 + 246.47 = 518.35
    assert "total_ne_sum_of_sellers" in cng.qa_flag


def test_wrapped_label_serial_number_is_not_a_value() -> None:
    """"Maurel and Prom / 4 879.5 1,045.48 ..." - the 4 is the row number."""
    recs = ewura_gas.parse_pages(load_pages("ewura/gas_fy2223_pages.json"),
                                 make_doc(FY2223, "gas_report"))
    mp = _get(recs, "gas_supply_by_plant", "Maurel & Prom (M&P)", "2020/21")
    assert mp.value == 879.5
    total = _get(recs, "gas_supply_by_plant", "TOTAL", "2020/21")
    assert "total_ne_sum" not in total.qa_flag
    songo = _get(recs, "plant_installed_capacity", "TPDC Songo Songo", "2022/23")
    assert songo.value == 140


def test_short_row_is_right_aligned_and_flagged() -> None:
    recs = ewura_gas.parse_pages(load_pages("ewura/gas_fy2223_pages.json"),
                                 make_doc(FY2223, "gas_report"))
    cng = _get(recs, "gas_consumption", "CNG Vehicles", "2021/22", "Total")
    assert cng.value == 94.60 and "short_row_right_aligned" in cng.qa_flag
    assert not [r for r in recs if r.category == "Industries" and r.value == 35]  # prose, not data


def test_glyph_encoded_pages_are_decoded_and_flagged() -> None:
    pages = load_pages("ewura/gas_fy1920_cid_pages.json")
    assert all(p.cid_decoded for p in pages)
    recs = ewura_gas.parse_pages(pages, make_doc(FY1920, "gas_report"))
    total = _get(recs, "gas_supply_by_plant", "TOTAL", "2019/20")
    assert total.value == 59831.43 and "text_glyph_decoded" in total.qa_flag
    assert _get(recs, "gas_supply_by_plant", "TOTAL", "2018/19").value == 60911.87
    assert _get(recs, "pipeline_capacity_utilisation", "TPDC", "2019/20").value == 11.18


def test_decode_cids_offset() -> None:
    text, ok = decode_cids("(cid:45)(cid:80)(cid:78)(cid:92)(cid:89)(cid:76)(cid:3)(cid:24)")
    assert ok and text == "Figure 1"
