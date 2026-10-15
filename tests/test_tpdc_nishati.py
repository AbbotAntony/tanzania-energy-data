"""TPDC OCR-narrative extraction, new listing layouts, Ministry narrative."""

from __future__ import annotations

from pathlib import Path

from parsers import tpdc_fs
from parsers.listing import parse_article_cards, parse_publication_cards
from parsers.narrative import extract_sentences
from parsers.ocr import boxes_to_lines
from parsers.pdf_text import PageText
from scrapers.nishati import classify as nishati_classify
from tests.conftest import make_doc

FY2023 = "TPDC Audited Consolidated Financial Statements for the Year Ended June 2023"
FY2017 = "TPDC Audited Consolidated Financial Statements for the Year Ended June 2017"

# OCR output of PDF pages 16 and 20 of the FY2022/23 statement (verified against the page image)
P16 = ("During the period under review, the total amount of gas dispatched to power plants was\n"
       "38,622.32 MMSCF up from 32,661.00 MMSCF recorded in 2021/22. For industrial segment,\n"
       "a total of 6,409.75 MMSCF down from 7,990.00 MMSCF in 2021-22 of gas was supplied to\n"
       "industrial customers in Dar es Salaam. A total of 5.99 MMSCF up from 3.06 MMSCF in 2021/22\n"
       "was supplied to household customers in Dar es Salaam and Mtwara.")
P20 = ("A total volume of 45,988.04 MMSCF of natural gas was received for\n"
       "processing, whereby Madimba plant received 32,738.51 MMSCF from the Mnazi bay gas\n"
       "field while the Songo Songo plant received 13,249.53 MMSCF from the Songo Songo gas\n"
       "field. The total quantity of 45,724.17 MMSCF was exported through the pipeline.")
KPI_2017 = ("KPI Anticipated Actual Remarks\nQuantity 17,201 mmscf 12,441 Less volumes of gas were consumed\n"
            "processed and mmscf compared to initial projections")


def _get(recs, category, fy, plant=None):
    hits = [r for r in recs if r.category == category and r.fy == fy
            and (plant is None or r.extra.get("plant") == plant)]
    assert len(hits) == 1, f"{category}/{fy}/{plant}: {[r.value for r in hits]}"
    return hits[0]


def test_tpdc_sentences_current_and_prior_year() -> None:
    recs = tpdc_fs.extract([PageText(16, P16, False)], make_doc(FY2023, "financial_statements"))
    power = _get(recs, "Power generation", "2022/23")
    assert power.value == 38622.32 and power.unit == "MMscf" and power.year == 2023
    assert "ocr_extracted" in power.qa_flag and "dispatched to power plants" in power.extra["sentence"]
    assert _get(recs, "Power generation", "2021/22").value == 32661.00
    assert _get(recs, "Industries", "2021/22").value == 7990.00   # "in 2021-22"
    assert _get(recs, "Households", "2022/23").value == 5.99


def test_tpdc_plants_and_processing() -> None:
    recs = tpdc_fs.extract([PageText(20, P20, False)], make_doc(FY2023, "financial_statements"))
    assert _get(recs, "Received at processing plants", "2022/23", plant="Madimba").value == 32738.51
    assert _get(recs, "Received at processing plants", "2022/23", plant="Songo Songo").value == 13249.53
    assert _get(recs, "Exported to pipeline", "2022/23").value == 45724.17


def test_tpdc_kpi_table_target_and_actual() -> None:
    recs = tpdc_fs.extract([PageText(17, KPI_2017, False)], make_doc(FY2017, "financial_statements"))
    vals = {r.variable: r.value for r in recs}
    assert vals == {"tpdc_gas_kpi_target": 17201.0, "tpdc_gas_kpi_actual": 12441.0}


def test_tpdc_older_layouts() -> None:
    doc18 = make_doc("TPDC Audited Consolidated Financial Statements for the Year Ended June 2018", "fs")
    kpi = tpdc_fs.extract([PageText(14, "Quantity 33,396m 25,005m The shortfall on target of 8,391m scf", False)], doc18)
    assert {r.variable: r.value for r in kpi} == {"tpdc_gas_kpi_target": 33396.0, "tpdc_gas_kpi_actual": 25005.0}
    doc19 = make_doc("TPDC Audited Consolidated Financial Statements for the Year Ended June 2019", "fs")
    bcf = tpdc_fs.extract([PageText(14, "vi) Processed and transported 15.48 Billion Cubic Feet (BCF) of natural gas", False)], doc19)
    assert [(r.value, r.unit, r.fy) for r in bcf] == [(15.48, "Bcf", "2018/19")]


def test_clean_ocr_digit_confusions() -> None:
    assert tpdc_fs.clean_ocr("JUNE 2O23 and 1,2O0.5") == "JUNE 2023 and 1,200.5"
    assert tpdc_fs.parse_number("38.622.32") == 38622.32


def test_ocr_stop_ignores_table_of_contents() -> None:
    from parsers.ocr import _stops
    from parsers.tpdc_fs import _STOP
    assert not _stops(_STOP, "3.0 STATEMENT OF DIRECTORS RESPONSIBILITIES ........ 36")
    assert _stops(_STOP, "3.0 STATEMENT OF DIRECTORS RESPONSIBILITIES\nThe directors are required")
    assert _stops(_STOP, "STATEMENTOFDIRECTORS'RESPONSIBILITIES FOR THE YEAR")
    # prose in the auditor's report must not stop the OCR
    assert not _stops(_STOP, '"the Group"), which comprise the statement of financial position as at 30 June')


def test_repeated_values_are_flagged() -> None:
    doc = make_doc(FY2017, "fs")
    recs = tpdc_fs.extract([PageText(17, KPI_2017, False)], doc)
    recs += tpdc_fs.extract([PageText(13, "Quantity processed 24,000 12,441 delays mmscf mmscf", False)],
                            make_doc("TPDC Audited Consolidated Financial Statements for the Year Ended June 2016", "fs"))
    tpdc_fs.flag_repeated_values(recs)
    actuals = [r for r in recs if r.variable == "tpdc_gas_kpi_actual"]
    assert len(actuals) == 2 and all("same_value_reported_for_other_year" in r.qa_flag for r in actuals)


def test_ocr_boxes_grouped_into_lines() -> None:
    boxes = [(100.0, 300.0, "MMSCF"), (101.0, 10.0, "38,622.32"), (140.0, 10.0, "next line")]
    assert boxes_to_lines(boxes) == ["38,622.32 MMSCF", "next line"]


def test_publication_and_article_cards(fixtures_dir: Path) -> None:
    tpdc = parse_publication_cards((fixtures_dir / "tpdc" / "reports_page1.html").read_text(encoding="utf-8"),
                                   "https://www.tpdc.co.tz/documents/reports", "tpdc")
    fs = [r for r in tpdc if "June 2023" in r.title]
    assert len(fs) == 1 and fs[0].pub_date == "2026-07-08" and fs[0].url.endswith("June 2023.pdf")
    nish = parse_article_cards((fixtures_dir / "nishati" / "useful_documents_page1.html").read_text(encoding="utf-8"),
                               "https://www.nishati.go.tz/documents/useful-documents", "nishati")
    assert len(nish) == 10 and nish[0].pub_date


def test_nishati_classify() -> None:
    assert nishati_classify("Bei Kikomo Za Mafuta Kuanzia 1 Juni 2022", "") == "cap_prices"
    assert nishati_classify("Hotuba Ya Bajeti Mwaka 2025-2026", "") == "budget_speech"
    assert nishati_classify("Power System Master Plan", "") == "master_plan"
    assert nishati_classify("Code Of Ethics Utumishi", "") == "other"


def test_narrative_keeps_swahili_gas_sentences() -> None:
    text = ("Uzalishaji wa gesi asilia ulifikia futi za ujazo bilioni 61.5 mwaka 2024/25. "
            "Hii ni sentensi isiyohusiana na nishati yenye namba 12.")
    recs = extract_sentences([PageText(3, text, False)], make_doc("Hotuba", "budget_speech"), "Ministry of Energy")
    assert len(recs) == 1 and "gesi asilia" in recs[0].value_as_printed and recs[0].pdf_page == 3


def test_tanesco_generation_table_two_column_layout() -> None:
    """Cropped text of the FY2024/25 report (p23): first row's label sits above its numbers."""
    from parsers import tanesco_ar
    from parsers.text_tables import find_tables
    text = ("Total Units Generated by TANESCO Plants, Isolated, Imports, IPPs and SPPs\nPercentage\n"
            "2023/2024 2024/2025\nChange\nGrid Own Generation\n3,349,343,806 7,901,434,438 135.91%\n"
            "Hydro kWh\nGrid Own Generation- 5,714,696,211 4,306,406,413 -24.64%\nThermal kWh\n"
            "Grid Imports kWh 1,640,596,575 83,743,720.42 -52.23%\n"
            "Isolated Station (Off-grid) 85,046,876 72,867,319 -14.32%\n"
            "Total Generation kWh 10,789,683,468 13,064,451,890 21.08%")
    block = find_tables([PageText(23, text, False)], tanesco_ar._HEADER, max_lines=14)[0]
    recs = tanesco_ar._rows(block, make_doc("TANESCO ANNUAL REPORT 2024/2025", "annual_report"))
    hydro = {r.fy: r.value for r in recs if r.category == "Grid own generation: hydro"}
    assert hydro == {"2023/24": 3349343806.0, "2024/25": 7901434438.0}
    total = next(r for r in recs if r.category == "TOTAL" and r.fy == "2024/25")
    assert "diff=700,000,000" in total.qa_flag  # imports printed without a leading 7


def test_category_keyword_after_number() -> None:
    doc = make_doc("TPDC Audited Consolidated Financial Statements for the Year Ended June 2025", "fs")
    text = "A total quantity of 40,824.45 MMSCF was exported through the pipeline to customers."
    recs = tpdc_fs.extract([PageText(38, text, False)], doc)
    assert [(r.category, r.value) for r in recs] == [("Exported to pipeline", 40824.45)]
