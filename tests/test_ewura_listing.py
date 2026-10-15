"""EWURA listing pages: card parsing, pagination, classification (saved HTML)."""

from __future__ import annotations

from pathlib import Path

import pytest

from parsers.listing import find_page_links, parse_card_date, parse_egov_cards
from scrapers.ewura import classify

PERF_URL = "https://www.ewura.go.tz/publications/performance-report?page=2"


@pytest.fixture
def perf_refs(fixtures_dir: Path):
    html = (fixtures_dir / "ewura" / "performance_report_page2.html").read_text(encoding="utf-8")
    return parse_egov_cards(html, PERF_URL, "ewura", classify)


def test_cards_have_title_date_and_pdf_url(perf_refs) -> None:
    gas = [r for r in perf_refs if r.url.endswith("NaturalGasReport.pdf")]
    assert len(gas) == 1
    ref = gas[0]
    assert ref.title == ("Natural Gas Sub-Sector Regulatory Performance Report "
                         "for the Year Ended 30th June 2023")
    assert ref.pub_date == "2024-06-14"
    assert ref.category == "gas_report"
    assert ref.listing_url == PERF_URL


def test_all_cards_are_pdfs(perf_refs) -> None:
    assert perf_refs and all(r.url.lower().endswith(".pdf") for r in perf_refs)


def test_pagination_links(fixtures_dir: Path) -> None:
    html = (fixtures_dir / "ewura" / "petroleum_price_page1.html").read_text(encoding="utf-8")
    pages = find_page_links(html, "https://www.ewura.go.tz/publications/petroleum-price")
    assert pages[0].endswith("petroleum-price?page=2")
    assert len(pages) >= 7


@pytest.mark.parametrize("title, expected", [
    ("Natural Gas Sub-Sector Regulatory Performance Report for the Year Ended 30th June 2025",
     "gas_report"),
    ("The Mid and Downstream Petroleum Sub-sector Performance Report FY 2022/23", "petroleum_report"),
    ("Cap Prices for Petroleum Products effective from Wednesday 7th October 2026", "cap_prices"),
    ("Indicative and Cap Petroleum Products Prices in the Country Effective 9th December 2009",
     "cap_prices"),
    ("The Electricity Sub-Sector Regulatory Performance Updates – March 2026", "electricity_update"),
    ("Electricity Sub-Sector Performance Report FY 2023/24", "electricity_report"),
    ("Electricity Regulatory Index for Africa 2022", "other"),
    ("Water Utilities Performance Review Report 2023/24", "other"),
])
def test_classify(title: str, expected: str) -> None:
    assert classify(title, "https://www.ewura.go.tz/uploads/documents/x.pdf") == expected


def test_parse_card_date() -> None:
    assert parse_card_date("14 Jun, 2024") == "2024-06-14"
    assert parse_card_date("not a date") is None
