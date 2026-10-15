"""News topic matching, Citizen sitemap, Daily News parsing, NBS listing."""

from __future__ import annotations

import json
from pathlib import Path

from parsers.news import html_to_text, match_topics
from pipeline.provenance import DocumentRef
from pipeline.storage import StoredDoc
from scrapers.nbs import parse_listing
from scrapers.news_citizen import parse_sitemap
from scrapers.news_dailynews import DailyNewsScraper


def test_match_topics_requires_a_strong_topic() -> None:
    assert match_topics("Natural gas output falls as Julius Nyerere Hydropower expands") == ["natural_gas", "jnhpp"]
    assert match_topics("Mtwara residents celebrate cashew harvest") == []          # weak only
    assert "mtwara_mnazi" in match_topics("Mtwara gas plant at Madimba runs at 40 percent; TPDC says")
    assert match_topics("TANESCO cuts HFO use") == ["hfo"]


def test_html_to_text() -> None:
    assert html_to_text("<p>Gas &amp; oil</p><p>2024</p>") == "Gas & oil 2024"


def test_citizen_sitemap_parsed_without_article_text(fixtures_dir: Path) -> None:
    items = parse_sitemap((fixtures_dir / "citizen" / "sitemap.xml").read_bytes())
    assert len(items) >= 20
    assert all(i["loc"].startswith("https://www.thecitizen.co.tz/") and i["title"] for i in items)
    assert set(items[0]) == {"loc", "title", "date", "access"}


def test_dailynews_keeps_only_phrase_matches(tmp_path: Path) -> None:
    page = [
        {"id": 1, "date": "2025-04-11T10:00:00", "modified": "2025-04-11T10:00:00",
         "link": "https://dailynews.co.tz/a/", "title": {"rendered": "Natural gas use expands"},
         "content": {"rendered": "<p>TPDC supplied more gas to industries.</p>"}, "excerpt": {"rendered": ""}},
        {"id": 2, "date": "2025-04-12T10:00:00", "modified": "", "link": "https://dailynews.co.tz/b/",
         "title": {"rendered": "Natural beauty of Mtwara"},   # matched the word search, not a phrase
         "content": {"rendered": "<p>Tourists visit the gas-free beaches.</p>"}, "excerpt": {"rendered": ""}},
    ]
    f = tmp_path / "page.json"
    f.write_text(json.dumps(page), encoding="utf-8")
    doc = StoredDoc(DocumentRef("dailynews", "u", "API search 'natural gas' page 1", "api_page"), f, "", "2026-10-15")
    recs = DailyNewsScraper.parse(DailyNewsScraper.__new__(DailyNewsScraper), [doc])["news_dailynews"]
    assert [r.url for r in recs] == ["https://dailynews.co.tz/a/"]
    r = recs[0]
    assert (r.year, r.month, r.pub_date) == (2025, 4, "2025-04-11") and "natural_gas" in r.extra["topics"]
    assert r.extra["text"] == "TPDC supplied more gas to industries."


def test_nbs_listing(fixtures_dir: Path) -> None:
    refs = parse_listing((fixtures_dir / "nbs" / "energy_balance.html").read_text(encoding="utf-8"))
    assert {r.category for r in refs} == {"report", "workbook"} and len(refs) == 3
    assert all(r.pub_date == "2025-03-21" for r in refs)


def test_nbs_energy_balance_sheet_blocks_and_power_flows() -> None:
    from parsers.nbs_energy_balance import parse_sheet
    from tests.conftest import make_doc
    rows = [
        (None, "2022 Energy Balance for Mainland Tanzania - Detailed"),
        (None, None),
        (None, "Thousand Tonnes of Oil Equivalent (ktoe)", "Fuel oil", "Natural gas", "Electricity"),
        (None, "Imports (+)", 25.17, 0, 13.46),
        (None, "Main activity producer electricity", -4.042, -1456.559, 799.985),
        (None, "Coal-to-liquids plants", 0, "#VALUE!", 0),
        (None, "GWh", "Fuel oil", "Natural gas", "Electricity"),
        (None, "Main activity producer electricity", -47.0, -16939.8, 9303.8),
    ]
    recs = parse_sheet(rows, "T2", make_doc("NBS 2022 Energy Balance - Excel Data", "workbook"))
    gas = [r for r in recs if r.category == "Natural gas" and r.extra["flow"] == "Main activity producer electricity"]
    assert sorted((r.unit, r.value) for r in gas) == [("GWh", -16939.8), ("ktoe", -1456.559)]
    assert all(r.extra["power_generation_flow"] for r in gas) and gas[0].year == 2022
    err = [r for r in recs if r.qa_flag.startswith("excel_error")]
    assert len(err) == 1 and err[0].value is None
    imp = next(r for r in recs if r.extra["flow"] == "Imports (+)" and r.category == "Fuel oil")
    assert not imp.extra["power_generation_flow"] and imp.extra["geography"] == "Mainland Tanzania"
