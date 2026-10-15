"""EWURA (Energy and Water Utilities Regulatory Authority) scraper.

Listing pages (static HTML, verified 2026-10-15; robots.txt allows all):
  https://www.ewura.go.tz/publications/performance-report   (?page=1..N)
  https://www.ewura.go.tz/publications/petroleum-price      (?page=1..N)
Note: the bare domain ``ewura.go.tz`` times out; always use ``www.``.
"""

from __future__ import annotations

import logging
import re

from parsers.listing import parse_egov_cards
from pipeline.provenance import DocumentRef, Record
from pipeline.storage import StoredDoc
from scrapers.base import Scraper

log = logging.getLogger(__name__)

BASE = "https://www.ewura.go.tz"
LISTINGS: tuple[str, ...] = (
    f"{BASE}/publications/performance-report",
    f"{BASE}/publications/petroleum-price",
)

MIN_YEAR = 2014  # one year before the study window, for base-year comparisons
CATEGORIES: tuple[str, ...] = (
    "gas_report", "petroleum_report", "electricity_report",
    "electricity_update", "cap_prices",
)


def classify(title: str, url: str) -> str:
    """Assign an EWURA document to a category from its title (and URL)."""
    t = f"{title} {url.rsplit('/', 1)[-1]}".lower()
    if "water" in t or "maji" in t:
        return "other"
    if re.search(r"cap (petroleum products )?prices|indicative price|en_cap_\d", t):
        return "cap_prices"
    if "natural gas" in t:
        return "gas_report"
    if "petroleum" in t and re.search(r"perform|review|report", t):
        return "petroleum_report"
    if ("electric" in t or "electricty" in t) and "index" not in t:
        if "update" in t:
            return "electricity_update"
        if "muhtasari" in t or "summary" in t:
            return "other"
        return "electricity_report"
    return "other"


class EwuraScraper(Scraper):
    """Lists, downloads and parses EWURA sector reports and cap prices."""

    name = "ewura"

    def list_documents(self) -> list[DocumentRef]:
        refs: dict[str, DocumentRef] = {}
        for start in LISTINGS:
            for ref in self.crawl_listing(start, self._parse_page):
                refs.setdefault(ref.url, ref)
        return list(refs.values())

    def wanted(self, ref: DocumentRef) -> bool:
        if ref.category not in CATEGORIES:
            return False
        years = [int(y) for y in re.findall(r"\b(20\d\d)\b", ref.title)]
        # Listing dates are unreliable, so filter on the year in the title.
        return not years or max(years) >= MIN_YEAR

    def parse(self, docs: list[StoredDoc]) -> dict[str, list[Record]]:
        from parsers import ewura_cap_prices, ewura_gas, ewura_petroleum

        parsers = {
            "gas_report": ewura_gas.parse,
            "petroleum_report": ewura_petroleum.parse,
            "cap_prices": ewura_cap_prices.parse,
        }
        datasets: dict[str, list[Record]] = {}
        for doc in docs:
            parser = parsers.get(doc.ref.category)
            if parser is None:
                continue  # electricity reports: stored now, parsed in a later step
            try:
                for name, records in parser(doc).items():
                    datasets.setdefault(name, []).extend(records)
            except Exception as exc:  # one bad PDF must not stop the run
                self._fail(doc.ref.url, exc)
        return datasets

    def _parse_page(self, html: str, url: str) -> list[DocumentRef]:
        return parse_egov_cards(html, url, self.name, classify)
