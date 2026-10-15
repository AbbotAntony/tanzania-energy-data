"""National Bureau of Statistics (NBS) scraper: Energy Balance.

Listing (static HTML, verified 2026-10-15; robots.txt allows all):
  https://www.nbs.go.tz/statistics/topic/energy-balance
Only the 2022 Energy Balance is published: Key Insights (PDF), Methodology
(PDF) and the full balance as an Excel workbook. The PDFs are mined for
energy sentences; the workbook is parsed cell by cell (openpyxl).
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from parsers.listing import parse_card_date
from pipeline.provenance import DocumentRef, Record
from pipeline.storage import StoredDoc
from scrapers.base import Scraper

log = logging.getLogger(__name__)

LISTING = "https://www.nbs.go.tz/statistics/topic/energy-balance"
_DATE = re.compile(r"^\s*(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})\s*(.*)$")


def parse_listing(html: str, base_url: str = LISTING) -> list[DocumentRef]:
    """Links read "21 March, 2025 2022 Energy Balance - Excel Data"."""
    soup = BeautifulSoup(html, "lxml")
    refs: dict[str, DocumentRef] = {}
    for a in soup.select("a[href]"):
        url = urljoin(base_url, a["href"].strip())
        if "/uploads/" not in url or not url.lower().endswith((".pdf", ".xlsx", ".xls")):
            continue
        text = " ".join(a.get_text(" ").split())
        m = _DATE.match(text)
        title, pub = (m.group(2), parse_card_date(m.group(1))) if m else (text, None)
        category = "workbook" if url.lower().endswith((".xlsx", ".xls")) else "report"
        refs.setdefault(url, DocumentRef("nbs", url, title, category, pub, base_url))
    return list(refs.values())


class NbsScraper(Scraper):
    """NBS Energy Balance 2022 (Excel workbook parsed; PDF sentences)."""

    name = "nbs"

    def list_documents(self) -> list[DocumentRef]:
        return self.crawl_listing(LISTING, lambda html, url: parse_listing(html, url))

    def parse(self, docs: list[StoredDoc]) -> dict[str, list[Record]]:
        from parsers.narrative import extract_sentences
        from parsers.pdf_text import read_pages

        from parsers import nbs_energy_balance

        narrative: list[Record] = []
        balance: list[Record] = []
        for doc in docs:
            try:
                if doc.ref.category == "workbook":
                    balance += nbs_energy_balance.parse(doc)
                else:
                    narrative += extract_sentences(read_pages(doc.path), doc, "NBS")
            except Exception as exc:  # one bad file must not stop the run
                self._fail(doc.ref.url, exc)
        power = [r for r in balance if r.extra.get("power_generation_flow")]
        return {"nbs_narrative": narrative, "nbs_energy_balance": balance,
                "nbs_power_generation_inputs": power}
