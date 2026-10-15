"""TPDC (Tanzania Petroleum Development Corporation) scraper.

Listing (static HTML, verified 2026-10-15; robots.txt allows all):
  https://www.tpdc.co.tz/documents/reports   (?page=1..N)
The audited consolidated financial statements (FY2014/15 - FY2024/25) are
scanned images, so text is obtained by OCR (RapidOCR). Gas volumes appear in
the narrative "Report by those charged with governance", not in tables.
"""

from __future__ import annotations

import logging

from parsers.listing import parse_publication_cards
from pipeline.provenance import DocumentRef, Record
from pipeline.storage import StoredDoc
from scrapers.base import Scraper

log = logging.getLogger(__name__)

LISTING = "https://www.tpdc.co.tz/documents/reports"


def classify(title: str, url: str) -> str:
    return "financial_statements" if "financial statements" in title.lower() else "other"


class TpdcScraper(Scraper):
    """TPDC audited financial statements (gas volumes from the narrative, via OCR)."""

    name = "tpdc"

    def list_documents(self) -> list[DocumentRef]:
        return self.crawl_listing(
            LISTING, lambda html, url: parse_publication_cards(html, url, self.name, classify))

    def wanted(self, ref: DocumentRef) -> bool:
        return ref.category == "financial_statements"

    def parse(self, docs: list[StoredDoc]) -> dict[str, list[Record]]:
        from parsers import tpdc_fs

        records: list[Record] = []
        for doc in docs:
            try:
                records += tpdc_fs.parse(doc)
            except Exception as exc:  # one bad PDF must not stop the run
                self._fail(doc.ref.url, exc)
        tpdc_fs.flag_repeated_values(records)
        return {"tpdc_gas": records}
