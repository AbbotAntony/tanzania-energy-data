"""Ministry of Energy (Wizara ya Nishati) scraper.

Listing (static HTML, verified 2026-10-15; robots.txt allows all):
  https://www.nishati.go.tz/documents/useful-documents   (?page=1..N)
Kept: budget speeches (Hotuba ya Bajeti, mostly Swahili), Power System Master
Plans, the Energy Sector M&E Plan, and petroleum cap-price notices republished
from EWURA (fills months EWURA's own files lack, e.g. June 2022).
"""

from __future__ import annotations

import logging
import re

from parsers.listing import parse_article_cards
from pipeline.provenance import DocumentRef, Record
from pipeline.storage import StoredDoc
from scrapers.base import Scraper

log = logging.getLogger(__name__)

LISTING = "https://www.nishati.go.tz/documents/useful-documents"


def classify(title: str, url: str) -> str:
    t = title.lower()
    if re.search(r"bei kikomo|cap price", t):
        return "cap_prices"
    if re.search(r"budget speech|hotuba|bajeti", t):
        return "budget_speech"
    if "master plan" in t:
        return "master_plan"
    if "monitoring and evaluation plan" in t:
        return "me_plan"
    return "other"


class NishatiScraper(Scraper):
    """Ministry of Energy budget speeches, master plans and republished cap prices."""

    name = "nishati"

    def list_documents(self) -> list[DocumentRef]:
        return self.crawl_listing(
            LISTING, lambda html, url: parse_article_cards(html, url, self.name, classify))

    def wanted(self, ref: DocumentRef) -> bool:
        return ref.category != "other"

    def parse(self, docs: list[StoredDoc]) -> dict[str, list[Record]]:
        from parsers import ewura_cap_prices
        from parsers.narrative import extract_sentences
        from parsers.pdf_text import read_pages

        datasets: dict[str, list[Record]] = {"nishati_cap_prices": [], "nishati_narrative": []}
        for doc in docs:
            try:
                pages = read_pages(doc.path)
                scanned = all(len(p.text.strip()) < 30 for p in pages)
                if doc.ref.category == "cap_prices":
                    if scanned:  # e.g. the June 2022 notice is an image-only PDF
                        from parsers.ocr import ocr_pages

                        pages = ocr_pages(doc.path, max_pages=8)
                    recs = ewura_cap_prices.parse_pages(pages, doc)
                    for r in recs:
                        r.source = "EWURA notice (via Ministry of Energy site)"
                        if scanned:
                            r.qa_flag = ";".join(filter(None, [r.qa_flag, "ocr_extracted"]))
                    datasets["nishati_cap_prices"] += recs
                else:
                    datasets["nishati_narrative"] += extract_sentences(pages, doc, "Ministry of Energy")
            except Exception as exc:  # one bad PDF must not stop the run
                self._fail(doc.ref.url, exc)
        return datasets
