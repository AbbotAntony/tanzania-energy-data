"""Bank of Tanzania scraper.

Listing pages (static HTML tables, verified 2026-10-15; no robots.txt published):
  https://www.bot.go.tz/Publications/Filter/36   Consolidated Zonal Economic Report
      (quarterly, Mar 2016 - ; electricity generation by zone/source, natural gas
      production and consumption, sourced from TANESCO and TPDC)
  https://www.bot.go.tz/Publications/Filter/1    Monthly Economic Review
      (petroleum import values in USD, exchange rates)
"""

from __future__ import annotations

import logging
import re

from parsers.listing import parse_bot_publications
from pipeline.provenance import DocumentRef, Record
from pipeline.storage import StoredDoc
from scrapers.base import Scraper

log = logging.getLogger(__name__)

BASE = "https://www.bot.go.tz"
LISTINGS: dict[str, str] = {
    "zonal": f"{BASE}/Publications/Filter/36?lang=en",
    "mer": f"{BASE}/Publications/Filter/1?lang=en",
}
MIN_YEAR = 2014
_MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()
_PERIOD = re.compile(r"\b(" + "|".join(_MONTHS) + r")[a-z]*\s*[-–]?\s*(\d{4}|\d{2})\b", re.I)


def title_period(title: str) -> tuple[int, int] | None:
    """"... - Sep 26" / "Sep-18" / "Jun - 2024" / "Apr 2003" -> (year, month)."""
    tail = title.split(" - ", 1)[-1] if " - " in title else title
    m = _PERIOD.search(tail)
    if not m:
        return None
    year = int(m.group(2))
    year = year + 2000 if year < 100 else year
    return year, _MONTHS.index(m.group(1).lower()[:3]) + 1


class BotScraper(Scraper):
    """Bank of Tanzania zonal economic reports and Monthly Economic Reviews."""

    name = "bot"

    def list_documents(self) -> list[DocumentRef]:
        refs: list[DocumentRef] = []
        for category, url in LISTINGS.items():
            refs += self.crawl_listing(
                url, lambda html, u, c=category: parse_bot_publications(html, u, self.name, c))
        return refs

    def wanted(self, ref: DocumentRef) -> bool:
        period = title_period(ref.title)
        return period is not None and period[0] >= MIN_YEAR

    def parse(self, docs: list[StoredDoc]) -> dict[str, list[Record]]:
        from parsers import bot_mer, bot_zonal

        parsers = {"zonal": bot_zonal.parse, "mer": bot_mer.parse}
        datasets: dict[str, list[Record]] = {}
        for doc in docs:
            try:
                for name, records in parsers[doc.ref.category](doc).items():
                    datasets.setdefault(name, []).extend(records)
            except Exception as exc:  # one bad PDF must not stop the run
                self._fail(doc.ref.url, exc)
        return datasets
