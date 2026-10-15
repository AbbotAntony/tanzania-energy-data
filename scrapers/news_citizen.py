"""The Citizen (thecitizen.co.tz): headline metadata only, from the news sitemap.

robots.txt allows "/" but disallows "/service/" (site search), and many
articles are paywalled ("Subscription"), so only title, date and URL are kept,
never article text. The news sitemap lists only the latest ~40 articles
(1-2 days), so it is re-fetched on every run (no cache) and every stored copy
is parsed: history accumulates when the scraper is run regularly.
"""

from __future__ import annotations

import logging

import requests
from lxml import etree

from parsers.news import article_record, match_topics
from pipeline.http_client import RobotsDisallowed
from pipeline.provenance import DocumentRef, Record
from pipeline.storage import StoredDoc, store_raw
from scrapers.base import Scraper

log = logging.getLogger(__name__)

SITEMAP = "https://www.thecitizen.co.tz/sitemap.xml"
NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9",
      "n": "http://www.google.com/schemas/sitemap-news/0.9"}


def parse_sitemap(xml: bytes) -> list[dict[str, str]]:
    root = etree.fromstring(xml)
    out = []
    for url in root.findall("s:url", NS):
        out.append({
            "loc": url.findtext("s:loc", default="", namespaces=NS),
            "title": url.findtext("n:news/n:title", default="", namespaces=NS).strip(),
            "date": url.findtext("n:news/n:publication_date", default="", namespaces=NS)[:10],
            "access": url.findtext("n:news/n:access", default="", namespaces=NS),
        })
    return out


class CitizenScraper(Scraper):
    """The Citizen headlines (title/date/URL only) matching energy topics."""

    name = "citizen"

    def list_documents(self) -> list[DocumentRef]:
        return [DocumentRef(self.name, SITEMAP, "news sitemap", "sitemap", None, SITEMAP)]

    def download(self, refs: list[DocumentRef]) -> list[StoredDoc]:
        stored = []
        for ref in refs:
            try:
                stored.append(store_raw(self.name, ref, self.client.get(ref.url, use_cache=False),
                                        self.raw_dir))
            except (requests.RequestException, RobotsDisallowed) as exc:
                self._fail(ref.url, exc)
        return stored

    def parse(self, docs: list[StoredDoc]) -> dict[str, list[Record]]:
        records: dict[str, Record] = {}
        for path in sorted((self.raw_dir / self.name).glob("*/sitemap.xml")):  # all runs so far
            retrieved = path.parent.name
            for item in parse_sitemap(path.read_bytes()):
                topics = match_topics(item["title"])
                if topics and item["loc"] not in records:
                    records[item["loc"]] = article_record(
                        "The Citizen", item["loc"], item["title"], item["date"] or None, retrieved,
                        "", topics, access=item["access"], text_collected="no (headline only)")
        log.info("%d relevant Citizen headlines across stored sitemaps", len(records))
        return {"news_citizen": list(records.values())}
